"""
train.py
========
Orchestrates the full training pipeline end to end:

  1. Load dataset            -> preprocessing.load_raw_data
  2. Clean data               -> preprocessing.clean_dataframe
  5. Train/Val/Test split     -> preprocessing.split_data
  3. Tokenization              \
  4. Preprocessing into tensors > dataset.build_all_dataloaders
  6. DataLoader creation       /
  7. Model loading            -> model.load_model_and_tokenizer
  8. Fine-tuning               \
  9. Loss function              |
 10. Optimizer                  |  this file's train() loop
 11. LR scheduler                > (Accelerate handles device
 12. Mixed precision            |   placement + AMP)
 13. Gradient accumulation     /
 14. Early stopping          -> utils.EarlyStopping
 15. Checkpoint saving       -> utils.save_checkpoint
 16-21. Evaluation & metrics -> metrics.compute_metrics
 22. Save best model         -> save_model.export_trained_model
 23. Export tokenizer        -> save_model.export_trained_model

Run with:
    python train.py
    accelerate launch train.py        # multi-GPU / better AMP handling
"""

import os
import time

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_scheduler

import config
from . import preprocessing
from .dataset import build_all_dataloaders
from metrics import compute_metrics, save_training_curves
from .model import load_model_and_tokenizer
from .save_model import export_trained_model
from utils import EarlyStopping, ETATracker, get_logger, save_checkpoint, set_seed

logger = get_logger(__name__)


@torch.no_grad()
def run_evaluation(model, dataloader, accelerator, loss_fn):
    """Step 16: Evaluation loop. Returns avg loss + full metric dict + raw predictions."""
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_labels, all_preds, all_probs = [], [], []

    for batch in dataloader:
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        logits = outputs.logits
        loss = loss_fn(logits, batch["labels"])

        preds = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)[:, 1]

        gathered_labels, gathered_preds, gathered_probs = accelerator.gather_for_metrics(
            (batch["labels"], preds, probs)
        )
        all_labels.extend(gathered_labels.cpu().tolist())
        all_preds.extend(gathered_preds.cpu().tolist())
        all_probs.extend(gathered_probs.cpu().tolist())

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics, (all_labels, all_preds, all_probs)


def train():
    set_seed(config.RANDOM_SEED)

    accelerator = Accelerator(
        mixed_precision=config.MIXED_PRECISION,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
    )
    logger.info(f"Using device: {accelerator.device} | mixed_precision={config.MIXED_PRECISION}")

    # ---- Steps 1, 2, 5: load -> clean -> split -------------------------
    train_df, val_df, test_df = preprocessing.load_clean_split(config.RAW_DATA_PATH)

    # ---- Step 7: model + tokenizer --------------------------------------
    model, tokenizer = load_model_and_tokenizer()

    # ---- Steps 3, 4, 6: tokenize + DataLoaders --------------------------
    train_loader, val_loader, test_loader = build_all_dataloaders(train_df, val_df, test_df, tokenizer)

    # ---- Step 9: loss function -------------------------------------------
    # CrossEntropyLoss with class weights to counter label imbalance
    # (product-matching datasets are frequently imbalanced towards
    # "different product" negatives).
    label_counts = train_df["label"].value_counts().to_dict()
    total = sum(label_counts.values())
    class_weights = torch.tensor(
        [total / (2 * label_counts.get(0, 1)), total / (2 * label_counts.get(1, 1))],
        dtype=torch.float,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(accelerator.device))

    # ---- Step 10: optimizer ----------------------------------------------
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # ---- Step 11: LR scheduler --------------------------------------------
    steps_per_epoch = (len(train_loader) + config.GRADIENT_ACCUMULATION_STEPS - 1) // config.GRADIENT_ACCUMULATION_STEPS
    total_training_steps = steps_per_epoch * config.NUM_EPOCHS
    warmup_steps = int(total_training_steps * config.WARMUP_RATIO)
    lr_scheduler = get_scheduler(
        name=config.LR_SCHEDULER_TYPE,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    # ---- Step 12: mixed precision + device placement (via Accelerate) ---
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )
    if test_loader is not None:
        test_loader = accelerator.prepare(test_loader)

    # ---- Step 14: early stopping ------------------------------------------
    early_stopper = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        min_delta=config.EARLY_STOPPING_MIN_DELTA,
        mode="min",
    )

    history = {"train_loss": [], "val_loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
    eta_tracker = ETATracker(total_steps=total_training_steps)
    best_metric_value = -1.0
    global_step = 0

    logger.info(
        f"Starting training | train={len(train_df)} val={len(val_df)} test={len(test_df)} | "
        f"epochs={config.NUM_EPOCHS} | effective batch size="
        f"{config.TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}"
    )

    for epoch in range(1, config.NUM_EPOCHS + 1):
        # ---- Step 8: fine-tuning (training loop for this epoch) --------
        model.train()
        running_loss, n_steps = 0.0, 0
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{config.NUM_EPOCHS}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in progress_bar:
            with accelerator.accumulate(model):  # Step 13: gradient accumulation
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                )
                loss = loss_fn(outputs.logits, batch["labels"])

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                eta_str = eta_tracker.step(global_step)

            running_loss += loss.item()
            n_steps += 1
            current_lr = lr_scheduler.get_last_lr()[0]

            progress_bar.set_postfix(
                loss=f"{running_loss / n_steps:.4f}",
                lr=f"{current_lr:.2e}",
                eta=eta_str if accelerator.sync_gradients else "-",
            )

        train_loss = running_loss / max(n_steps, 1)

        # ---- Steps 16-21: evaluation ------------------------------------
        val_loss, val_metrics, _ = run_evaluation(model, val_loader, accelerator, loss_fn)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["accuracy"].append(val_metrics["accuracy"])
        history["precision"].append(val_metrics["precision"])
        history["recall"].append(val_metrics["recall"])
        history["f1"].append(val_metrics["f1"])
        history["roc_auc"].append(val_metrics["roc_auc"])

        if accelerator.is_local_main_process:
            logger.info(
                f"Epoch {epoch}/{config.NUM_EPOCHS} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"lr={lr_scheduler.get_last_lr()[0]:.2e} | "
                f"accuracy={val_metrics['accuracy']:.4f} | "
                f"precision={val_metrics['precision']:.4f} | "
                f"recall={val_metrics['recall']:.4f} | "
                f"f1={val_metrics['f1']:.4f} | "
                f"roc_auc={val_metrics['roc_auc']:.4f}"
            )

        # ---- Step 15: checkpoint saving -----------------------------------
        current_metric_value = val_metrics[config.PRIMARY_METRIC]
        is_best = current_metric_value > best_metric_value

        if accelerator.is_local_main_process:
            if config.SAVE_EVERY_EPOCH:
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": unwrapped.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    os.path.join(config.CHECKPOINT_DIR, "last_epoch.pt"),
                )

            if is_best:
                best_metric_value = current_metric_value
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": unwrapped.state_dict(),
                        "val_metrics": val_metrics,
                    },
                    config.BEST_CHECKPOINT_PATH,
                )
                logger.info(f"  -> New best model ({config.PRIMARY_METRIC}={current_metric_value:.4f}), checkpoint saved.")

        # ---- Step 14: early stopping check (on val_loss) -----------------
        early_stopper.step(val_loss)
        if early_stopper.should_stop:
            logger.info(f"Early stopping triggered after epoch {epoch} "
                        f"(no val_loss improvement for {config.EARLY_STOPPING_PATIENCE} epochs).")
            break

    # ---- Plots: loss / accuracy / F1 graphs -------------------------------
    if accelerator.is_local_main_process:
        saved_plots = save_training_curves(history)
        logger.info(f"Saved training curves: {saved_plots}")

        # ---- Steps 22-23: save best model + tokenizer ----------------------
        unwrapped_model = accelerator.unwrap_model(model)
        best_state = torch.load(config.BEST_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        unwrapped_model.load_state_dict(best_state["model_state_dict"])
        export_trained_model(unwrapped_model, tokenizer)
        logger.info(f"Best model (epoch {best_state['epoch']}, "
                    f"{config.PRIMARY_METRIC}={best_state['val_metrics'][config.PRIMARY_METRIC]:.4f}) "
                    f"exported to {config.TRAINED_MODEL_DIR}")

    return history


if __name__ == "__main__":
    start = time.time()
    train()
    logger.info(f"Total training time: {time.time() - start:.1f}s")