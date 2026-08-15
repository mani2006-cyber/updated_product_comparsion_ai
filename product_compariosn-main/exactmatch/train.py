"""
exactmatch/train.py
====================
Standalone binary same/different training loop. Same recipe that produced
v7-v11 (class-weighted CrossEntropyLoss, AdamW, linear warmup+decay, fp16,
early stopping on val loss) written fresh against exactmatch/'s own
dataset.py and model.py -- no LoRA branch, no 5-class path, no schema
detection.

    python -m exactmatch.train

Reads data/real_corpus_train.csv / real_corpus_valid.csv (build these
first with the existing build_real_corpus.py [+ mine_hard_negatives.py]
at the repo root -- this package does not regenerate them). Exports the
best checkpoint to exactmatch/trained_model/.
"""
import argparse
import os
import sys

# Under Kaggle's `!python`/subprocess stdout capture, tqdm can't rewrite its
# line, so every step becomes a new line and the real per-epoch summaries get
# buried in megabytes of progress-bar text (trap 6.4 in the handover -- other
# driver scripts in this repo already guard against it; this one didn't, and
# the first exactmatch training run produced a 3.9MB log because of it).
# Off by default; --progress re-enables it for interactive use.
if "--progress" not in sys.argv:
    os.environ["TQDM_DISABLE"] = "1"

import torch
import torch.nn as nn
from accelerate import Accelerator
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_scheduler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import compute_metrics, save_training_curves  # noqa: E402
from utils import EarlyStopping, ETATracker, get_logger, load_checkpoint, save_checkpoint, set_seed  # noqa: E402

import exactmatch.config as config  # noqa: E402
from exactmatch.dataset import build_dataloaders  # noqa: E402
from exactmatch.model import load_model_and_tokenizer  # noqa: E402
from exactmatch.save_model import export_trained_model  # noqa: E402

logger = get_logger("exactmatch")


def run_evaluation(model, dataloader, accelerator, loss_fn):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            logits = outputs.logits
            loss = loss_fn(logits, batch["labels"])

            preds = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

            g_labels, g_preds, g_probs = accelerator.gather_for_metrics(
                (batch["labels"], preds, probs))
            all_labels.extend(g_labels.cpu().tolist())
            all_preds.extend(g_preds.cpu().tolist())
            all_probs.extend(g_probs.cpu().tolist())

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, compute_metrics(all_labels, all_preds, all_probs)


def train():
    set_seed(config.RANDOM_SEED)

    accelerator = Accelerator(
        mixed_precision=config.MIXED_PRECISION,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
    )
    logger.info(f"Using device: {accelerator.device} | mixed_precision={config.MIXED_PRECISION} "
                f"| seed={config.RANDOM_SEED}")

    model, tokenizer = load_model_and_tokenizer()
    train_loader, val_loader, info = build_dataloaders(tokenizer)
    logger.info(f"train={info['train_n']:,} (pos={info['train_pos']:,}, "
                f"{info['train_pos'] / info['train_n']:.1%})  "
                f"valid={info['valid_n']:,} (pos={info['valid_pos']:,}, "
                f"{info['valid_pos'] / info['valid_n']:.1%})")

    # Class-weighted CrossEntropyLoss, capped at 5x -- same guard v7-v11 used
    # against extreme weights destabilizing training on an imbalanced corpus.
    import pandas as pd
    train_df = pd.read_csv(config.TRAIN_CSV)
    counts = train_df["label"].value_counts().to_dict()
    total = sum(counts.values())
    raw_weights = [total / (config.NUM_LABELS * counts.get(i, 1)) for i in range(config.NUM_LABELS)]
    min_w = min(raw_weights)
    capped = [min(w, min_w * 5.0) for w in raw_weights]
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(capped, dtype=torch.float).to(accelerator.device))

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    steps_per_epoch = (len(train_loader) + config.GRADIENT_ACCUMULATION_STEPS - 1) // config.GRADIENT_ACCUMULATION_STEPS
    total_steps = steps_per_epoch * config.NUM_EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    lr_scheduler = get_scheduler(config.LR_SCHEDULER_TYPE, optimizer,
                                 num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler)

    early_stopper = EarlyStopping(patience=config.EARLY_STOPPING_PATIENCE,
                                  min_delta=config.EARLY_STOPPING_MIN_DELTA, mode="min")
    history = {"train_loss": [], "val_loss": [], "accuracy": [], "precision": [], "recall": [], "f1": [], "roc_auc": []}
    eta_tracker = ETATracker(total_steps=total_steps)
    best_f1, best_epoch, global_step = -1.0, 0, 0

    logger.info(f"Starting training | train={info['train_n']} val={info['valid_n']} | "
                f"epochs={config.NUM_EPOCHS} | effective batch size="
                f"{config.TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        running_loss, n_steps = 0.0, 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}",
                  disable=not accelerator.is_local_main_process)

        for batch in bar:
            with accelerator.accumulate(model):
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
            bar.set_postfix(loss=f"{running_loss / n_steps:.4f}",
                           lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                           eta=eta_str if accelerator.sync_gradients else "-")

        train_loss = running_loss / max(n_steps, 1)
        val_loss, val_metrics = run_evaluation(model, val_loader, accelerator, loss_fn)

        for k in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            history[k].append(val_metrics[k])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if accelerator.is_local_main_process:
            logger.info(f"Epoch {epoch}/{config.NUM_EPOCHS} | train_loss={train_loss:.4f} | "
                       f"val_loss={val_loss:.4f} | lr={lr_scheduler.get_last_lr()[0]:.2e} | "
                       f"accuracy={val_metrics['accuracy']:.4f} | precision={val_metrics['precision']:.4f} | "
                       f"recall={val_metrics['recall']:.4f} | f1={val_metrics['f1']:.4f} | "
                       f"roc_auc={val_metrics['roc_auc']:.4f}")

        if val_metrics["f1"] > best_f1:
            best_f1, best_epoch = val_metrics["f1"], epoch
            if accelerator.is_local_main_process:
                unwrapped = accelerator.unwrap_model(model)
                save_checkpoint({"model_state": unwrapped.state_dict(), "epoch": epoch, "f1": best_f1},
                               os.path.join(config.CHECKPOINT_DIR, "best_model.pt"))
                logger.info(f"  -> New best model (f1={best_f1:.4f}), checkpoint saved.")

        early_stopper.step(val_loss)
        if early_stopper.should_stop:
            logger.info(f"Early stopping triggered after epoch {epoch} "
                       f"(no val_loss improvement for {config.EARLY_STOPPING_PATIENCE} epochs).")
            break

    if accelerator.is_local_main_process:
        save_training_curves(history, config.PLOTS_DIR)
        best = load_checkpoint(os.path.join(config.CHECKPOINT_DIR, "best_model.pt"))
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.load_state_dict(best["model_state"])
        export_trained_model(unwrapped, tokenizer, info)
        logger.info(f"Best model (epoch {best_epoch}, f1={best_f1:.4f}) exported to {config.TRAINED_MODEL_DIR}")


if __name__ == "__main__":
    train()
