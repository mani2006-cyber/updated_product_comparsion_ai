"""
evaluate.py
===========
Standalone evaluation script. Loads the exported `trained_model/`
directory and runs it over the held-out test split, producing:
  - accuracy / precision / recall / F1 / ROC-AUC
  - confusion matrix image
  - classification report (text + saved to file)

Run with:
    python evaluate.py
    python evaluate.py --data data/products.csv --split test
"""

import argparse
import os

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from . import preprocessing
from .dataset import build_dataloader
from metrics import compute_metrics, get_classification_report, save_confusion_matrix
from utils import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def evaluate_model(model, dataloader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    all_labels, all_preds, all_probs = [], [], []
    total_loss, n_batches = 0.0, 0

    for batch in tqdm(dataloader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
        )
        loss = loss_fn(outputs.logits, batch["labels"])
        preds = torch.argmax(outputs.logits, dim=-1)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]

        all_labels.extend(batch["labels"].cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics, all_labels, all_preds


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained product-comparison model.")
    parser.add_argument("--data", default=config.RAW_DATA_PATH, help="Path to CSV/JSON dataset")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model_dir", default=config.TRAINED_MODEL_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {args.model_dir} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    train_df, val_df, test_df = preprocessing.load_clean_split(args.data)
    split_map = {"train": train_df, "val": val_df, "test": test_df}
    eval_df = split_map[args.split]
    logger.info(f"Evaluating on '{args.split}' split ({len(eval_df)} rows)")

    dataloader = build_dataloader(eval_df, tokenizer, batch_size=config.EVAL_BATCH_SIZE, shuffle=False)
    avg_loss, metrics, labels, preds = evaluate_model(model, dataloader, device)

    logger.info(f"Loss: {avg_loss:.4f}")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")

    report = get_classification_report(labels, preds)
    logger.info("\n" + report)

    report_path = os.path.join(config.REPORTS_DIR, f"classification_report_{args.split}.txt")
    os.makedirs(config.REPORTS_DIR, exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"Loss: {avg_loss:.4f}\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n" + report)
    logger.info(f"Saved classification report to {report_path}")

    cm_path = save_confusion_matrix(
        labels, preds, save_path=os.path.join(config.PLOTS_DIR, f"confusion_matrix_{args.split}.png")
    )
    logger.info(f"Saved confusion matrix to {cm_path}")


if __name__ == "__main__":
    main()
