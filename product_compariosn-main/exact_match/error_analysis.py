"""
exact_match/error_analysis.py
==============================
Step 22 (extension of evaluate.py): per-example error analysis.

evaluate.py only reports aggregate metrics. This script runs the trained
model over a data split and saves, for EVERY row:
  - the full input text (text_a, text_b -- title/brand/detail, as the
    model actually sees it after preprocessing.build_product_text)
  - brand_a / brand_b (parsed back out of text_a/text_b -- see note below)
  - true label / predicted label (class names, not just ids)
  - the full per-class probability distribution
  - confidence (top-1 probability) and margin (top-1 minus top-2 prob)
  - correct/incorrect flag
  - a binary MATCH/NON-MATCH confusion tag (TP/TN/FP/FN), where MATCH is
    defined as predicted/true class == --match_class (default
    EXACT_MATCH, matching the existing binary-compat flag in
    inference.py's ComparisonResult.label)

Outputs, under config.REPORTS_DIR:
  - error_analysis_<split>.csv   every row, for spreadsheet/Excel review
  - error_report_<split>.txt     misclassified rows only, human-readable

Also logs to console/run.log:
  - binary MATCH confusion counts
  - per-true-class accuracy (this is where SIMILAR_ALTERNATIVE /
    WEAKLY_SIMILAR show up as the weak classes)
  - count of high-confidence (>0.9) WRONG predictions -- these usually
    point at a labeling problem rather than a model problem
  - the 20 lowest-accuracy brands (n>=5 rows), as a proxy for
    category/brand-level performance since the current schema has no
    separate product-category column

NOTE on brand parsing: preprocessing.clean_dataframe() collapses
title/brand/description into a single "text_a"/"text_b" string via
build_product_text() and does not keep the original product1_brand /
product2_brand columns downstream. Rather than modify preprocessing.py
(out of scope for this file), brand is parsed back out of the
"| brand X |" segment that build_product_text() already writes into
text_a/text_b. If a row has no brand, brand shows as "unknown".

Does not change train.py, evaluate.py, dataset.py, or preprocessing.py.
Read-only consumer of the existing pipeline.

Run with:
    python -m exact_match.error_analysis --split test
    python -m exact_match.error_analysis --split test --data data/relationship_pairs_final.csv --num_labels 5
"""

import argparse
import os
import re

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from . import preprocessing
from .dataset import build_dataloader
from utils import get_logger

logger = get_logger(__name__)

_BRAND_RE = re.compile(r"\|\s*brand\s+([^|]+?)\s*(?:\||$)")


def _extract_brand(text: str) -> str:
    m = _BRAND_RE.search(text)
    return m.group(1).strip() if m else "unknown"


@torch.no_grad()
def run_error_analysis(
    model,
    tokenizer,
    df: pd.DataFrame,
    device,
    id2label: dict,
    match_class: str = "EXACT_MATCH",
    batch_size: int = 32,
) -> pd.DataFrame:
    model.eval()
    dataloader = build_dataloader(df, tokenizer, batch_size=batch_size, shuffle=False)
    label2id = {v: k for k, v in id2label.items()}
    if match_class not in label2id:
        raise ValueError(f"--match_class '{match_class}' is not one of {list(label2id)}")
    match_id = label2id[match_class]

    rows = []
    idx = 0
    for batch in tqdm(dataloader, desc="Running inference for error analysis"):
        batch_gpu = {k: v.to(device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch_gpu["input_ids"],
            attention_mask=batch_gpu["attention_mask"],
            token_type_ids=batch_gpu.get("token_type_ids"),
        )
        probs = torch.softmax(outputs.logits, dim=-1).cpu()
        preds = torch.argmax(probs, dim=-1)
        k = min(2, probs.shape[1])
        top_vals = torch.topk(probs, k=k, dim=-1).values

        bsz = probs.shape[0]
        for i in range(bsz):
            row = df.iloc[idx]
            true_id = int(row["label"])
            pred_id = int(preds[i].item())
            prob_row = probs[i].tolist()
            confidence = prob_row[pred_id]
            margin = float(top_vals[i, 0] - top_vals[i, 1]) if k > 1 else confidence

            true_is_match = true_id == match_id
            pred_is_match = pred_id == match_id
            if true_is_match and pred_is_match:
                confusion_tag = "TP"
            elif (not true_is_match) and (not pred_is_match):
                confusion_tag = "TN"
            elif (not true_is_match) and pred_is_match:
                confusion_tag = "FP"
            else:
                confusion_tag = "FN"

            record = {
                "text_a": row["text_a"],
                "text_b": row["text_b"],
                "brand_a": _extract_brand(row["text_a"]),
                "brand_b": _extract_brand(row["text_b"]),
                "true_label": id2label[true_id],
                "pred_label": id2label[pred_id],
                "correct": true_id == pred_id,
                "confidence": confidence,
                "margin": margin,
                "match_confusion": confusion_tag,
            }
            for c in range(len(prob_row)):
                record[f"prob_{id2label[c]}"] = prob_row[c]
            rows.append(record)
            idx += 1

    return pd.DataFrame(rows)


def _why_failed(row: pd.Series) -> str:
    if row["confidence"] > 0.9:
        return ("high-confidence WRONG prediction -- check whether the ground-truth "
                "label itself is correct before assuming the model is at fault")
    if row["margin"] < 0.15:
        return "low margin between top two classes -- genuinely ambiguous boundary case"
    return "moderate-confidence miss -- inspect feature/text overlap between the two products"


def _write_error_report(df_err: pd.DataFrame, path: str, max_rows: int = 300) -> None:
    with open(path, "w", encoding="utf-8") as f:
        header = f"ERROR REPORT -- {len(df_err)} misclassified rows"
        if len(df_err) > max_rows:
            header += f" (showing worst {max_rows} by confidence)"
        f.write(header + "\n" + "=" * 70 + "\n\n")

        prob_cols = [c for c in df_err.columns if c.startswith("prob_")]
        for _, row in df_err.head(max_rows).iterrows():
            f.write(f"Product A: {row['text_a']}\n")
            f.write(f"Product B: {row['text_b']}\n\n")
            f.write(f"Expected:  {row['true_label']}\n")
            f.write(f"Predicted: {row['pred_label']}\n\n")
            f.write(f"Prediction Score: {row['confidence']:.4f}  "
                    f"(margin over 2nd class: {row['margin']:.4f})\n\n")
            f.write(f"Why the model may have failed: {_why_failed(row)}\n\n")
            f.write("Relevant feature values (class probabilities):\n")
            for c in prob_cols:
                f.write(f"  {c.replace('prob_', '')}: {row[c]:.4f}\n")
            f.write("\n" + "-" * 70 + "\n\n")


def main():
    parser = argparse.ArgumentParser(description="Per-example error analysis for the trained model.")
    parser.add_argument("--data", default=config.RAW_DATA_PATH)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model_dir", default=config.TRAINED_MODEL_DIR)
    parser.add_argument("--num_labels", type=int, default=config.NUM_LABELS,
                         help="2 for the binary exact-match model, 5 for the relationship model.")
    parser.add_argument("--match_class", default="EXACT_MATCH",
                         help="Class name treated as MATCH for the binary TP/FP/FN/TN tag.")
    parser.add_argument("--out_dir", default=config.REPORTS_DIR)
    args = parser.parse_args()
    config.NUM_LABELS = args.num_labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {args.model_dir} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}

    train_df, val_df, test_df = preprocessing.load_clean_split(args.data)
    split_map = {"train": train_df, "val": val_df, "test": test_df}
    eval_df = split_map[args.split]
    logger.info(f"Running error analysis on '{args.split}' split ({len(eval_df)} rows)")

    df_all = run_error_analysis(model, tokenizer, eval_df, device, id2label, match_class=args.match_class)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"error_analysis_{args.split}.csv")
    df_all.to_csv(csv_path, index=False)
    logger.info(f"Saved full per-row predictions to {csv_path}")

    df_err = df_all[~df_all["correct"]].sort_values("confidence", ascending=False)
    txt_path = os.path.join(args.out_dir, f"error_report_{args.split}.txt")
    _write_error_report(df_err, txt_path)
    logger.info(f"Saved human-readable error report to {txt_path} ({len(df_err)} misclassified rows)")

    counts = df_all["match_confusion"].value_counts().to_dict()
    logger.info(f"Binary MATCH confusion (match == '{args.match_class}' only): {counts}")

    high_conf_wrong = df_err[df_err["confidence"] > 0.9]
    logger.info(f"High-confidence (>0.9) WRONG predictions: {len(high_conf_wrong)} / {len(df_err)} total errors")

    per_class = df_all.groupby("true_label")["correct"].agg(["mean", "count"]).rename(
        columns={"mean": "accuracy", "count": "n"}
    ).sort_values("accuracy")
    logger.info("Per-class accuracy (lowest first):\n" + per_class.to_string())

    brand_acc = df_all.groupby("brand_a")["correct"].agg(["mean", "count"]).rename(
        columns={"mean": "accuracy", "count": "n"}
    )
    brand_acc = brand_acc[brand_acc["n"] >= 5].sort_values("accuracy").head(20)
    logger.info("20 lowest-accuracy brands (n>=5 rows):\n" + brand_acc.to_string())


if __name__ == "__main__":
    main()