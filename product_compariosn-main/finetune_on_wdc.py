"""
finetune_on_wdc.py
==================
Fine-tunes the existing DeBERTa cross-encoder on WDC Products' REAL,
human-labelled training pairs, then scores the three gold standards --
producing the first number in this project directly comparable to the
published RoBERTa / Ditto results.

WHY
---
Zero-shot, this model scored F1 45.9 / 45.3 / 46.3 on SEEN / HALF-SEEN /
UNSEEN against supervised RoBERTa 78.58 / 75.91 / 71.14. A trivial all-match
baseline scores ~20, so there is real signal but only about half the distance
to SOTA. That gap has two possible causes demanding opposite responses:

    (a) the synthetic training data was the bottleneck  -> fix the data
    (b) something in this setup underperforms          -> fix the model

Training the SAME architecture on real labelled pairs separates them.

MATCHING THE PUBLISHED SETUP
----------------------------
Defaults here follow the reference command in the benchmark repo's README
(`run_finetune_baseline.sh roberta-base True 64 5e-05 ... large`) and
`run_finetune_baseline.sh` itself, NOT this project's config.py, because a
comparison run at different hyperparameters measures the hyperparameters:

    learning rate 5e-5   (config.py uses 2e-5)
    effective batch 64   (config.py uses 32)
    warmup ratio 0.05    (config.py uses 0.06)
    weight decay 0.01, max_grad_norm 1.0, linear schedule, fp16  -- already match

Batch 64 is reached as 32 x 2 gradient-accumulation steps so it fits a T4;
the reference used 64 directly with gradient checkpointing.

SERIALIZATION
-------------
The published baselines feed the model Ditto-style tagged text:

    [COL] brand [VAL] <5 words> [COL] title [VAL] <50 words>
    [COL] description [VAL] <100 words>

while this project feeds `title | brand X | description`. That is a real
confound, so both are supported via --serialization and the choice is applied
identically to training and evaluation. Running both answers a question from
the earlier research review: does tagged serialization actually help us?

DESIGN
------
`exact_match.train.train()` is reused UNMODIFIED. Two things are arranged
around it rather than inside it:

1. config is overridden BEFORE `exact_match` is imported. model.py and
   save_model.py read config values as DEFAULT ARGUMENT VALUES, which Python
   binds once at import; setting them afterwards is silently ignored and
   yields a 5-class model by accident.

2. `preprocessing.load_clean_split` is replaced with one returning WDC's OWN
   train/valid frames. train() would otherwise re-split with our entity-level
   splitter, destroying the seen/unseen structure that makes the comparison
   meaningful -- WDC's splits ARE the benchmark.

The 5-class model in trained_model/ is untouched; output goes to
trained_model_wdc/.

Usage:
    python finetune_on_wdc.py --wdc /kaggle/working/wdc
    python finetune_on_wdc.py --wdc /kaggle/working/wdc --serialization pipeline
"""

import argparse
import glob
import gzip
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def _read_json_rows(path: str) -> List[dict]:
    """Reads JSON-lines, gzipped or not (Kaggle decompresses on upload)."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        text = fh.read()
    try:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    except json.JSONDecodeError:
        rows = json.loads(text)
        return rows if isinstance(rows, list) else [rows]


def _find(work: str, suffix: str) -> str:
    hits = sorted(glob.glob(os.path.join(work, "**", f"*{suffix}.json*"), recursive=True))
    if not hits:
        raise SystemExit(f"Could not find a '*{suffix}.json[.gz]' file under {work}")
    return hits[0]


def _words(value, limit: int) -> str:
    return " ".join(str(value or "").split(" ")[:limit]).strip()


def serialize_wdc(row: dict, side: str) -> str:
    """Ditto-style tagged serialization, matching the published baselines.

    Field order and per-field word limits are copied from
    src/contrastive/data/datasets.py::serialize_sample_lspc_contrastive so the
    model sees what RoBERTa/Ditto saw. price is omitted, matching the
    baseline's use_price=False default.
    """
    return (
        f"[COL] brand [VAL] {_words(row.get(f'brand_{side}'), 5)} "
        f"[COL] title [VAL] {_words(row.get(f'title_{side}'), 50)} "
        f"[COL] description [VAL] {_words(row.get(f'description_{side}'), 100)}"
    ).strip()


def make_texts(rows: List[dict], mode: str) -> Tuple[List[str], List[str]]:
    """Serializes both sides. Used for train, valid AND test, so training and
    evaluation can never silently disagree on input format."""
    if mode == "wdc":
        return ([serialize_wdc(r, "left") for r in rows],
                [serialize_wdc(r, "right") for r in rows])

    from exact_match.preprocessing import build_product_text
    return (
        [build_product_text(r.get("title_left") or "", brand=r.get("brand_left") or "",
                            description=r.get("description_left") or "") for r in rows],
        [build_product_text(r.get("title_right") or "", brand=r.get("brand_right") or "",
                            description=r.get("description_right") or "") for r in rows],
    )


def to_frame(rows: List[dict], mode: str) -> pd.DataFrame:
    text_a, text_b = make_texts(rows, mode)
    return pd.DataFrame({"text_a": text_a, "text_b": text_b,
                         "label": [int(r["label"]) for r in rows]})


def load_wdc_training(work: str, size: str, mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads WDC's own train/valid splits.

    Only the 000un variant ships train/valid files: the training set is the
    same regardless of how unseen the *test* set is, which is the benchmark's
    design.
    """
    train_df = to_frame(_read_json_rows(_find(work, f"_train_{size}")), mode)
    val_df = to_frame(_read_json_rows(_find(work, f"_valid_{size}")), mode)
    for name, df in (("train", train_df), ("valid", val_df)):
        pos = int(df["label"].sum())
        print(f"  WDC {name}_{size}: {len(df):,} pairs | positives {pos:,} ({pos / len(df):.1%})")
    return train_df, val_df


def main():
    ap = argparse.ArgumentParser(description="Fine-tune on WDC Products and score the gold standards.")
    ap.add_argument("--wdc", required=True, help="Folder holding the WDC json files")
    ap.add_argument("--size", default="medium", choices=["small", "medium", "large"],
                    help="WDC training set size. 'medium' matches the published Table 3 row.")
    ap.add_argument("--serialization", default="wdc", choices=["wdc", "pipeline"],
                    help="'wdc' = [COL]/[VAL] tagged (what the baselines used); "
                         "'pipeline' = this project's build_product_text.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5, help="reference baseline uses 5e-5")
    ap.add_argument("--batch_size", type=int, default=32, help="x2 accumulation = effective 64")
    ap.add_argument("--out_dir", default="trained_model_wdc")
    args = ap.parse_args()

    # ---- config overrides MUST happen before exact_match is imported ------
    import config
    config.NUM_LABELS = 2                       # WDC is binary match / non-match
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.TRAIN_BATCH_SIZE = args.batch_size
    config.GRADIENT_ACCUMULATION_STEPS = 2      # effective batch 64, as published
    config.WARMUP_RATIO = 0.05
    config.TRAINED_MODEL_DIR = os.path.abspath(args.out_dir)
    config.CHECKPOINT_DIR = os.path.join(config.OUTPUT_DIR, "checkpoints_wdc")
    config.BEST_CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_DIR, "best_model_wdc.pt")
    for d in (config.TRAINED_MODEL_DIR, config.CHECKPOINT_DIR):
        os.makedirs(d, exist_ok=True)

    print(f"serialization={args.serialization} | NUM_LABELS={config.NUM_LABELS} "
          f"| lr={config.LEARNING_RATE} | effective batch="
          f"{config.TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS} "
          f"| epochs={config.NUM_EPOCHS}")

    print("\nLoading WDC training data ...")
    train_df, val_df = load_wdc_training(args.wdc, args.size, args.serialization)
    print(f"  example text_a: {train_df.text_a.iloc[0][:130]}")

    from exact_match import preprocessing

    def _fixed_splits(_path=None):
        return train_df, val_df, val_df

    preprocessing.load_clean_split = _fixed_splits

    from exact_match.train import train
    print("\nFine-tuning (existing exact_match/train.py, unmodified) ...")
    train()

    # ---- score the three gold standards ---------------------------------
    import torch
    from sklearn.metrics import precision_recall_fscore_support
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from evaluate_on_wdc import PUBLISHED_F1, load_wdc_local, predict_probabilities, sweep_threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.TRAINED_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(config.TRAINED_MODEL_DIR).to(device)
    print(f"\nScoring gold standards with {config.TRAINED_MODEL_DIR}")

    splits = load_wdc_local(args.wdc)
    order = [t for t in ("SEEN", "HALF-SEEN", "UNSEEN") if t in splits]
    results = {}

    for tag in order:
        rows = splits[tag]
        y_true = np.array([r["label"] for r in rows], dtype=int)
        # Same serialization as training -- otherwise the model is graded on
        # an input format it never saw.
        text_a, text_b = make_texts(rows, args.serialization)
        probs = predict_probabilities(model, tokenizer, text_a, text_b, device, 64)
        y_pred = probs.argmax(axis=1)          # binary head: class 1 = match
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                     zero_division=0)
        best_f1, best_t = sweep_threshold(y_true, probs[:, 1])

        hard = np.array([bool(x.get("is_hard_negative")) for x in rows]) & (y_true == 0)
        fp = float(y_pred[hard].mean()) if hard.any() else float("nan")

        print(f"\n{tag}: precision {p:.4f} | recall {r:.4f} | F1 {f:.4f}")
        print(f"  best F1 over P(match) sweep: {best_f1:.4f} at threshold {best_t:.2f}")
        print(f"  false-positive rate on {int(hard.sum()):,} HARD negatives: {fp:.1%}")
        results[tag] = (f, best_f1)

    print("\n" + "=" * 76)
    print("SUMMARY -- F1 x 100")
    print("=" * 76)
    print(f"{'system':<32}" + "".join(f"{c:>13}" for c in order))
    print(f"{'THIS MODEL (argmax)':<32}" + "".join(f"{results[c][0] * 100:>13.2f}" for c in order))
    print(f"{'THIS MODEL (best threshold)':<32}" + "".join(f"{results[c][1] * 100:>13.2f}" for c in order))
    print(f"{'  (was, 0-shot synthetic)':<32}{45.89:>13.2f}{45.31:>13.2f}{46.31:>13.2f}")
    for sysname, byt in PUBLISHED_F1.items():
        print(f"{sysname + ' (published)':<32}" + "".join(f"{byt[c]:>13.2f}" for c in order))

    print("\n" + "=" * 76)
    print("HOW TO READ THIS")
    print("=" * 76)
    print("This row IS comparable to the published ones: same benchmark, same")
    print("splits, same supervised setting, matched hyperparameters.")
    print("  ~70-79  -> architecture is at SOTA. The synthetic data was the")
    print("            entire bottleneck -- invest in data, not models.")
    print("  ~55-65  -> something in this setup underperforms; only NOW is it")
    print("            worth investigating pooling / loss / augmentation.")
    print("  <50     -> suspect a pipeline bug before any modelling change.")
    print("\nNote: published rows used roberta-base; this uses deberta-v3-small")
    print("(~1/3 fewer params), so a few points below RoBERTa is unremarkable.")


if __name__ == "__main__":
    main()
