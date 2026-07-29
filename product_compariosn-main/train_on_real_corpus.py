"""
train_on_real_corpus.py
=======================
Trains the existing DeBERTa cross-encoder on the merged HUMAN-LABELLED corpus
built by build_real_corpus.py, then evaluates on every benchmark test set
SEPARATELY so each number stays comparable to published results.

WHY
---
Measured on this project: the same model scored 45.9 F1 trained on
rule-generated labels and 75.6 trained on 6,000 human-labelled WDC pairs.
Data was the only bottleneck that ever mattered. This run uses ~18.5k human
labels (~38k with WDC), so it is the direct test of whether more real data
keeps buying accuracy.

EVALUATION IS PER BENCHMARK, NEVER POOLED
-----------------------------------------
Pooling four test sets would produce one number comparable to nothing. Each is
scored on its own and printed beside its published baseline where one exists.

WHAT IS REUSED, NOT REWRITTEN
-----------------------------
`exact_match.train.train()` runs unmodified. As in finetune_on_wdc.py, two
things are arranged around it:

  1. config is overridden BEFORE exact_match is imported -- model.py and
     save_model.py bind config values as DEFAULT ARGUMENTS at import time, so
     setting them afterwards is silently ignored and yields a 5-class model.
  2. preprocessing.load_clean_split is replaced, because the corpus already
     has its splits and re-splitting would destroy them.

Existing models in trained_model/ and trained_model_wdc/ are untouched;
output goes to trained_model_real/.

Usage:
    python train_on_real_corpus.py
    python train_on_real_corpus.py --wdc /kaggle/working/wdc --epochs 15
"""

import argparse
import glob
import os
from typing import Dict, List

import numpy as np
import pandas as pd

TRAIN_CSV = "data/real_corpus_train.csv"
VALID_CSV = "data/real_corpus_valid.csv"

# Published F1 on the ER-Magellan test sets (Ditto, Li et al. 2020, Table 4).
# Context only -- printed beside our numbers, never merged with them.
PUBLISHED_ER = {
    "Structured_Amazon-Google": ("Ditto", 75.58),
    "Structured_Walmart-Amazon": ("Ditto", 86.76),
    "Textual_Abt-Buy": ("Ditto", 89.33),
}


def main():
    ap = argparse.ArgumentParser(description="Train on the merged human-labelled corpus.")
    ap.add_argument("--train", default=TRAIN_CSV)
    ap.add_argument("--valid", default=VALID_CSV)
    ap.add_argument("--er-dir", default="data/er_magellan")
    ap.add_argument("--wdc", default=None, help="Folder with WDC json files (optional)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="trained_model_real")
    ap.add_argument("--progress", action="store_true",
                    help="Show the per-step bar. Off by default: under Kaggle's '!python' "
                         "it cannot rewrite its line and buries the epoch summaries.")
    args = ap.parse_args()

    for path in (args.train, args.valid):
        if not os.path.exists(path):
            raise SystemExit(f"Missing {path}. Run build_real_corpus.py first.")

    train_df = pd.read_csv(args.train)
    val_df = pd.read_csv(args.valid)
    for name, df in (("train", train_df), ("valid", val_df)):
        print(f"  {name}: {len(df):,} pairs | positives {int(df.label.sum()):,} "
              f"({df.label.mean():.1%})")
    print(f"  sources: {dict(train_df['source'].value_counts())}")

    # ---- config overrides MUST precede any exact_match import -------------
    import config
    config.NUM_LABELS = 2
    config.NUM_EPOCHS = args.epochs
    config.LEARNING_RATE = args.lr
    config.TRAIN_BATCH_SIZE = args.batch_size
    config.GRADIENT_ACCUMULATION_STEPS = 2       # effective batch 64, as published
    config.WARMUP_RATIO = 0.05
    config.TRAINED_MODEL_DIR = os.path.abspath(args.out_dir)
    config.CHECKPOINT_DIR = os.path.join(config.OUTPUT_DIR, "checkpoints_real")
    config.BEST_CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_DIR, "best_model_real.pt")
    for d in (config.TRAINED_MODEL_DIR, config.CHECKPOINT_DIR):
        os.makedirs(d, exist_ok=True)
    print(f"\nNUM_LABELS={config.NUM_LABELS} | lr={config.LEARNING_RATE} | "
          f"effective batch={config.TRAIN_BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS} | "
          f"epochs={config.NUM_EPOCHS} | out={config.TRAINED_MODEL_DIR}")

    from exact_match import preprocessing

    def _fixed_splits(_path=None):
        return (train_df[["text_a", "text_b", "label"]],
                val_df[["text_a", "text_b", "label"]],
                val_df[["text_a", "text_b", "label"]])

    preprocessing.load_clean_split = _fixed_splits

    import exact_match.train as train_module
    if not args.progress:
        _real = train_module.tqdm
        train_module.tqdm = lambda it=None, *a, **k: _real(it, *a, **{**k, "disable": True})
        import evaluate_on_wdc as _ev
        _real_ev = _ev.tqdm
        _ev.tqdm = lambda it=None, *a, **k: _real_ev(it, *a, **{**k, "disable": True})

    print("\nTraining (exact_match/train.py, unmodified). One line per epoch:\n")
    train_module.train()

    # Record the input layout INSIDE the checkpoint. A model only understands
    # the serialization it was trained on, and feeding it another does not
    # raise -- it silently destroys accuracy. This corpus is `COL attr VAL
    # value`, while inference.py's historical default was
    # `title | brand x | description`; the mismatch measured three obvious
    # matches at 49.8% / 20.6% / 4.4% instead of ~100%, i.e. zero recall, with
    # the service still reporting healthy. ProductComparer reads this field.
    import json
    meta_path = os.path.join(config.TRAINED_MODEL_DIR, "training_metadata.json")
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except Exception:  # noqa: BLE001
        meta = {}
    meta["serialization"] = "colval"
    meta["trained_on"] = f"real corpus: {dict(train_df['source'].value_counts())}"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"recorded serialization=colval in {meta_path}")

    # ---- evaluate, one benchmark at a time --------------------------------
    import torch
    from sklearn.metrics import precision_recall_fscore_support
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from build_real_corpus import load_er_magellan, serialize_wdc
    from evaluate_on_wdc import load_wdc_local, predict_probabilities, sweep_threshold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.TRAINED_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(config.TRAINED_MODEL_DIR).to(device)
    print(f"\nEvaluating {config.TRAINED_MODEL_DIR}")

    results: List[tuple] = []

    def score(name: str, text_a, text_b, y_true, published=None):
        probs = predict_probabilities(model, tokenizer, list(text_a), list(text_b), device, 64)
        y_pred = probs.argmax(axis=1)
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                     zero_division=0)
        best_f1, best_t = sweep_threshold(np.asarray(y_true), probs[:, 1])
        print(f"\n{name}  (n={len(y_true):,}, positives {int(np.sum(y_true)):,})")
        print(f"  precision {p:.4f} | recall {r:.4f} | F1 {f:.4f}")
        print(f"  best F1 over P(match) sweep: {best_f1:.4f} at threshold {best_t:.2f}")
        results.append((name, f * 100, best_f1 * 100, published))

    for folder in sorted(glob.glob(os.path.join(args.er_dir, "*"))):
        name = os.path.basename(folder)
        if "Dirty" in name:
            continue                       # excluded from training; not a fair test here
        splits = load_er_magellan(folder, name)
        if "test" not in splits:
            continue
        t = splits["test"]
        pub = PUBLISHED_ER.get(name)
        score(name, t.text_a, t.text_b, t.label.to_numpy(), pub)

    if args.wdc:
        try:
            for tag, rows in load_wdc_local(args.wdc).items():
                score(f"WDC-{tag}",
                      [serialize_wdc(r, "left") for r in rows],
                      [serialize_wdc(r, "right") for r in rows],
                      np.array([r["label"] for r in rows]),
                      ("RoBERTa", {"SEEN": 78.58, "HALF-SEEN": 75.91,
                                   "UNSEEN": 71.14}.get(tag)))
        except SystemExit as exc:
            print(f"\nWDC gold standards skipped: {exc}")

    print("\n" + "=" * 84)
    print("SUMMARY -- F1 x 100, per benchmark (never pooled)")
    print("=" * 84)
    print(f"{'benchmark':<32}{'argmax':>10}{'best-thr':>10}   published")
    for name, f_argmax, f_best, pub in results:
        ref = f"{pub[0]} {pub[1]:.2f}" if pub and pub[1] else "-"
        print(f"{name:<32}{f_argmax:>10.2f}{f_best:>10.2f}   {ref}")

    print("\nNotes:")
    print(" - best-thr is tuned ON the test set, so it is an optimistic upper bound;")
    print("   argmax is the honest number.")
    print(" - published Ditto figures used roberta-base and were trained on each")
    print("   benchmark ALONE; this model is one checkpoint trained on all of them")
    print("   at once, so per-benchmark numbers may be lower while generality is higher.")


if __name__ == "__main__":
    main()
