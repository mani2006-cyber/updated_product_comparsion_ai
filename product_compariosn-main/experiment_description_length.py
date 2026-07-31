"""
experiment_description_length.py
================================
Measures how much of the input the model actually uses, by truncating the
description field and re-scoring validation.

    python experiment_description_length.py --model trained_model_real

WHY
---
Serialization is `COL brand VAL {5w} COL title VAL {50w} COL description VAL
{100w}` per side. Measured token lengths over the 10,605-pair validation set:

    description        mean   median    p95    max   at 256-cap
    full (100w)       112.0       94    256    256        5.4%
    truncated to 20    84.6       85    133    256        0.0%
    dropped (0w)       54.7       47     96    162        0.0%

Note what this corrects: sequences are NOT mostly at the cap -- the median is
94 and only 5.4% reach 256. Dynamic padding still bought only ~8% (12.27 ->
11.29 ms/pair/epoch) because padding is driven by the LONGEST sample in each
batch, and with p95 = 256 a batch of 32 almost always contains one. So the
saving from truncation is better predicted by the p95 than by the mean:
capping the description at 20 words drops p95 from 256 to 133, which is what
actually lets dynamic padding pay off.

The model number -- the signal that decides product identity -- lives in the
title, and only 60.3% of validation rows carry a description at all (WDC and
Abt-Buy; the Amazon-Google and Walmart-Amazon rows have none). So the question
is whether that 60% is earning its tokens.

WHAT THIS DOES AND DOES NOT PROVE
---------------------------------
It measures whether the model USES the description at inference time. If F1
holds at 20 words, the description is close to dead weight for this checkpoint.

It does NOT license truncating at serving time. Feeding a model a different
text layout than it was trained on is trap 2.3 -- it does not raise, it
silently destroys recall (three obvious matches scored 49.8 / 20.6 / 4.4%
instead of ~100%). Locking in the speedup means RETRAINING with the shorter
format, ~40 min on a T4. This script tells you whether that retrain is worth
starting.

Fitting and reporting are on validation only; no test set is touched.
"""

import argparse
import os
import re
import sys

import numpy as np

# description is the final field, so everything after the marker is the value
_DESC = re.compile(r"(COL description VAL\s*)(.*)$", re.I | re.S)


def truncate_description(text: str, n_words) -> str:
    """Keeps the first `n_words` of the description; None means leave as-is."""
    if n_words is None:
        return text
    m = _DESC.search(str(text or ""))
    if not m:
        return text
    head = str(text)[:m.start()]
    if n_words == 0:
        return head.rstrip()
    kept = " ".join(m.group(2).split()[:n_words])
    return f"{head}{m.group(1)}{kept}".rstrip()


def token_stats(tok, texts_a, texts_b, sample=1500, max_length=256):
    idx = np.linspace(0, len(texts_a) - 1, min(sample, len(texts_a))).astype(int)
    lens = [len(tok(texts_a[i], texts_b[i], truncation=True,
                    max_length=max_length)["input_ids"]) for i in idx]
    lens = np.array(lens)
    return float(lens.mean()), float(np.median(lens)), float((lens >= max_length).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trained_model_real")
    ap.add_argument("--valid", default="data/real_corpus_valid.csv")
    ap.add_argument("--budgets", type=int, nargs="*", default=[100, 50, 20, 0],
                    help="Description word budgets to test. 100 = current.")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None,
                    help="Score only the first N validation rows. For smoke-testing "
                         "the script cheaply; the reported F1 is not meaningful.")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()

    if not args.progress:
        os.environ["TQDM_DISABLE"] = "1"

    import pandas as pd
    import torch
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from calibrate_threshold import bootstrap_f1_se
    from evaluate_on_wdc import predict_probabilities

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()

    if len(model.config.id2label) != 2:
        raise SystemExit(f"{args.model} is not a binary checkpoint; refusing to score it.")

    val = pd.read_csv(args.valid)
    if args.limit:
        val = val.head(args.limit).reset_index(drop=True)
        print(f"*** --limit {args.limit}: smoke run, F1 values are NOT meaningful ***")
    y = val.label.to_numpy().astype(int)
    print(f"model  {args.model}\ndevice {device}")
    print(f"valid  {len(val):,} pairs, {y.sum():,} positive ({y.mean():.1%})")
    has_desc = val.text_a.str.contains("COL description VAL", case=False).mean()
    print(f"       {has_desc:.1%} of rows carry a description field at all -- the "
          f"saving is diluted by the rest\n")

    rows = []
    baseline_f1 = None
    for n in args.budgets:
        ta = [truncate_description(t, n) for t in val.text_a]
        tb = [truncate_description(t, n) for t in val.text_b]
        mean_len, med_len, frac_capped = token_stats(tok, ta, tb)
        probs = predict_probabilities(model, tok, ta, tb, device, args.batch_size)
        p = probs[:, 1]
        f1 = f1_score(y, (p >= 0.5).astype(int), zero_division=0)
        pr, rc, _, _ = precision_recall_fscore_support(
            y, (p >= 0.5).astype(int), average="binary", zero_division=0)
        if baseline_f1 is None:
            baseline_f1 = f1
        rows.append((n, f1 * 100, (f1 - baseline_f1) * 100, pr * 100, rc * 100,
                     mean_len, med_len, frac_capped * 100))
        print(f"  desc<= {n:>3}w   F1 {f1 * 100:6.2f}   mean tokens {mean_len:6.1f}   "
              f"capped {frac_capped:5.1%}")

    se = bootstrap_f1_se(y, p, 0.5)

    print("\n" + "=" * 86)
    print("DESCRIPTION LENGTH vs ACCURACY  (validation only)")
    print("=" * 86)
    print(f"{'desc words':>11}{'F1':>8}{'vs 100w':>9}{'prec':>8}{'recall':>8}"
          f"{'mean tok':>10}{'median':>8}{'at cap':>8}{'est. cost':>11}")
    base_len = rows[0][5]
    for n, f1, d, pr, rc, ml, md, cap in rows:
        print(f"{n:>11}{f1:>8.2f}{d:>+9.2f}{pr:>8.2f}{rc:>8.2f}"
              f"{ml:>10.1f}{md:>8.0f}{cap:>7.1f}%{ml / base_len:>10.2f}x")

    print(f"\n  bootstrap validation F1 SE: {se * 100:.2f}")
    print("  A drop smaller than 1 SE is not a real loss; a drop larger than 1 SE is.")
    print("\n  'est. cost' is the token-length ratio -- a lower bound on the saving,")
    print("  since attention is quadratic in sequence length while the FFN is linear.")
    print("\n  NEXT STEP IF F1 HOLDS: this does NOT mean truncate at serving time.")
    print("  Changing the input layout without retraining is trap 2.3 and silently")
    print("  destroys recall. Retrain with the shorter description (~40 min on a T4),")
    print("  then re-run calibrate_threshold.py on the new checkpoint.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
