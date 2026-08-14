"""
calibrate_category_threshold.py
================================
Does a PER-CATEGORY decision threshold beat the single global one?

Open question from PROJECT_HANDOVER.pdf S5.2: "Per-category threshold
calibration -- unknown, ~1 day, needs category at query time." Never
measured before this. Reuses calibrate_threshold.py's scoring, caching,
and minimum-improvement-SE gate wholesale -- the only new thing here is
splitting validation and test by product_taxonomy.categorize() before
fitting.

METHODOLOGY, same rules as calibrate_threshold.py
--------------------------------------------------
  * Fit ONLY on validation, per category. Test is scored, never fitted on.
  * A category's fitted threshold ships only if it beats the DEFAULT
    (config.INFERENCE_THRESHOLD, 0.50) by >= --min-improvement-se bootstrap
    F1 SEs on THAT CATEGORY's validation slice. Small categories have large
    SE and will usually fail the gate -- that is the gate working, not a
    bug. A category with too few positives to bootstrap meaningfully is
    reported as SKIPPED, not silently given a threshold.
  * category is decided from text_a + text_b via product_taxonomy.categorize()
    (UNKNOWN is its own bucket, not dropped -- it is the honest answer for
    "the API will not know the category either" and needs its own row).

    python calibrate_category_threshold.py --model trained_model_v11 --wdc 50pair
"""
import argparse
import glob
import os

import numpy as np

from calibrate_threshold import (
    _load_model, bootstrap_f1_se, collect_splits, f1_at, fit_threshold,
    score_split,
)
from product_taxonomy import categorize


def _categorize_rows(text_a, text_b) -> np.ndarray:
    return np.array([categorize(f"{a} {b}") for a, b in zip(text_a, text_b)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--valid", default="data/real_corpus_valid.csv")
    ap.add_argument("--wdc", default=None)
    ap.add_argument("--er-dir", default="data/er_magellan")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cache-dir", default="outputs/calibration")
    ap.add_argument("--min-improvement-se", type=float, default=1.0)
    ap.add_argument("--min-positives", type=int, default=20,
                    help="Below this many positives in a category's validation "
                         "slice, bootstrap SE is too noisy to trust -- skip "
                         "fitting and report the category as too small.")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    if not args.progress:
        os.environ["TQDM_DISABLE"] = "1"

    import pandas as pd
    import config

    model, tok, device, match_idx = _load_model(args.model)
    print(f"model  {args.model}\ndevice {device}\n")
    default_t = float(config.INFERENCE_THRESHOLD)

    # ---- validation: score once, then split by category -------------------
    val = pd.read_csv(args.valid)
    print(f"scoring validation ({len(val):,} pairs) ...")
    p_val, y_val = score_split(model, tok, device, val.text_a, val.text_b,
                               val.label.to_numpy(), args.batch_size,
                               args.cache_dir, args.model, "valid", match_idx)
    cat_val = _categorize_rows(val.text_a, val.text_b)

    print("\nvalidation category composition:")
    for cat in sorted(set(cat_val)):
        mask = cat_val == cat
        print(f"  {cat:<12} n={mask.sum():>6,}  positives={int(y_val[mask].sum()):>5,}")

    # ---- test: score once per benchmark, then split by category -----------
    print("\nscoring test benchmarks ...")
    splits = collect_splits(args)
    test_p, test_y, test_cat = [], [], []
    for name, ta, tb, y_true in splits:
        p, y = score_split(model, tok, device, ta, tb, y_true, args.batch_size,
                           args.cache_dir, args.model, name, match_idx)
        test_p.append(p)
        test_y.append(y)
        test_cat.append(_categorize_rows(ta, tb))
    test_p = np.concatenate(test_p)
    test_y = np.concatenate(test_y)
    test_cat = np.concatenate(test_cat)

    print("\ntest category composition (pooled across all 6 benchmarks):")
    for cat in sorted(set(test_cat)):
        mask = test_cat == cat
        print(f"  {cat:<12} n={mask.sum():>6,}  positives={int(test_y[mask].sum()):>5,}")

    # ---- fit per category on validation, gated, then read out on test -----
    print("\n" + "=" * 96)
    print(f"PER-CATEGORY RESULTS  (default {default_t:.2f}, min-improvement-se="
          f"{args.min_improvement_se:g})")
    print("=" * 96)
    w = 12
    print(f"{'category':<{w}}{'val n':>8}{'val pos':>8}{'fitted':>8}{'gain':>8}"
          f"{'need':>8}{'gate':>10}{'test n':>8}{'F1@0.50':>9}{'F1@cat':>9}{'delta':>8}")

    all_categories = sorted(set(cat_val) | set(test_cat))
    pooled_default_pred = np.zeros(len(test_y), dtype=int)
    pooled_category_pred = np.zeros(len(test_y), dtype=int)

    for cat in all_categories:
        vmask = cat_val == cat
        tmask = test_cat == cat
        n_val, pos_val = int(vmask.sum()), int(y_val[vmask].sum())
        n_test = int(tmask.sum())

        pooled_default_pred[tmask] = (test_p[tmask] >= default_t).astype(int)

        if pos_val < args.min_positives or (n_val - pos_val) < args.min_positives:
            print(f"{cat:<{w}}{n_val:>8,}{pos_val:>8,}{'--':>8}{'--':>8}{'--':>8}"
                  f"{'TOO SMALL':>10}{n_test:>8,}{'--':>9}{'--':>9}{'--':>8}")
            pooled_category_pred[tmask] = (test_p[tmask] >= default_t).astype(int)
            continue

        fit_f1, fitted = fit_threshold(y_val[vmask], p_val[vmask])
        base_f1 = f1_at(y_val[vmask], p_val[vmask], default_t)
        se = bootstrap_f1_se(y_val[vmask], p_val[vmask], fitted)
        gain = fit_f1 - base_f1
        need = args.min_improvement_se * se

        if gain >= need:
            threshold, gate = fitted, "MOVED"
        else:
            threshold, gate = default_t, "kept"

        pooled_category_pred[tmask] = (test_p[tmask] >= threshold).astype(int)

        if n_test > 0:
            f_default = f1_at(test_y[tmask], test_p[tmask], default_t) * 100
            f_cat = f1_at(test_y[tmask], test_p[tmask], threshold) * 100
            delta = f_cat - f_default
        else:
            f_default = f_cat = delta = float("nan")

        print(f"{cat:<{w}}{n_val:>8,}{pos_val:>8,}{threshold:>8.2f}{gain * 100:>+8.2f}"
              f"{need * 100:>8.2f}{gate:>10}{n_test:>8,}{f_default:>9.2f}{f_cat:>9.2f}"
              f"{delta:>+8.2f}")

    # ---- the number that actually answers the question --------------------
    from sklearn.metrics import f1_score
    f1_global = f1_score(test_y, pooled_default_pred, zero_division=0) * 100
    f1_category = f1_score(test_y, pooled_category_pred, zero_division=0) * 100
    print("\n" + "=" * 96)
    print("POOLED TEST F1 (all benchmarks, all categories combined)")
    print("=" * 96)
    print(f"  single global threshold (0.50)         {f1_global:6.2f}")
    print(f"  per-category threshold (gated, above)  {f1_category:6.2f}")
    print(f"  delta                                  {f1_category - f1_global:+6.2f}")
    print("\n  This is the number that decides it: if per-category calibration is worth")
    print("  building (API needs category at query time, ranker needs a threshold table),")
    print("  this delta has to be real, not a category or two moving on noise.")


if __name__ == "__main__":
    main()
