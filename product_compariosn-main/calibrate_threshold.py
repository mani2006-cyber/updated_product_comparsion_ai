"""
calibrate_threshold.py
======================
Fits ONE global decision threshold on the validation set, records it inside the
checkpoint, and reports honest calibrated numbers per benchmark.

WHY THIS EXISTS
---------------
Comparing the 38k-pair model (v7) against the 78k-pair LSPC model (v8) showed
the two are indistinguishable at a tuned threshold (mean best-thr 82.37 vs
83.02) while differing by up to 2.75 F1 at argmax. The gap was never capability
-- it was the operating point. Adding 40k LSPC pairs at 14.7% positives pulled
the corpus positive rate from 26.8% to 20.6% and shifted the probability
distribution, so a hardcoded 0.5 stopped sitting anywhere near the optimum:
v8's best cut-point on WDC-UNSEEN was 0.94, not 0.5. Precision fell from 0.669
to 0.612 while recall rose from 0.848 to 0.904 -- a model saying "match" too
readily, not a model that got worse.

FIT ON VALIDATION, NEVER ON TEST
--------------------------------
train_on_real_corpus.py already prints a `best-thr` column. That column is swept
ON the test set, so it is an optimistic upper bound and cannot be shipped -- it
is the number you could have got if you had known the answers. This script fits
on data/real_corpus_valid.csv, which the model never trained on and which no
benchmark test set overlaps, then applies that single number to every test set.
The difference between the two columns is the honest cost of not knowing.

ONE GLOBAL THRESHOLD, NOT PER BENCHMARK
---------------------------------------
Per-benchmark thresholds are unservable: a live query does not arrive labelled
"Abt-Buy", so there is no way to pick which threshold applies. Per-CATEGORY
calibration is viable later if category is known at query time, but that is a
different feature and is deliberately not built here. The per-benchmark optima
are printed for diagnosis only, clearly marked as not shippable.

THRESHOLD SPREAD IS A SELECTION CRITERION, NOT JUST F1
------------------------------------------------------
Because one number has to serve every domain, HOW FAR APART the per-benchmark
optima sit matters independently of F1. A model whose optimum swings 0.61 -> 0.94
across domains cannot be served well by any single threshold, however good its
best-case numbers look; a model whose optima cluster in 0.74-0.80 can. So the
summary reports the spread of per-benchmark optima and the calibrated-vs-
test-tuned gap -- the measured cost of not knowing a query's domain -- alongside
the F1. When two models tie on F1 within noise, prefer the tighter spread.

THE VALIDATION SET MUST COVER EVERY BENCHMARK DOMAIN
----------------------------------------------------
Fitting on a validation set that omits a domain silently produces a threshold
tuned to whatever remains. Measured: fitting on a stale 6,105-pair validation
set containing no WDC-Products rows produced 0.17 -- essentially
Amazon-Google's own optimum of 0.16 -- and cost WDC-SEEN 1.00 F1 and
WDC-HALF-SEEN 1.20. The correct set is the 10,605-pair one that includes
WDC-Products validation. This is checked and warned about at startup.

Usage:
    python calibrate_threshold.py --model trained_model_real --wdc 50pair
    python calibrate_threshold.py --model trained_model_v8 --wdc 50pair --write
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np


def _load_model(model_dir: str):
    """Loads a checkpoint and REFUSES anything that is not a binary matcher.

    Without this check the script silently reads probs[:, 1] as P(match). On a
    5-class checkpoint column 1 is a different class entirely, and the output
    still looks like a result: measured on a mislabelled checkpoint, validation
    F1 came out at 13.20 and the fitted threshold slid to the bottom of the
    sweep (0.05) because the score column was noise. Same family as the
    serialization bug -- wrong input, no exception, plausible-looking numbers.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()

    id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
    if len(id2label) != 2:
        raise SystemExit(
            f"\n{model_dir} has {len(id2label)} labels: {id2label}\n"
            "calibrate_threshold.py only handles BINARY same/different checkpoints.\n"
            "A 5-class model here would be scored on the wrong probability column\n"
            "and would produce numbers that look real but mean nothing."
        )

    # Find the match column by NAME rather than assuming index 1.
    match_idx = next((i for i, n in id2label.items()
                      if n.lower() in ("same_product", "match", "label_1", "1")), None)
    if match_idx is None:
        match_idx = 1
        print(f"  NOTE: id2label={id2label} has no recognisable match class; "
              f"assuming index 1.")
    return model, tok, device, match_idx


def _cache_path(cache_dir: str, model_dir: str, split: str) -> str:
    tag = os.path.basename(os.path.normpath(model_dir))
    return os.path.join(cache_dir, tag, f"{split}.npz")


def score_split(model, tok, device, text_a, text_b, y_true, batch_size,
                cache_dir: str, model_dir: str, split: str,
                match_idx: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """P(match) for one split, cached so re-runs cost nothing."""
    from evaluate_on_wdc import predict_probabilities

    path = _cache_path(cache_dir, model_dir, split)
    if os.path.exists(path):
        blob = np.load(path)
        if len(blob["y"]) == len(y_true):
            print(f"  {split:<28} cached ({len(y_true):,} pairs)")
            return blob["p"], blob["y"]

    probs = predict_probabilities(model, tok, list(text_a), list(text_b), device, batch_size)
    p, y = probs[:, match_idx], np.asarray(y_true, dtype=int)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, p=p, y=y)
    return p, y


def f1_at(y: np.ndarray, p: np.ndarray, t: float) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y, (p >= t).astype(int), zero_division=0))


def fit_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Best F1 and its cut-point. Same 0.05..0.95 grid the existing sweep uses,
    so fitted and test-tuned numbers stay directly comparable."""
    best_f1, best_t = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_at(y, p, float(t))
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_f1, best_t


def collect_splits(args) -> List[Tuple[str, list, list, np.ndarray]]:
    """(name, text_a, text_b, y_true) for every test benchmark, serialized
    exactly as training did -- a different layout would silently change scores."""
    from build_real_corpus import load_er_magellan, serialize_wdc
    from evaluate_on_wdc import load_wdc_local

    out = []
    for folder in sorted(glob.glob(os.path.join(args.er_dir, "*"))):
        name = os.path.basename(folder)
        if "Dirty" in name:
            continue  # excluded from training; not a fair test
        splits = load_er_magellan(folder, name)
        if "test" not in splits:
            continue
        t = splits["test"]
        out.append((name, list(t.text_a), list(t.text_b), t.label.to_numpy()))

    if args.wdc:
        for tag, rows in load_wdc_local(args.wdc).items():
            out.append((f"WDC-{tag}",
                        [serialize_wdc(r, "left") for r in rows],
                        [serialize_wdc(r, "right") for r in rows],
                        np.array([r["label"] for r in rows], dtype=int)))
    return out


def _check_valid_coverage(val, args) -> None:
    """Refuses to fit quietly on a validation set that omits a benchmark domain.

    A threshold fitted where a domain is absent is tuned to whatever remains,
    and nothing in the output would reveal it -- see the module docstring for
    the measured case that motivated this check.
    """
    if "source" not in val.columns:
        print("  WARNING: validation CSV has no 'source' column; cannot verify "
              "that every benchmark domain is represented.\n")
        return

    counts = val["source"].value_counts()
    print(f"validation composition ({len(val):,} pairs, "
          f"positives {val.label.mean():.1%}):")
    for src, n in counts.items():
        print(f"  {src:<28} {n:>7,}  {n / len(val):6.1%}")

    expected = {"WDC-Products"}
    if args.wdc:
        pass  # WDC gold standards are test-side; the training-side name is above
    for folder in sorted(glob.glob(os.path.join(args.er_dir, "*"))):
        name = os.path.basename(folder)
        if "Dirty" not in name:
            expected.add(name)

    missing = sorted(d for d in expected if d not in set(counts.index))
    if missing:
        print("\n  *** WARNING: these benchmark domains are ABSENT from validation:")
        for d in missing:
            print(f"        {d}")
        print("  The fitted threshold will be tuned to the domains that remain and")
        print("  will likely hurt the missing ones. Rebuild with:")
        print("      python build_real_corpus.py --wdc <dir> --wdc-size large")
    print()


def main():
    ap = argparse.ArgumentParser(description="Fit a global decision threshold on validation.")
    ap.add_argument("--model", required=True, help="Checkpoint directory")
    ap.add_argument("--valid", default="data/real_corpus_valid.csv")
    ap.add_argument("--wdc", default=None, help="Folder holding the WDC gold standards")
    ap.add_argument("--er-dir", default="data/er_magellan")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--cache-dir", default="outputs/calibration")
    ap.add_argument("--write", action="store_true",
                    help="Record the fitted threshold in the checkpoint's training_metadata.json")
    ap.add_argument("--progress", action="store_true",
                    help="Show tqdm bars. Off by default -- under `!python` they cannot "
                         "rewrite their line and bury the results table (trap 6.4).")
    args = ap.parse_args()

    if not args.progress:
        os.environ["TQDM_DISABLE"] = "1"

    import pandas as pd

    model, tok, device, match_idx = _load_model(args.model)
    print(f"model  {args.model}\ndevice {device}\n")

    # ---- fit on validation, and only validation ---------------------------
    val = pd.read_csv(args.valid)
    _check_valid_coverage(val, args)
    print("scoring validation (the ONLY data the threshold is fitted on):")
    p_val, y_val = score_split(model, tok, device, val.text_a, val.text_b,
                               val.label.to_numpy(), args.batch_size,
                               args.cache_dir, args.model, "valid", match_idx)
    fit_f1, threshold = fit_threshold(y_val, p_val)
    base_f1 = f1_at(y_val, p_val, 0.5)
    print(f"\n  validation n={len(y_val):,}  positives {y_val.mean():.1%}")
    print(f"  F1 @ 0.50      {base_f1 * 100:6.2f}")
    print(f"  F1 @ {threshold:.2f} (fitted) {fit_f1 * 100:6.2f}   <- threshold shipped\n")

    # A trained matcher scores ~84 here. Anything near chance means the wrong
    # checkpoint, not a weak one -- stop rather than emit numbers that look real.
    if fit_f1 < 0.40:
        raise SystemExit(
            f"\nValidation F1 is {fit_f1 * 100:.2f} at the BEST threshold on data this\n"
            f"model should score ~84 on. That is not a weak checkpoint, it is the wrong\n"
            f"one -- an untrained head, a different task, or a directory that is not the\n"
            f"model you think it is. Check {args.model}/training_metadata.json and\n"
            f"config.json before trusting anything downstream. Refusing to continue."
        )

    # ---- apply that one number to every test benchmark --------------------
    print("scoring test benchmarks (threshold is NOT refitted on these):")
    splits = collect_splits(args)
    rows = []
    for name, ta, tb, y_true in splits:
        p, y = score_split(model, tok, device, ta, tb, y_true, args.batch_size,
                           args.cache_dir, args.model, name, match_idx)
        oracle_f1, oracle_t = fit_threshold(y, p)
        rows.append((name, f1_at(y, p, 0.5) * 100, f1_at(y, p, threshold) * 100,
                     oracle_f1 * 100, oracle_t))

    w = 30
    print("\n" + "=" * 88)
    print(f"CALIBRATED RESULTS -- global threshold {threshold:.2f}, fitted on validation only")
    print("=" * 88)
    print(f"{'benchmark':<{w}}{'@0.50':>9}{'calibrated':>12}{'delta':>9}"
          f"{'test-tuned':>12}{'its thr':>9}")
    for name, f_argmax, f_cal, f_oracle, t_oracle in rows:
        print(f"{name:<{w}}{f_argmax:>9.2f}{f_cal:>12.2f}{f_cal - f_argmax:>+9.2f}"
              f"{f_oracle:>12.2f}{t_oracle:>9.2f}")
    if rows:
        m_argmax = float(np.mean([r[1] for r in rows]))
        m_cal = float(np.mean([r[2] for r in rows]))
        m_oracle = float(np.mean([r[3] for r in rows]))
        print("-" * 88)
        print(f"{'mean':<{w}}{m_argmax:>9.2f}{m_cal:>12.2f}{m_cal - m_argmax:>+9.2f}"
              f"{m_oracle:>12.2f}{'':>9}")

    print("\n  'calibrated' is shippable: one threshold, fitted on validation, applied blind.")
    print("  'test-tuned' is the per-benchmark optimum -- an upper bound you cannot serve,")
    print("  because a live query does not tell you which benchmark it belongs to.")
    print("  The gap between them is the honest cost of not knowing.")

    # ---- selection criteria, one block per model so models compare directly
    if rows:
        thrs = np.array([r[4] for r in rows], dtype=float)
        gap = m_oracle - m_cal
        print("\n" + "=" * 88)
        print(f"SELECTION SUMMARY -- {args.model}")
        print("=" * 88)
        print(f"  fitted global threshold      {threshold:.2f}")
        print(f"  calibrated mean F1           {m_cal:6.2f}")
        print(f"  test-tuned mean F1           {m_oracle:6.2f}   (unservable upper bound)")
        print(f"  cost of not knowing domain   {gap:6.2f}   <- lower is better")
        print()
        # np.ptp(arr), not arr.ptp(): the method was removed in NumPy 2.0.
        print(f"  per-benchmark optimum spread {thrs.min():.2f} - {thrs.max():.2f}"
              f"   range {np.ptp(thrs):.2f}, sd {thrs.std(ddof=0):.3f}   <- lower is better")
        print(f"  optima: " + ", ".join(f"{n.replace('Structured_', '').replace('Textual_', '')}"
                                        f"={t:.2f}" for n, _, _, _, t in rows))
        print()
        print("  A wide spread means no single threshold serves every domain well, which")
        print("  is a fragility F1 alone does not show. When two models tie on calibrated")
        print("  F1 within noise (~1.5-2.5 F1 SE at these sample sizes), prefer the")
        print("  tighter spread and the smaller cost-of-not-knowing.")

    if args.write:
        meta_path = os.path.join(args.model, "training_metadata.json")
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:  # noqa: BLE001
            meta = {}
        meta["inference_threshold"] = round(threshold, 4)
        meta["threshold_fitted_on"] = (
            f"{os.path.basename(args.valid)} (n={len(y_val)}, "
            f"positives {y_val.mean():.1%}); validation only, never test"
        )
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        print(f"\nwrote inference_threshold={threshold:.2f} to {meta_path}")
    else:
        print("\n(--write not given; nothing recorded in the checkpoint)")


if __name__ == "__main__":
    main()
