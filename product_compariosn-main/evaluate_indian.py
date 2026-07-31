"""
evaluate_indian.py
==================
Scores the shipped model against the hand-labelled Indian e-commerce set.

    python evaluate_indian.py --model trained_model_real

This is the first measurement of this model on the market it targets. Every
other benchmark -- WDC Products, Amazon-Google, Walmart-Amazon, Abt-Buy -- is a
US/European catalog.

Read the breakdowns, not just the headline. 190 pairs with 63 positives gives
an F1 standard error around 4-6 points, so a two-point difference between
slices is noise. The value here is in WHICH cases fail, not in the aggregate.
"""

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trained_model_real")
    ap.add_argument("--eval", default="data/indian_eval.csv")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--show-errors", type=int, default=12)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    if not args.progress:
        os.environ["TQDM_DISABLE"] = "1"

    import pandas as pd
    import torch
    from sklearn.metrics import precision_recall_fscore_support
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from calibrate_threshold import bootstrap_f1_se
    from evaluate_on_wdc import predict_probabilities
    from exact_match.inference import ProductComparer

    d = pd.read_csv(args.eval)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()
    if len(model.config.id2label) != 2:
        raise SystemExit(f"{args.model} is not a binary checkpoint.")

    thr = ProductComparer._detect_threshold(args.model)
    print(f"model {args.model} | device {device} | threshold {thr}")
    print(f"eval  {len(d)} pairs, {d.label.sum()} positive ({d.label.mean():.1%})\n")

    probs = predict_probabilities(model, tok, list(d.text_a), list(d.text_b),
                                  device, args.batch_size)
    d["p"] = probs[:, 1]
    d["pred"] = (d.p >= thr).astype(int)

    def line(name, sub):
        if len(sub) == 0 or sub.label.nunique() < 1:
            return
        p, r, f, _ = precision_recall_fscore_support(
            sub.label, sub.pred, average="binary", zero_division=0)
        acc = (sub.label == sub.pred).mean()
        print(f"  {name:<22} n={len(sub):>4}  pos={int(sub.label.sum()):>3}  "
              f"P {p * 100:6.2f}  R {r * 100:6.2f}  F1 {f * 100:6.2f}  acc {acc * 100:6.2f}")

    print("=" * 92)
    print("OVERALL")
    print("=" * 92)
    line("all", d)
    se = bootstrap_f1_se(d.label.to_numpy(), d.p.to_numpy(), thr)
    print(f"  bootstrap F1 SE: {se * 100:.2f}  -- differences below this are noise")

    print("\n" + "=" * 92)
    print("BY DIFFICULTY (mine, assigned during labelling)")
    print("=" * 92)
    for k in ("easy", "medium", "hard"):
        line(k, d[d.difficulty == k])

    print("\n" + "=" * 92)
    print("BY CANDIDATE SOURCE")
    print("=" * 92)
    for k, _ in d.source.value_counts().items():
        line(k, d[d.source == k])

    print("\n" + "=" * 92)
    print("EXCLUDING THE 14 PAIRS I FLAGGED UNSURE")
    print("=" * 92)
    line("confident only", d[d.uncertain == 0])
    line("uncertain only", d[d.uncertain == 1])

    print("\n" + "=" * 92)
    print("VARIANT HANDLING -- the sibling-SKU case (labelled 0)")
    print("=" * 92)
    var = d[d.label3 == "SAME_PRODUCT_DIFFERENT_VARIANT"]
    if len(var):
        wrong = int(var.pred.sum())
        print(f"  {len(var)} variant pairs (colour/size/capacity siblings)")
        print(f"  called SAME by the model: {wrong}/{len(var)} "
              f"({wrong / len(var):.0%}) -- each one is a false positive")
        for _, r in var.iterrows():
            mark = "WRONG" if r.pred else "ok   "
            print(f"    [{mark}] {r.p * 100:6.2f}%  {r.pair_id}  {str(r.title_a)[:52]}")
            print(f"                          vs  {str(r.title_b)[:52]}")

    print("\n" + "=" * 92)
    print(f"WORST ERRORS (top {args.show_errors} by confidence)")
    print("=" * 92)
    err = d[d.label != d.pred].copy()
    err["conf"] = np.where(err.pred == 1, err.p, 1 - err.p)
    for _, r in err.sort_values("conf", ascending=False).head(args.show_errors).iterrows():
        kind = "FALSE POSITIVE" if r.pred == 1 else "FALSE NEGATIVE"
        print(f"\n  {kind}  {r.p * 100:6.2f}%  {r.pair_id}  [{r.label3}, {r.difficulty}]")
        print(f"    A: {str(r.title_a)[:86]}")
        print(f"    B: {str(r.title_b)[:86]}")
        print(f"    why labelled so: {str(r.justification)[:150]}")

    print(f"\n\n  totals: {len(err)} errors of {len(d)} "
          f"({int((err.pred == 1).sum())} false positives, "
          f"{int((err.pred == 0).sum())} false negatives)")


if __name__ == "__main__":
    main()
