"""
exactmatch/evaluate.py
=======================
Scores a trained exactmatch checkpoint against the six benchmark test sets
(ER-Magellan Amazon-Google/Walmart-Amazon/Abt-Buy + WDC SEEN/HALF-SEEN/
UNSEEN), the same comparison surface every prior checkpoint (v7-v11) was
measured on. Reuses collect_splits() from the root calibrate_threshold.py
for benchmark loading -- that's generic data-loading, not something this
package needs to reimplement.

    python -m exactmatch.evaluate --model exactmatch/trained_model --wdc 50pair
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--wdc", default=None)
    ap.add_argument("--er-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    if not args.progress:
        os.environ["TQDM_DISABLE"] = "1"

    import numpy as np
    import torch
    from sklearn.metrics import precision_recall_fscore_support
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    import exactmatch.config as exconfig
    from calibrate_threshold import collect_splits, fit_threshold

    model_dir = args.model or exconfig.TRAINED_MODEL_DIR
    args.er_dir = args.er_dir or os.path.join(exconfig.ROOT_DIR, "data", "er_magellan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    print(f"model {model_dir} | device {device}\n")

    from evaluate_on_wdc import predict_probabilities

    splits = collect_splits(args)
    rows = []
    for name, ta, tb, y_true in splits:
        probs = predict_probabilities(model, tok, ta, tb, device, args.batch_size)
        p = probs[:, 1]
        pred = (p >= exconfig.INFERENCE_THRESHOLD).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0)
        oracle_f1, oracle_t = fit_threshold(y_true, p)
        rows.append((name, f1 * 100, oracle_f1 * 100, oracle_t))
        print(f"{name:<28} n={len(y_true):>6,}  precision {prec:.4f}  recall {rec:.4f}  F1 {f1:.4f}")

    print("\n" + "=" * 74)
    print(f"{'benchmark':<30}{'argmax':>10}{'best-thr':>10}")
    for name, f_argmax, f_oracle, t_oracle in rows:
        print(f"{name:<30}{f_argmax:>10.2f}{f_oracle:>10.2f}")
    if rows:
        print("-" * 74)
        print(f"{'mean':<30}{np.mean([r[1] for r in rows]):>10.2f}"
              f"{np.mean([r[2] for r in rows]):>10.2f}")


if __name__ == "__main__":
    main()
