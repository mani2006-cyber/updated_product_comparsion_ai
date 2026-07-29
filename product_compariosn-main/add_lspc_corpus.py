"""
add_lspc_corpus.py
==================
Adds the WDC LSPC 2017 corpus (`wdc/products-2017`, ~219k human-labelled pairs
at xlarge) to the training corpus built by build_real_corpus.py -- and, before
anything else, measures how much of it leaks into the WDC gold standards this
project evaluates against.

WHY THE LEAKAGE CHECK COMES FIRST
---------------------------------
`products-2017` is an EARLIER crawl of the same web corpus that WDC Products
(2023) was later built from. If offers are shared, training on LSPC puts test
products into training, and the reported scores stop meaning anything.

The UNSEEN split is the acute case. Its entire value is that those products
never appeared in training -- it is the number that best predicts real-world
behaviour, and the one where this model currently beats every published
baseline (74.78 vs RoBERTa 71.14). Contaminate it and the strongest evidence
in this project silently becomes worthless, with the score going UP, which is
exactly the kind of failure that took days to find last time.

So this script measures overlap and reports it. It does not assume the answer.

  overlap ~0%      -> safe: merge and train
  overlap small    -> use --drop-leaked to remove offending training pairs
  overlap large    -> LSPC and WDC share a corpus; keep LSPC but stop
                      evaluating on the WDC gold standards, and rely on the
                      ER-Magellan test sets instead

MATCHING IS ON NORMALISED TITLES
--------------------------------
Sources serialize differently, so exact string comparison would miss real
overlaps. Titles are lowercased and stripped to alphanumerics before
comparison, which catches the same offer reformatted by a different pipeline.

Usage:
    python add_lspc_corpus.py --wdc /kaggle/working/wdc --check-only
    python add_lspc_corpus.py --wdc /kaggle/working/wdc --size xlarge --drop-leaked
"""

import argparse
import os
import re
from typing import Dict, List, Set, Tuple

import pandas as pd

CATEGORIES = ["computers", "watches", "shoes", "cameras"]
OUT_TRAIN = "data/real_corpus_train.csv"
OUT_VALID = "data/real_corpus_valid.csv"

_NORM = re.compile(r"[^a-z0-9]+")
_TITLE_FROM_COLVAL = re.compile(r"COL title VAL (.*?)(?= COL |$)", re.I | re.S)


def norm_title(text: str) -> str:
    """Lowercase alphanumeric signature of a title, truncated.

    Truncation matters: merchants append varying marketing tails to the same
    product, so comparing full strings under-reports genuine overlap.
    """
    return _NORM.sub("", str(text or "").lower())[:80]


def title_from_serialized(text: str) -> str:
    """Pulls the title back out of `COL title VAL ...` serialized text."""
    m = _TITLE_FROM_COLVAL.search(str(text or ""))
    return m.group(1).strip() if m else str(text or "")


def _words(v, n: int) -> str:
    return " ".join(str(v or "").split(" ")[:n]).strip()


def serialize(row: dict, side: str) -> str:
    return (f"COL brand VAL {_words(row.get(f'brand_{side}'), 5)} "
            f"COL title VAL {_words(row.get(f'title_{side}'), 50)} "
            f"COL description VAL {_words(row.get(f'description_{side}'), 100)}").strip()


def load_lspc(categories: List[str], size: str) -> Dict[str, pd.DataFrame]:
    """Loads wdc/products-2017 from HuggingFace.

    Sizes are NESTED (small subset of medium subset of large subset of xlarge),
    so exactly one size is loaded; combining them would duplicate pairs.
    """
    from datasets import load_dataset

    frames: Dict[str, List[pd.DataFrame]] = {"train": [], "valid": []}
    for cat in categories:
        config = f"{cat}_{size}"
        for split, key in (("train", "train"), ("valid", "validation")):
            try:
                ds = load_dataset("wdc/products-2017", config, split=key)
            except Exception as exc:  # noqa: BLE001
                print(f"  {config}/{key}: unavailable ({exc})")
                continue
            rows = [dict(r) for r in ds]
            frames[split].append(pd.DataFrame({
                "text_a": [serialize(r, "left") for r in rows],
                "text_b": [serialize(r, "right") for r in rows],
                "label": [int(r["label"]) for r in rows],
                "source": f"LSPC-{cat}",
                "raw_a": [r.get("title_left") or "" for r in rows],
                "raw_b": [r.get("title_right") or "" for r in rows],
            }))
            print(f"  {config}/{key}: {len(rows):,} pairs")
    return {k: pd.concat(v, ignore_index=True) for k, v in frames.items() if v}


def gold_standard_titles(wdc_dir: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """Normalised titles appearing in each WDC gold standard."""
    from evaluate_on_wdc import load_wdc_local

    per_split: Dict[str, Set[str]] = {}
    everything: Set[str] = set()
    for tag, rows in load_wdc_local(wdc_dir).items():
        titles = {norm_title(r.get("title_left")) for r in rows}
        titles |= {norm_title(r.get("title_right")) for r in rows}
        titles.discard("")
        per_split[tag] = titles
        everything |= titles
    return everything, per_split


def report_leakage(df: pd.DataFrame, per_split: Dict[str, Set[str]]) -> pd.Series:
    """Reports, per gold standard, how much of it this corpus already contains.
    Returns a boolean mask marking rows that touch ANY gold-standard product."""
    norm_a = df["raw_a"].map(norm_title) if "raw_a" in df else \
        df["text_a"].map(lambda t: norm_title(title_from_serialized(t)))
    norm_b = df["raw_b"].map(norm_title) if "raw_b" in df else \
        df["text_b"].map(lambda t: norm_title(title_from_serialized(t)))

    corpus_titles = set(norm_a) | set(norm_b)
    corpus_titles.discard("")

    print("\n" + "=" * 72)
    print("LEAKAGE: how much of each WDC gold standard is already in training?")
    print("=" * 72)
    for tag, titles in per_split.items():
        hit = titles & corpus_titles
        pct = len(hit) / max(len(titles), 1)
        flag = "OK" if pct < 0.02 else ("WARN" if pct < 0.15 else "SEVERE")
        print(f"  {tag:<12} {len(hit):>6,} / {len(titles):>6,} products  ({pct:6.2%})  {flag}")

    all_gold = set().union(*per_split.values()) if per_split else set()
    mask = norm_a.isin(all_gold) | norm_b.isin(all_gold)
    print(f"\n  training rows touching any gold-standard product: {int(mask.sum()):,} "
          f"({mask.mean():.2%})")
    return mask


def main():
    ap = argparse.ArgumentParser(description="Add WDC LSPC 2017 with a gold-standard leakage check.")
    ap.add_argument("--wdc", required=True, help="Folder holding the WDC gold standards")
    ap.add_argument("--size", default="xlarge", choices=["small", "medium", "large", "xlarge"])
    ap.add_argument("--categories", nargs="*", default=CATEGORIES)
    ap.add_argument("--check-only", action="store_true",
                    help="Measure leakage and stop, writing nothing.")
    ap.add_argument("--drop-leaked", action="store_true",
                    help="Remove training rows touching any gold-standard product.")
    ap.add_argument("--train", default=OUT_TRAIN)
    ap.add_argument("--valid", default=OUT_VALID)
    args = ap.parse_args()

    print(f"Loading wdc/products-2017 ({args.size}) ...")
    lspc = load_lspc(args.categories, args.size)
    if "train" not in lspc:
        raise SystemExit("No LSPC data loaded. Check internet access and `datasets`.")
    print(f"\nLSPC train {len(lspc['train']):,} | valid {len(lspc.get('valid', [])):,}")

    _, per_split = gold_standard_titles(args.wdc)
    mask = report_leakage(lspc["train"], per_split)

    if args.check_only:
        print("\n--check-only: nothing written.")
        return

    train_new = lspc["train"]
    if args.drop_leaked and mask.any():
        train_new = train_new[~mask].reset_index(drop=True)
        print(f"\ndropped {int(mask.sum()):,} leaked rows -> {len(train_new):,} remain")

    existing_train = pd.read_csv(args.train) if os.path.exists(args.train) else pd.DataFrame()
    existing_valid = pd.read_csv(args.valid) if os.path.exists(args.valid) else pd.DataFrame()
    if existing_train.empty:
        print(f"\nWARNING: {args.train} not found -- writing LSPC alone. "
              "Run build_real_corpus.py first to include the other benchmarks.")

    cols = ["text_a", "text_b", "label", "source"]
    train = pd.concat([existing_train, train_new[cols]], ignore_index=True)
    valid = pd.concat([existing_valid,
                       lspc["valid"][cols] if "valid" in lspc else pd.DataFrame()],
                      ignore_index=True)

    # Order-invariant dedup, then a hard train/valid separation. Same discipline
    # as build_real_corpus.py: merging sources creates overlaps none of them has
    # alone (124 such pairs were found the first time).
    def dedupe(df):
        k = [tuple(sorted((a, b))) for a, b in zip(df.text_a, df.text_b)]
        df = df.assign(_ka=[x[0] for x in k], _kb=[x[1] for x in k])
        n = len(df)
        df = df.drop_duplicates(subset=["_ka", "_kb", "label"]).drop(columns=["_ka", "_kb"])
        return df.reset_index(drop=True), n - len(df)

    train, d1 = dedupe(train)
    valid, d2 = dedupe(valid)
    key_tr = {tuple(sorted((a, b))) for a, b in zip(train.text_a, train.text_b)}
    va_k = [tuple(sorted((a, b))) for a, b in zip(valid.text_a, valid.text_b)]
    leaked = sum(1 for k in va_k if k in key_tr)
    if leaked:
        valid = valid[[k not in key_tr for k in va_k]].reset_index(drop=True)

    print(f"\ndeduplicated: train -{d1:,}, valid -{d2:,} | pair leak train/valid: {leaked} dropped")
    train.to_csv(args.train, index=False)
    valid.to_csv(args.valid, index=False)
    print(f"\nTRAIN {len(train):,} pairs ({int(train.label.sum()):,} positive, "
          f"{train.label.mean():.1%}) -> {args.train}")
    print(f"VALID {len(valid):,} pairs ({int(valid.label.sum()):,} positive, "
          f"{valid.label.mean():.1%}) -> {args.valid}")
    print("\ncomposition:")
    for src, n in train["source"].value_counts().items():
        print(f"  {src:<28} {n:>8,}")


if __name__ == "__main__":
    main()
