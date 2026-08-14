"""
build_real_corpus.py
====================
Merges every HUMAN-LABELLED product-matching dataset available into one binary
training corpus, with an honest split and leakage checks.

WHY
---
The measured story of this project in one line: the same model scored 45.9 F1
trained on rule-generated labels and 75.6 trained on 6,000 human-labelled
ones. Data was the only bottleneck. This file gathers all the human labels
that already exist so no further labelling is needed to get a strong matcher.

SOURCES
-------
  ER-Magellan (Ditto's copies of the DeepMatcher benchmarks), tab-separated
  `left <TAB> right <TAB> label`, already serialized as `COL attr VAL value`:
      Structured/Walmart-Amazon   10,242 pairs   electronics, cross-retailer
      Structured/Amazon-Google    11,460 pairs   software
      Textual/Abt-Buy              9,575 pairs   product, long noisy text
  WDC Products train/valid        real multi-shop offers, the benchmark this
                                  project is already measured against

WHAT IS DELIBERATELY EXCLUDED
-----------------------------
`Dirty/Walmart-Amazon` is the SAME 10,242 pairs as the Structured version with
attribute values shuffled between columns. Training on both would put
near-duplicate pairs in the corpus and quietly leak across splits -- the exact
defect that cost this project a full training run when 328 mirrored pairs were
found in relationship_pairs_final.csv. It is kept as a separate robustness
test instead (--include-dirty to override).

SPLITS
------
Each benchmark ships its own train/valid/test, engineered to control leakage.
Those are respected: train merges with train, valid with valid. Test sets are
NOT merged -- a single number over pooled test sets from four benchmarks would
be uninterpretable and not comparable to any published result. Evaluate per
benchmark instead.

Usage:
    python build_real_corpus.py
    python build_real_corpus.py --wdc /kaggle/working/wdc
"""

import argparse
import collections
import glob
import gzip
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

import config

ER_DIR = "data/er_magellan"
OUT_TRAIN = "data/real_corpus_train.csv"
OUT_VALID = "data/real_corpus_valid.csv"


def load_er_magellan(folder: str, name: str) -> Dict[str, pd.DataFrame]:
    """Reads Ditto's `left <TAB> right <TAB> label` files."""
    out: Dict[str, pd.DataFrame] = {}
    for split in ("train", "valid", "test"):
        path = os.path.join(folder, f"{split}.txt")
        if not os.path.exists(path):
            continue
        a, b, y = [], [], []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 3:
                    continue
                a.append(parts[0].strip())
                b.append(parts[1].strip())
                y.append(int(parts[2]))
        out[split] = pd.DataFrame({"text_a": a, "text_b": b, "label": y, "source": name})
    return out


def _read_json_rows(path: str) -> List[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:
        text = fh.read()
    try:
        return [json.loads(l) for l in text.splitlines() if l.strip()]
    except json.JSONDecodeError:
        rows = json.loads(text)
        return rows if isinstance(rows, list) else [rows]


def _words(v, n: int) -> str:
    return " ".join(str(v or "").split(" ")[:n]).strip()


def serialize_wdc(row: dict, side: str) -> str:
    """Same COL/VAL shape as ER-Magellan, so every source looks alike to the
    model. Field order and word limits copied from the WDC baseline code."""
    return (f"COL brand VAL {_words(row.get(f'brand_{side}'), 5)} "
            f"COL title VAL {_words(row.get(f'title_{side}'), 50)} "
            f"COL description VAL {_words(row.get(f'description_{side}'), config.DESCRIPTION_WORDS)}").strip()


def load_wdc(work: str, size: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for split, suffix in (("train", f"_train_{size}"), ("valid", f"_valid_{size}")):
        hits = sorted(glob.glob(os.path.join(work, "**", f"*{suffix}.json*"), recursive=True))
        if not hits:
            print(f"  WDC {split}: no '*{suffix}.json[.gz]' found under {work} -- skipped")
            continue
        rows = _read_json_rows(hits[0])
        out[split] = pd.DataFrame({
            "text_a": [serialize_wdc(r, "left") for r in rows],
            "text_b": [serialize_wdc(r, "right") for r in rows],
            "label": [int(r["label"]) for r in rows],
            "source": "WDC-Products",
        })
    return out


def dedupe(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Order-invariant dedup. (A,B) and (B,A) are one pair -- 328 such mirrored
    pairs were previously found in this project's own data, 2 of which leaked
    across the train/test boundary."""
    key = [tuple(sorted((a, b))) for a, b in zip(df["text_a"], df["text_b"])]
    df = df.assign(_ka=[k[0] for k in key], _kb=[k[1] for k in key])
    before = len(df)
    df = df.drop_duplicates(subset=["_ka", "_kb", "label"]).drop(columns=["_ka", "_kb"])
    return df.reset_index(drop=True), before - len(df)


def entities(df: pd.DataFrame) -> set:
    return set(df["text_a"]) | set(df["text_b"])


def main():
    ap = argparse.ArgumentParser(description="Merge all human-labelled product-matching data.")
    ap.add_argument("--er-dir", default=ER_DIR)
    ap.add_argument("--wdc", default=None, help="Folder holding WDC json files (optional)")
    ap.add_argument("--wdc-size", default="large", choices=["small", "medium", "large"])
    ap.add_argument("--swap-augment", action="store_true",
                    help="Emit every TRAIN pair in both orders so the model learns "
                         "that matching is symmetric. Doubles the corpus and the "
                         "training time. Validation is left untouched.")
    ap.add_argument("--include-dirty", action="store_true",
                    help="Also train on Dirty/Walmart-Amazon. Off by default: it duplicates "
                         "the Structured pairs and would leak near-duplicates across splits.")
    args = ap.parse_args()

    datasets: Dict[str, Dict[str, pd.DataFrame]] = {}
    for folder in sorted(glob.glob(os.path.join(args.er_dir, "*"))):
        name = os.path.basename(folder)
        if "Dirty" in name and not args.include_dirty:
            print(f"  {name}: EXCLUDED (duplicate pairs of the Structured variant)")
            continue
        loaded = load_er_magellan(folder, name)
        if loaded:
            datasets[name] = loaded

    if args.wdc:
        wdc = load_wdc(args.wdc, args.wdc_size)
        if wdc:
            datasets["WDC-Products"] = wdc

    if not datasets:
        raise SystemExit("No datasets found. Check --er-dir / --wdc.")

    print("\nper-source pair counts:")
    for name, splits in datasets.items():
        parts = " ".join(f"{s}={len(d):,}({int(d.label.sum()):,}+)" for s, d in splits.items())
        print(f"  {name:<28} {parts}")

    train = pd.concat([s["train"] for s in datasets.values() if "train" in s], ignore_index=True)
    valid = pd.concat([s["valid"] for s in datasets.values() if "valid" in s], ignore_index=True)

    train, dup_tr = dedupe(train)
    valid, dup_va = dedupe(valid)
    print(f"\ndeduplicated (order-invariant): train -{dup_tr}, valid -{dup_va}")

    # dedupe() keys on (pair, LABEL), so a pair annotated both ways survives
    # twice and the model is told one input is simultaneously a match and a
    # non-match. Measured: 12 such pairs in the 38,145-row train split, present
    # in every run since v7. They come from distinct source records that
    # serialize to identical COL/VAL text, which the merge notes above already
    # flags as possible.
    #
    # Both copies are dropped rather than one being picked. There is no
    # evidence for which annotation is right, and a contradictory label is
    # worse than a missing one -- it is noise pointing in two directions at
    # once. Same reasoning as the EXCLUDE rows in the Indian eval set.
    def drop_contradictions(df, name):
        if df.empty:
            return df, 0
        # Plain dict, deliberately not pandas. Two pandas attempts got this
        # wrong in ways that looked plausible: pd.Index over tuples builds a
        # MultiIndex so groupby(level=0) grouped on the first text alone and
        # found nothing, and a "\x00"-joined string key was truncated at the
        # null byte, collapsing 18,526 distinct pairs into 6,457 groups and
        # manufacturing 1,026 false contradictions. Ground truth is 12.
        seen = {}
        for a, b, lab in zip(df.text_a, df.text_b, df.label):
            seen.setdefault(tuple(sorted((a, b))), set()).add(lab)
        bad = {k for k, labels in seen.items() if len(labels) > 1}
        if not bad:
            return df, 0
        keep = [tuple(sorted((a, b))) not in bad
                for a, b in zip(df.text_a, df.text_b)]
        return df[keep].reset_index(drop=True), len(df) - sum(keep)

    train, contra_tr = drop_contradictions(train, "train")
    valid, contra_va = drop_contradictions(valid, "valid")
    if contra_tr or contra_va:
        print(f"contradictory labels (same pair marked both 0 and 1): "
              f"train -{contra_tr}, valid -{contra_va} rows dropped")

    # Any entity shared between train and valid inflates validation, so it is
    # measured and reported rather than assumed away. The benchmarks control
    # this internally, but merging four of them is a new situation.
    shared = entities(train) & entities(valid)
    print(f"entities shared train/valid: {len(shared):,} "
          f"({len(shared) / max(len(entities(valid)), 1):.2%} of valid entities)")

    # Merging four benchmarks creates overlaps none of them has alone: distinct
    # source records can serialize to identical COL/VAL text, so a pair held
    # out by one benchmark can appear in another's training split. Measured at
    # 123 pairs on the first run. Those are dropped from VALID, never from
    # train, so validation stays a clean holdout.
    key_tr = {tuple(sorted((a, b))) for a, b in zip(train.text_a, train.text_b)}
    va_key = [tuple(sorted((a, b))) for a, b in zip(valid.text_a, valid.text_b)]
    leaked = sum(1 for k in va_key if k in key_tr)
    if leaked:
        valid = valid[[k not in key_tr for k in va_key]].reset_index(drop=True)
        print(f"exact pair overlap train/valid: {leaked} -> dropped from valid")
    else:
        print("exact pair overlap train/valid: 0")

    key_va = {tuple(sorted((a, b))) for a, b in zip(valid.text_a, valid.text_b)}
    assert not (key_tr & key_va), "pair leak survived the filter"

    # Entity overlap is NOT fixed here. ER-Magellan splits at the pair level by
    # design, so one product legitimately appears in many pairs across splits.
    # Re-splitting by entity would break comparability with every published
    # number on these benchmarks. It is reported so the merged valid F1 is read
    # as optimistic; the per-benchmark TEST sets remain the real measurement.

    # ---- optional swap augmentation ---------------------------------------
    # "Is A the same product as B" is symmetric, but a cross-encoder is not:
    # [CLS] a [SEP] b [SEP] and its mirror are different inputs with different
    # token_type_ids. Because dedupe() above is order-invariant, every pair
    # reaches training in exactly ONE order and the model never sees the
    # mirror. Measured consequence on the 190-pair Indian set: median gap
    # between the two orders 0.06 pp, p95 35.9 pp, worst 91.8 pp, and 8 pairs
    # (4.2%) flip across the decision threshold on argument order alone.
    #
    # ProductComparer canonicalises at serving time, which makes the SERVICE
    # consistent. This makes the MODEL consistent, which is the durable fix.
    #
    # ORDER MATTERS HERE: this runs after dedupe() and after the train/valid
    # leak filter. Augmenting earlier would be a no-op, because the
    # order-invariant dedup would immediately collapse each mirror back into
    # its original, and the leak filter keys on sorted pairs so mirrors cannot
    # smuggle a validation pair into training.
    if args.swap_augment:
        mirror = train.copy()
        mirror["text_a"], mirror["text_b"] = train["text_b"].values, train["text_a"].values
        before = len(train)
        train = pd.concat([train, mirror], ignore_index=True)
        print(f"\nswap augmentation: {before:,} -> {len(train):,} train pairs "
              f"(each pair now appears in both orders)")
        print("  valid is NOT augmented -- early stopping and the calibration gate")
        print("  must judge this model on the same yardstick as the previous ones.")

    os.makedirs("data", exist_ok=True)
    train.to_csv(OUT_TRAIN, index=False)
    valid.to_csv(OUT_VALID, index=False)

    print(f"\nTRAIN {len(train):,} pairs  ({int(train.label.sum()):,} positive, "
          f"{train.label.mean():.1%})  -> {OUT_TRAIN}")
    print(f"VALID {len(valid):,} pairs  ({int(valid.label.sum()):,} positive, "
          f"{valid.label.mean():.1%})  -> {OUT_VALID}")
    print("\ntrain composition:")
    for src, n in collections.Counter(train["source"]).most_common():
        print(f"  {src:<28} {n:>7,}")

    print("\nTest sets are intentionally NOT merged -- evaluate per benchmark so the")
    print("numbers stay comparable to published results.")


if __name__ == "__main__":
    main()
