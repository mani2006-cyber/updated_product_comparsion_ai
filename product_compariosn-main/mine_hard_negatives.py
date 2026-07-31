"""
mine_hard_negatives.py
======================
Adds hard pairs from WDC LSPC 2017 to the training corpus, selected by how
close the two titles are rather than by how many distinct products they cover.

    python mine_hard_negatives.py --wdc 50pair --check-only
    python mine_hard_negatives.py --wdc 50pair --max-pairs 20000

WHY THIS IS NOT "MORE DATA" AGAIN
---------------------------------
Merging 40k LSPC pairs moved the six-benchmark mean by +0.01, and three
independently trained checkpoints all collapse the same way on hard pairs
(precision 15.38 / 13.33 / 16.00). So volume is not the problem.

But that merge sampled for DIVERSITY -- maximising distinct products, ~3 pairs
each. That is the opposite of what the failures need. LSPC holds 171,787 train
pairs; the ones where two titles are nearly identical and the human label still
says "different" are exactly the shape the model gets wrong:

    Noise Buds VS104  vs  Noise Buds VS104 Max        scored 99.99%
    boAt Airdopes 141 vs  boAt Airdopes 141 Pro       scored 99.60%
    JBL Tune 215      vs  JBL Tune 215TWS             scored 99.96%

Diversity-first sampling actively selects AGAINST those, because a near-
duplicate title adds no new product. This script selects FOR them.

THE LABELS ARE NOT GENERATED
----------------------------
This is the distinction from the rule-labelled dataset that scored 95% while
being worthless (45.9 F1 against human labels). There, a rule DECIDED whether a
pair matched. Here the label is LSPC's own human annotation, unchanged; the
only thing computed is which already-labelled pairs to keep. A selection rule
cannot teach the model the rule, because the rule never touches the target.

The one risk it does carry is distribution shift: selecting only near-duplicate
negatives makes the training set unrepresentative. So hard POSITIVES -- pairs
labelled the same product whose titles look nothing alike -- are mined too, in
proportion, and the positive rate is held near the existing corpus. Otherwise
the model would simply learn to say "different" more often, which trades the
false positives for false negatives and measures as an improvement on precision
while being no better.

VERIFY, DO NOT ASSUME
---------------------
--check-only reports the leakage check against the WDC gold standards and the
pattern census WITHOUT writing anything. If the mined pairs do not actually
contain suffix / third-party / form-factor cases, this will not work and the
census says so before an hour of GPU time is spent.
"""

import argparse
import os
import re
from typing import Dict, List, Set, Tuple

import pandas as pd

import config
from add_lspc_corpus import (gold_standard_titles, load_lspc, norm_title,
                             report_leakage, serialize)

OUT_TRAIN = "data/real_corpus_train.csv"
OUT_VALID = "data/real_corpus_valid.csv"

_TOK = re.compile(r"[a-z0-9]+")

# Patterns the shipped model demonstrably cannot see. Counted, never used to
# assign a label -- purely a census so we know whether the mined data contains
# the signal we are short of.
_SUFFIX = re.compile(
    r"\b(max|pro|plus|lite|ultra|mini|air|neo|gen\s*\d|v\d|mk\s*\d|\d+nc|se)\b", re.I)
_THIRD_PARTY = re.compile(
    r"\b(for|compatible\s+with|designed\s+for|replacement|original\s+like|"
    r"suitable\s+for)\b", re.I)
_FORM_FACTOR = re.compile(
    r"\b(tws|neckband|over[- ]ear|on[- ]ear|in[- ]ear|wired|wireless|earbuds|"
    r"headphones|headset)\b", re.I)


def tokens(text: str) -> Set[str]:
    return set(_TOK.findall(str(text or "").lower()))


def title_similarity(a: str, b: str) -> float:
    """Jaccard over title tokens. Cheap, and it is the axis the failures live
    on: every error above is two titles sharing almost every token."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def classify_patterns(a: str, b: str) -> List[str]:
    """Which known failure shapes this pair exhibits. Diagnostic only."""
    out = []
    ta, tb = tokens(a), tokens(b)
    diff = (ta ^ tb)
    if any(_SUFFIX.fullmatch(t) for t in diff):
        out.append("model-suffix")
    if _THIRD_PARTY.search(a) or _THIRD_PARTY.search(b):
        out.append("third-party")
    fa = {t for t in ta if _FORM_FACTOR.fullmatch(t)}
    fb = {t for t in tb if _FORM_FACTOR.fullmatch(t)}
    if fa != fb and (fa or fb):
        out.append("form-factor")
    if any(t.isdigit() or re.fullmatch(r"\d+(gb|ml|g|w|mm|l)", t) for t in diff):
        out.append("spec-token")
    return out or ["other"]


def census(df: pd.DataFrame, title: str) -> None:
    counts: Dict[str, int] = {}
    for _, r in df.iterrows():
        for p in classify_patterns(r.raw_a, r.raw_b):
            counts[p] = counts.get(p, 0) + 1
    print(f"\n  {title} ({len(df):,} pairs)")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<14} {v:>7,}  {v / max(len(df), 1):6.1%}")


def main():
    ap = argparse.ArgumentParser(
        description="Mine hard pairs from LSPC by title similarity.")
    ap.add_argument("--wdc", required=True, help="Folder holding the WDC gold standards")
    ap.add_argument("--size", default="xlarge",
                    choices=["small", "medium", "large", "xlarge"])
    ap.add_argument("--categories", nargs="*",
                    default=["computers", "watches", "shoes", "cameras"])
    ap.add_argument("--max-pairs", type=int, default=20000,
                    help="Total hard pairs to add.")
    ap.add_argument("--neg-similarity", type=float, default=0.50,
                    help="Minimum title Jaccard for a negative to count as hard.")
    ap.add_argument("--pattern-similarity", type=float, default=0.15,
                    help="Lower bar for negatives showing a third-party or "
                         "form-factor pattern, which title similarity misses.")
    ap.add_argument("--pos-similarity", type=float, default=0.35,
                    help="Maximum title Jaccard for a positive to count as hard.")
    ap.add_argument("--check-only", action="store_true",
                    help="Report leakage and the pattern census; write nothing.")
    ap.add_argument("--train", default=OUT_TRAIN)
    ap.add_argument("--valid", default=OUT_VALID)
    args = ap.parse_args()

    print(f"Loading wdc/products-2017 ({args.size}) ...")
    lspc = load_lspc(args.categories, args.size)
    if "train" not in lspc:
        raise SystemExit("No LSPC data loaded.")
    pool = lspc["train"].copy()
    print(f"\nLSPC train pool: {len(pool):,} pairs "
          f"({int(pool.label.sum()):,} positive, {pool.label.mean():.1%})")

    # ---- leakage first, exactly as before ---------------------------------
    _, per_split = gold_standard_titles(args.wdc)
    report_leakage(pool, per_split)

    # ---- rank by hardness -------------------------------------------------
    print("\nscoring title similarity ...")
    pool["sim"] = [title_similarity(a, b) for a, b in zip(pool.raw_a, pool.raw_b)]

    # Title similarity alone misses two of the four failure shapes, measured on
    # the real examples:
    #
    #   JBL Tune 215 vs Tune 215TWS        sim 0.20  -- "215TWS" is one token
    #   Thirty First For Boat 141 vs boAt  sim 0.50  -- just under any sane cut
    #
    # Both are exactly what we are short of, so pairs exhibiting the
    # third-party or form-factor pattern are admitted at a much lower
    # similarity. This is still SELECTION over human labels, not labelling:
    # the pattern decides whether a pair is shown to the model, never what the
    # model is told about it.
    pool["patterns"] = [classify_patterns(a, b) for a, b in zip(pool.raw_a, pool.raw_b)]
    targeted = pool.patterns.map(
        lambda ps: ("third-party" in ps) or ("form-factor" in ps))

    is_hard_neg = (pool.label == 0) & (
        (pool.sim >= args.neg_similarity)
        | (targeted & (pool.sim >= args.pattern_similarity))
    )
    # DROP pairs whose serialized input is identical but whose label says
    # "different". The census turned up real ones:
    #
    #     "Tissot T Classic Couturier "@en   vs   "Tissot T Classic Couturier "@en
    #     "Puma EvoSpeed SL II Leather Tricks FG"@en  vs  (byte-identical)
    #
    # LSPC is right that these are different products -- the distinguishing
    # fact is in the cluster id -- but it is not in anything the model can see.
    # An unlearnable pair is not a hard example, it is label noise pointing the
    # wrong way: it teaches that identical input can mean "different", which
    # trades false positives for false negatives and would damage the easy and
    # medium slices currently at 100.00 and 89.23.
    identical = pool.text_a.str.strip() == pool.text_b.str.strip()
    n_identical = int((is_hard_neg & identical).sum())
    is_hard_neg = is_hard_neg & ~identical

    neg = pool[is_hard_neg]
    pos = pool[(pool.label == 1) & (pool.sim <= args.pos_similarity)]
    neg = neg.sort_values("sim", ascending=False)
    pos = pos.sort_values("sim", ascending=True)

    print(f"  hard negatives available: {len(neg):,}  "
          f"(dropped {n_identical:,} with byte-identical input -- unlearnable)")
    print(f"  hard positives available (sim <= {args.pos_similarity}): {len(pos):,}")

    # Hold the positive rate near the existing corpus so this does not simply
    # teach the model to say "different" more often.
    base = pd.read_csv(args.train) if os.path.exists(args.train) else pd.DataFrame()
    target_pos = float(base.label.mean()) if len(base) else 0.268
    n_pos = min(len(pos), int(round(args.max_pairs * target_pos)))
    n_neg = min(len(neg), args.max_pairs - n_pos)
    picked = pd.concat([neg.head(n_neg), pos.head(n_pos)], ignore_index=True)

    print(f"\n  taking {n_neg:,} negatives + {n_pos:,} positives = {len(picked):,}")
    print(f"  positive rate {picked.label.mean():.1%} "
          f"(existing corpus {target_pos:.1%})")
    print(f"  negative similarity range: "
          f"{neg.head(n_neg).sim.min():.2f} - {neg.head(n_neg).sim.max():.2f}")

    census(neg.head(n_neg), "PATTERN CENSUS -- mined hard negatives")
    census(pos.head(n_pos), "PATTERN CENSUS -- mined hard positives")

    print("\n  Examples of what was mined (negatives, hardest first):")
    for _, r in neg.head(6).iterrows():
        note = ""
        if str(r.raw_a).strip() == str(r.raw_b).strip():
            note = "   [titles identical -- signal is in the description]"
        print(f"    sim {r.sim:.2f}  {str(r.raw_a)[:64]}{note}")
        print(f"              vs  {str(r.raw_b)[:64]}")

    if args.check_only:
        print("\n--check-only: nothing written.")
        return

    cols = ["text_a", "text_b", "label", "source"]
    picked = picked.assign(source="LSPC-hard")
    train = pd.concat([base, picked[cols]], ignore_index=True)

    def dedupe(df):
        if df.empty:
            return df.reset_index(drop=True), 0
        k = [tuple(sorted((a, b))) for a, b in zip(df.text_a, df.text_b)]
        df = df.assign(_ka=[x[0] for x in k], _kb=[x[1] for x in k])
        n = len(df)
        df = df.drop_duplicates(subset=["_ka", "_kb", "label"]).drop(columns=["_ka", "_kb"])
        return df.reset_index(drop=True), n - len(df)

    train, dropped = dedupe(train)

    # valid is left alone on purpose: it drives early stopping, and changing
    # both training data and the selection signal at once would make the result
    # uninterpretable -- the same reason LSPC was kept out of valid.
    valid = pd.read_csv(args.valid) if os.path.exists(args.valid) else pd.DataFrame()
    if len(valid):
        key = {tuple(sorted((a, b))) for a, b in zip(train.text_a, train.text_b)}
        vk = [tuple(sorted((a, b))) for a, b in zip(valid.text_a, valid.text_b)]
        leaked = sum(1 for k in vk if k in key)
        if leaked:
            valid = valid[[k not in key for k in vk]].reset_index(drop=True)
            valid.to_csv(args.valid, index=False)
        print(f"\n  train/valid pair leak: {leaked} dropped from valid")

    train.to_csv(args.train, index=False)
    print(f"  deduplicated: -{dropped:,}")
    print(f"\nTRAIN {len(train):,} pairs ({int(train.label.sum()):,} positive, "
          f"{train.label.mean():.1%}) -> {args.train}")
    print("\ncomposition:")
    for src, n in train["source"].value_counts().items():
        print(f"  {src:<28} {n:>8,}  {n / len(train):6.1%}")
    print("\nValidation is unchanged, so early stopping and the calibration gate")
    print("still judge this model on the same yardstick as the previous three.")


if __name__ == "__main__":
    main()
