"""
build_indian_eval.py
====================
Merges the hand-labelled batches into one evaluation set of real Indian
e-commerce pairs, serialized exactly as training was.

    python build_indian_eval.py            -> data/indian_eval.csv

THIS IS AN EVALUATION SET, NOT TRAINING DATA
--------------------------------------------
~190 pairs cannot train a 142M-parameter model; they measure one. Every
benchmark in this project is a US/European catalog, so this is the first
measurement of the shipped model on the market it is actually aimed at.

Rows labelled EXCLUDE are dropped: the evidence in those listings was
irreconcilable (a title contradicting a 4x price, a stated wattage that cannot
be true for the named model, a title consisting only of the word "Mist Grey").
A wrong label in an eval set is worse than a missing one -- it penalises a
model for being right.

The binary target collapses the 3-class labels the way the shipped model works:

    EXACT_MATCH                     -> 1  (same purchasable product)
    SAME_PRODUCT_DIFFERENT_VARIANT  -> 0  (sibling SKU: colour, size, capacity)
    DIFFERENT_PRODUCT               -> 0

Variants map to 0 deliberately. A price-comparison engine that offers a 900 ml
bottle when the shopper asked for 520 ml is wrong, and the WDC/ER-Magellan
labels this model trained on draw the line the same way (different GTIN =
different product). `label3` keeps the finer distinction for analysis.
"""

import os

import pandas as pd

BATCHES = [f"data/labels_batch{i}.csv" for i in (1, 2, 3, 4)]
CANDIDATES = "data/label_candidates.csv"
OUT = "data/indian_eval.csv"

POSITIVE = {"EXACT_MATCH"}
NEGATIVE = {"SAME_PRODUCT_DIFFERENT_VARIANT", "DIFFERENT_PRODUCT"}


def _words(v, n: int) -> str:
    return " ".join(str(v or "").split(" ")[:n]).strip()


def serialize(title, brand="", description="") -> str:
    """The training serialization, verbatim. A different layout here would make
    every score meaningless while raising nothing (trap 2.3)."""
    return (f"COL brand VAL {_words(brand, 5)} "
            f"COL title VAL {_words(title, 50)} "
            f"COL description VAL {_words(description, 100)}").strip()


def main():
    cand = pd.read_csv(CANDIDATES).set_index("pair_id")
    lab = pd.concat([pd.read_csv(b) for b in BATCHES], ignore_index=True)

    dupes = lab[lab.pair_id.duplicated()].pair_id.tolist()
    if dupes:
        raise SystemExit(f"duplicate pair_ids across batches: {dupes}")
    missing = set(lab.pair_id) - set(cand.index)
    if missing:
        raise SystemExit(f"labelled ids not in candidates: {sorted(missing)}")

    print(f"labelled {len(lab)} of {len(cand)} candidates")
    print(lab.label.value_counts().to_string())

    keep = lab[lab.label.isin(POSITIVE | NEGATIVE)].copy()
    print(f"\ndropped {len(lab) - len(keep)} EXCLUDE rows -> {len(keep)} usable")

    rows = []
    for _, r in keep.iterrows():
        c = cand.loc[r.pair_id]
        rows.append({
            "pair_id": r.pair_id,
            "text_a": serialize(c.title_a, c.get("brand_a", "")),
            "text_b": serialize(c.title_b, c.get("brand_b", "")),
            "label": int(r.label in POSITIVE),
            "label3": r.label,
            "difficulty": r.difficulty,
            "source": c.candidate_source,
            "title_a": c.title_a,
            "title_b": c.title_b,
            "justification": r.justification,
            "uncertain": int(str(r.justification).startswith("UNSURE")),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False, encoding="utf-8")

    print(f"\nwrote {OUT}: {len(out)} pairs, {out.label.sum()} positive "
          f"({out.label.mean():.1%})")
    print("\nby 3-class label:")
    print(out.label3.value_counts().to_string())
    print("\nby difficulty:")
    print(out.difficulty.value_counts().to_string())
    print("\nby source:")
    print(out.source.value_counts().to_string())
    print(f"\nflagged uncertain: {int(out.uncertain.sum())} "
          f"({out.uncertain.mean():.1%}) -- reportable separately")


if __name__ == "__main__":
    main()
