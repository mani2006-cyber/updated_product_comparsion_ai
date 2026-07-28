"""
audit_weakly_similar.py
=======================
READ-ONLY audit. Writes one new file
(data/audit_weakly_similar.csv) and never modifies or deletes any
existing dataset.

PURPOSE
-------
Establish, from measurable product attributes only, WHY each row carrying a
given relationship_label carries it -- and expose the rows for which no
measurable justification exists.

WHY THIS IS NEEDED
------------------
Tracing the pipeline shows WEAKLY_SIMILAR is assigned by THREE independent
code paths that do not share a definition:

  PATH 1  generate_relationship_pairs.label_pair()
          -> product_taxonomy.classify_negative_pair()
          STRUCTURAL: same category + (different form factor OR >=3x tier gap)

  PATH 2  generate_relationship_pairs.label_pair()  [abstain fallback]
          ABSOLUTE THRESHOLD: token_overlap < 0.25

  PATH 3a generate_synthetic_pairs.build_weakly_similar_pairs()
          ABSOLUTE BAND: cross-brand, same keyword-category,
          0.05 <= token_overlap < 0.25

  PATH 3b generate_catalog_relationship_pairs
            .build_alternative_and_weakly_similar_pairs()
          RELATIVE PERCENTILE: cross-brand, same REAL category, overlap
          between that category's OWN p50 and p90

Path 3b is the critical one: it is per-category RELATIVE. A pair labelled
WEAKLY_SIMILAR inside "Movies & TV" was chosen against the movie overlap
distribution; one inside "Electronics" against the electronics distribution.
Nothing makes those two rows mean the same thing, and Path 1 (structural)
means a third thing again. That is why the class cannot be learned: the
label is not one concept.

This script does not fix that. It measures it, so the fix can be chosen on
evidence.

Usage:
    python audit_weakly_similar.py
    python audit_weakly_similar.py --data data/relationship_pairs_final.csv \\
        --out data/audit_weakly_similar.csv --label WEAKLY_SIMILAR
"""

import argparse
import os

import pandas as pd

from generate_relationship_pairs import token_overlap_ratio
from product_taxonomy import (
    UNKNOWN,
    TIER_RATIO_THRESHOLD,
    categorize,
    classify_negative_pair,
    extract_tier_specs,
    form_factor,
    tier_ratio,
)

NEGATIVE_LABELS = ("UNRELATED", "SIMILAR_ALTERNATIVE", "WEAKLY_SIMILAR")


# --------------------------------------------------------------------------
# Per-product tier descriptor
# --------------------------------------------------------------------------
def tier_descriptor(text: str) -> str:
    """A readable tier value for one product, e.g. "battery_hours=42".

    Returns "NO_SPEC" when the listing states none of the comparable specs.
    Deliberately reports the raw measured value rather than a bucket name
    like "premium"/"budget" -- bucketing would be an opinion, and the point
    of this audit is to separate measurements from opinions.
    """
    specs = extract_tier_specs(text)
    if not specs:
        return "NO_SPEC"
    return ", ".join(f"{k}={v:g}" for k, v in sorted(specs.items()))


# --------------------------------------------------------------------------
# Which code path produced this row?
# --------------------------------------------------------------------------
def identify_source_rule(origin: str, source: str, text_a: str, text_b: str) -> str:
    """Attributes a row to the code path that labelled it.

    `origin` (which merged file) and `source` (derived vs synthetic) are
    recorded by merge_all_sources.py; the taxonomy verdict is recomputed to
    tell PATH 1 from PATH 2, since both live inside label_pair().
    """
    origin = str(origin or "")
    source = str(source or "")

    if "catalog_relationship_pairs" in origin:
        return "PATH_3b_catalog_relative_percentile"

    if source == "synthetic" and "nearmiss" in origin:
        return "PATH_3a_synthetic_absolute_band"

    # Everything else went through label_pair(); recompute to see whether the
    # taxonomy claimed it or abstained to the Jaccard fallback.
    verdict = classify_negative_pair(text_a, text_b)
    if verdict is None:
        return "PATH_2_jaccard_fallback"
    return f"PATH_1_taxonomy->{verdict}"


# --------------------------------------------------------------------------
# Measurable justification for a WEAKLY_SIMILAR label
# --------------------------------------------------------------------------
def derive_weak_reason(text_a: str, text_b: str) -> tuple:
    """Returns (weak_reason, confidence).

    Every branch is decided by a measured attribute. There is no branch that
    invents a reason to make a row look justified -- rows whose label cannot
    be traced to any measurement come back as UNJUSTIFIED_*, which is the
    number this audit exists to surface.
    """
    cat_a, cat_b = categorize(text_a), categorize(text_b)

    # Cannot even establish the categories match -> no measurable basis at all.
    if cat_a == UNKNOWN or cat_b == UNKNOWN:
        return "UNJUSTIFIED_CATEGORY_UNKNOWN", "none"

    if cat_a != cat_b:
        # Different known categories: by the taxonomy this is UNRELATED, so a
        # WEAKLY_SIMILAR label here actively contradicts the measurement.
        return "UNJUSTIFIED_CATEGORIES_DIFFER", "none"

    ff_a, ff_b = form_factor(text_a), form_factor(text_b)
    if ff_a != UNKNOWN and ff_b != UNKNOWN and ff_a != ff_b:
        return "SAME_CATEGORY_DIFFERENT_FORM_FACTOR", "high"

    ratio = tier_ratio(text_a, text_b)
    if ratio is not None and ratio >= TIER_RATIO_THRESHOLD:
        return "SAME_CATEGORY_LARGE_TIER_GAP", "high"

    if ratio is not None and ratio > 1.0:
        # Same category, comparable magnitude, but the configurations are not
        # identical (e.g. 34h vs 42h battery, 8GB vs 16GB RAM).
        return "SAME_CATEGORY_DIFFERENT_CONFIGURATION", "medium"

    if ratio is not None:
        # Shared spec, identical values -> nothing separates these two.
        return "UNJUSTIFIED_SPECS_IDENTICAL", "none"

    # Same known category, cross-brand, but no comparable spec stated. Being
    # same-category-different-product is a real (if weak) relationship, so
    # this is justified -- but only weakly, hence low confidence.
    return "OTHER_JUSTIFIED_WEAK_RELATIONSHIP", "low"


def build_audit_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    subset = df[df["relationship_label"] == label].copy().reset_index(drop=True)

    text_a = (subset["product1_title"].fillna("") + " " + subset.get(
        "product1_description", pd.Series([""] * len(subset))).fillna("")).tolist()
    text_b = (subset["product2_title"].fillna("") + " " + subset.get(
        "product2_description", pd.Series([""] * len(subset))).fillna("")).tolist()

    origins = subset["origin"].tolist() if "origin" in subset else [""] * len(subset)
    sources = subset["source"].tolist() if "source" in subset else [""] * len(subset)

    reasons = [derive_weak_reason(a, b) for a, b in zip(text_a, text_b)]

    out = pd.DataFrame({
        "pair_id": [f"{label[:4]}-{i:06d}" for i in range(len(subset))],
        "product1_id": subset.get("product1_id", ""),
        "product2_id": subset.get("product2_id", ""),
        "product1_title": subset["product1_title"],
        "product2_title": subset["product2_title"],
        "product1_brand": subset.get("product1_brand", ""),
        "product2_brand": subset.get("product2_brand", ""),
        "product1_category": [categorize(t) for t in text_a],
        "product2_category": [categorize(t) for t in text_b],
        "product1_form_factor": [form_factor(t) for t in text_a],
        "product2_form_factor": [form_factor(t) for t in text_b],
        "product1_tier": [tier_descriptor(t) for t in text_a],
        "product2_tier": [tier_descriptor(t) for t in text_b],
        "relationship_label": subset["relationship_label"],
        "weak_reason": [r[0] for r in reasons],
        "source_rule": [identify_source_rule(o, s, a, b)
                        for o, s, a, b in zip(origins, sources, text_a, text_b)],
        "confidence": [r[1] for r in reasons],
        # Kept for inspection: the number both threshold-based paths keyed on.
        "token_overlap": [round(token_overlap_ratio(a, b), 4)
                          for a, b in zip(text_a, text_b)],
        "tier_ratio": [tier_ratio(a, b) for a, b in zip(text_a, text_b)],
        "origin_file": origins,
    })
    return out


def report_taxonomy_coverage(df: pd.DataFrame) -> None:
    """STEP 3: what the taxonomy says about EVERY negative row, with the
    abstain bucket shown explicitly rather than folded into a label."""
    neg = df[df["relationship_label"].isin(NEGATIVE_LABELS)]
    text_a = (neg["product1_title"].fillna("") + " " + neg["product1_description"].fillna("")).tolist()
    text_b = (neg["product2_title"].fillna("") + " " + neg["product2_description"].fillna("")).tolist()
    verdicts = [classify_negative_pair(a, b) for a, b in zip(text_a, text_b)]

    counts = pd.Series(["UNKNOWN/ABSTAIN" if v is None else f"taxonomy->{v}"
                        for v in verdicts]).value_counts()
    print("\n" + "=" * 72)
    print("STEP 3 -- taxonomy verdict on every negative-class row")
    print("=" * 72)
    total = len(neg)
    for name, n in counts.items():
        print(f"  {name:<34} {n:>7,}  ({n / total:6.2%})")
    print(f"  {'TOTAL negative rows':<34} {total:>7,}")


def main():
    parser = argparse.ArgumentParser(description="Audit why rows carry a given relationship label.")
    parser.add_argument("--data", default="data/relationship_pairs_final.csv")
    parser.add_argument("--out", default="data/audit_weakly_similar.csv")
    parser.add_argument("--label", default="WEAKLY_SIMILAR")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Loaded {len(df):,} rows from {args.data}")
    print("\nclass distribution in source file:")
    for name, n in df["relationship_label"].value_counts().items():
        print(f"  {name:<34} {n:>7,}  ({n / len(df):6.2%})")

    audit = build_audit_table(df, args.label)

    print("\n" + "=" * 72)
    print(f"STEP 2 -- {args.label}: which CODE PATH assigned the label")
    print("=" * 72)
    for name, n in audit["source_rule"].value_counts().items():
        print(f"  {name:<40} {n:>7,}  ({n / len(audit):6.2%})")

    print("\n" + "=" * 72)
    print(f"STEP 2 -- {args.label}: MEASURABLE justification")
    print("=" * 72)
    justified = 0
    for name, n in audit["weak_reason"].value_counts().items():
        mark = "  " if name.startswith("UNJUSTIFIED") else "OK"
        justified += n if not name.startswith("UNJUSTIFIED") else 0
        print(f"  {mark} {name:<42} {n:>7,}  ({n / len(audit):6.2%})")
    print(f"\n  justified by a measurable attribute: {justified:,} / {len(audit):,} "
          f"({justified / max(len(audit), 1):.2%})")

    report_taxonomy_coverage(df)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    audit.to_csv(args.out, index=False)
    print(f"\nWrote audit table -> {args.out}  ({len(audit):,} rows, {len(audit.columns)} columns)")
    print("No existing file was modified or deleted.")


if __name__ == "__main__":
    main()
