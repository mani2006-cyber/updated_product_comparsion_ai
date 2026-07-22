"""
generate_synthetic_pairs.py
=============================
Phase 1 / Pass 2 of the migration plan. Pass 1 (generate_relationship_pairs.py)
showed a severe imbalance on real data:
    UNRELATED 2298 | EXACT_MATCH 1675 | WEAKLY_SIMILAR 269
    SAME_PRODUCT_DIFFERENT_VARIANT 66 | SIMILAR_ALTERNATIVE 35
This script manufactures additional SAME_PRODUCT_DIFFERENT_VARIANT and
SIMILAR_ALTERNATIVE rows and merges them into relationship_pairs.csv, tagged
source="synthetic" so you can weight/inspect/drop them separately later.

Usage:
    python generate_synthetic_pairs.py \
        --input data/products_combined.csv \
        --relationship-pairs data/relationship_pairs.csv \
        --target-per-class 280
"""

import argparse
import random
from typing import Dict, List, Optional

import pandas as pd

from generate_relationship_pairs import SIMILAR_ALTERNATIVE_OVERLAP_THRESHOLD, categorize, extract_attributes, token_overlap_ratio

COLOR_ALTERNATIVES = ["black", "white", "silver", "blue", "red", "grey", "gold"]
RAM_ALTERNATIVES = ["4", "8", "16", "32"]
STORAGE_ALTERNATIVES = ["128gb", "256gb", "512gb", "1tb"]


def collect_unique_products(pair_df: pd.DataFrame) -> pd.DataFrame:
    """Both product1_* and product2_* columns reference real products --
    pull them apart into one row per unique product_id."""
    side_a = pair_df[["product1_id", "product1_title", "product1_brand", "product1_description"]].rename(
        columns=lambda c: c.replace("product1_", ""))
    side_b = pair_df[["product2_id", "product2_title", "product2_brand", "product2_description"]].rename(
        columns=lambda c: c.replace("product2_", ""))
    products = pd.concat([side_a, side_b], ignore_index=True)
    products = products.drop_duplicates(subset="id").reset_index(drop=True)
    products = products[products["title"].notna() & (products["title"].str.len() > 0)]
    return products


def _swap_one_attribute(title: str, description: str) -> Optional[Dict[str, str]]:
    """Tries color, then RAM, then storage. Returns modified (title, description)
    with exactly one attribute value swapped, or None if no swappable attribute found."""
    combined = f"{title} {description}".lower()

    for word in COLOR_ALTERNATIVES:
        if word in combined:
            replacement = random.choice([c for c in COLOR_ALTERNATIVES if c != word])
            new_title = title.lower().replace(word, replacement)
            new_desc = description.lower().replace(word, replacement) if isinstance(description, str) else description
            return {"title": new_title, "description": new_desc, "attribute": "color"}

    ram_match = next((r for r in RAM_ALTERNATIVES if f"{r}gb ram" in combined), None)
    if ram_match:
        replacement = random.choice([r for r in RAM_ALTERNATIVES if r != ram_match])
        new_title = title.lower().replace(f"{ram_match}gb ram", f"{replacement}gb ram")
        new_desc = description.lower().replace(f"{ram_match}gb ram", f"{replacement}gb ram") if isinstance(description, str) else description
        return {"title": new_title, "description": new_desc, "attribute": "ram"}

    storage_match = next((s for s in STORAGE_ALTERNATIVES if s in combined), None)
    if storage_match:
        replacement = random.choice([s for s in STORAGE_ALTERNATIVES if s != storage_match])
        new_title = title.lower().replace(storage_match, replacement)
        new_desc = description.lower().replace(storage_match, replacement) if isinstance(description, str) else description
        return {"title": new_title, "description": new_desc, "attribute": "storage"}

    return None


def build_variant_pairs(products: pd.DataFrame, target_count: int) -> List[Dict]:
    rows = []
    shuffled = products.sample(frac=1, random_state=42).reset_index(drop=True)
    for _, product in shuffled.iterrows():
        if len(rows) >= target_count:
            break
        swapped = _swap_one_attribute(str(product["title"]), str(product.get("description", "")))
        if swapped is None:
            continue
        rows.append({
            "product1_id": product["id"],
            "product1_title": product["title"],
            "product1_brand": product.get("brand", ""),
            "product1_description": product.get("description", ""),
            "product2_id": f"{product['id']}-variant-{swapped['attribute']}",
            "product2_title": swapped["title"],
            "product2_brand": product.get("brand", ""),
            "product2_description": swapped["description"],
            "relationship_label": "SAME_PRODUCT_DIFFERENT_VARIANT",
            "source": "synthetic",
        })
    return rows


def _sample_cross_brand_pairs(
    by_category: Dict[str, pd.DataFrame],
    overlap_min: float,
    overlap_max: float,
    target_count: int,
    relationship_label: str,
    seen_pairs: set,
) -> List[Dict]:
    """Shared sampling loop for both SIMILAR_ALTERNATIVE and WEAKLY_SIMILAR
    synthetic generation -- same category, different (cross-brand) products,
    filtered to a specific token-overlap band."""
    rows = []
    attempts = 0
    max_attempts = target_count * 30

    while len(rows) < target_count and attempts < max_attempts:
        attempts += 1
        category = random.choice(list(by_category.keys()))
        group = by_category[category]
        if len(group) < 2:
            continue
        a, b = group.sample(n=2, random_state=None).iloc[0], group.sample(n=1, random_state=None).iloc[0]
        if a["id"] == b["id"]:
            continue
        pair_key = tuple(sorted([a["id"], b["id"]]))
        if pair_key in seen_pairs:
            continue
        if str(a.get("brand", "")).strip().lower() == str(b.get("brand", "")).strip().lower():
            continue  # want cross-brand, not same-brand variants

        overlap = token_overlap_ratio(a["_text"], b["_text"])
        if not (overlap_min <= overlap < overlap_max):
            continue

        seen_pairs.add(pair_key)
        rows.append({
            "product1_id": a["id"],
            "product1_title": a["title"],
            "product1_brand": a.get("brand", ""),
            "product1_description": a.get("description", ""),
            "product2_id": b["id"],
            "product2_title": b["title"],
            "product2_brand": b.get("brand", ""),
            "product2_description": b.get("description", ""),
            "relationship_label": relationship_label,
            "source": "synthetic",
        })
    return rows


def _build_category_index(products: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    products = products.reset_index(drop=True)
    products["_text"] = products["title"].astype(str) + " " + products["description"].fillna("").astype(str)
    products["_category"] = products["_text"].map(categorize)
    return {
        cat: group for cat, group in products.groupby("_category") if cat != "OTHER" and len(group) >= 2
    }


def build_alternative_pairs(products: pd.DataFrame, target_count: int, seen_pairs: Optional[set] = None) -> List[Dict]:
    by_category = _build_category_index(products)
    seen_pairs = seen_pairs if seen_pairs is not None else set()
    # Aligned with SIMILAR_ALTERNATIVE_OVERLAP_THRESHOLD (0.25) from
    # generate_relationship_pairs.py -- previously this used 0.15 as the
    # floor, which contradicted the derived-label boundary and gave the
    # model conflicting signal for the 0.15-0.25 overlap band.
    return _sample_cross_brand_pairs(
        by_category, overlap_min=SIMILAR_ALTERNATIVE_OVERLAP_THRESHOLD, overlap_max=0.55,
        target_count=target_count, relationship_label="SIMILAR_ALTERNATIVE", seen_pairs=seen_pairs,
    )


def build_weakly_similar_pairs(products: pd.DataFrame, target_count: int, seen_pairs: Optional[set] = None) -> List[Dict]:
    """WEAKLY_SIMILAR had no dedicated synthetic generator before -- it only
    existed as whatever derived data happened to fall below the
    SIMILAR_ALTERNATIVE threshold, which is likely part of why it stayed
    the weakest class across every training run so far."""
    by_category = _build_category_index(products)
    seen_pairs = seen_pairs if seen_pairs is not None else set()
    return _sample_cross_brand_pairs(
        by_category, overlap_min=0.05, overlap_max=SIMILAR_ALTERNATIVE_OVERLAP_THRESHOLD,
        target_count=target_count, relationship_label="WEAKLY_SIMILAR", seen_pairs=seen_pairs,
    )


def generate(input_path: str, relationship_pairs_path: str, target_per_class: int) -> pd.DataFrame:
    pair_df = pd.read_csv(input_path)
    products = collect_unique_products(pair_df)

    variant_rows = build_variant_pairs(products, target_per_class)
    seen_pairs = set()
    alternative_rows = build_alternative_pairs(products, target_per_class, seen_pairs=seen_pairs)
    weakly_similar_rows = build_weakly_similar_pairs(products, target_per_class, seen_pairs=seen_pairs)

    synthetic_df = pd.DataFrame(variant_rows + alternative_rows + weakly_similar_rows)
    existing_df = pd.read_csv(relationship_pairs_path)

    merged = pd.concat([existing_df, synthetic_df], ignore_index=True)
    merged.to_csv(relationship_pairs_path, index=False)

    print(f"Added {len(variant_rows)} SAME_PRODUCT_DIFFERENT_VARIANT + "
          f"{len(alternative_rows)} SIMILAR_ALTERNATIVE + "
          f"{len(weakly_similar_rows)} WEAKLY_SIMILAR synthetic rows")
    print(merged["relationship_label"].value_counts())
    print("\nBy source:")
    print(merged["source"].value_counts())
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic variant/alternative pairs to fix class imbalance")
    parser.add_argument("--input", required=True, help="Original binary pair CSV, e.g. data/products_combined.csv")
    parser.add_argument("--relationship-pairs", default="data/relationship_pairs.csv",
                         help="Output of generate_relationship_pairs.py -- will be overwritten with the merged result")
    parser.add_argument("--target-per-class", type=int, default=280)
    args = parser.parse_args()

    generate(args.input, args.relationship_pairs, args.target_per_class)