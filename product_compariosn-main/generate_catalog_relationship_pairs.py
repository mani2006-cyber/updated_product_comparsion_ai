"""
generate_catalog_relationship_pairs.py
=========================================
Generates the 5-class relationship_pairs.csv directly from a flat product
catalog that has REAL category labels (balanced_catalog.csv) but no
pre-existing pairs at all -- unlike generate_relationship_pairs.py (which
refines existing binary label==0/1 pairs), every pair here is generated
from scratch:

    EXACT_MATCH                     -- same product, reworded title/phrasing
    SAME_PRODUCT_DIFFERENT_VARIANT  -- same product, one attribute swapped
                                        (reuses generate_synthetic_pairs.py's
                                        swap logic)
    SIMILAR_ALTERNATIVE             -- cross-brand pairs within the REAL
                                        category, overlap >= threshold
    WEAKLY_SIMILAR                  -- cross-brand pairs within the REAL
                                        category, overlap below threshold
    UNRELATED                       -- cross-category pairs

Usage:
    python generate_catalog_relationship_pairs.py \
        --input balanced_catalog.csv \
        --out catalog_relationship_pairs.csv \
        --target-per-class 5000
"""

import argparse
import random
from typing import Dict, List, Optional

import pandas as pd

from generate_relationship_pairs import token_overlap_ratio
from generate_synthetic_pairs import _swap_one_attribute, build_variant_pairs


def _build_subcategory_lookup(categories_txt_path: str, depth: int = 2) -> Dict[str, str]:
    """Re-parses categories.txt to build a FINER grouping than the coarse
    top-level category (e.g. 'Electronics, Portable Audio & Video' instead
    of just 'Electronics') -- the top-level alone is too broad for
    meaningful cross-brand SIMILAR_ALTERNATIVE/WEAKLY_SIMILAR sampling
    (verified: random pairs within top-level 'Electronics' average ~0.06
    token overlap, never reaching the 0.25 threshold)."""
    from filter_and_test_large_dataset import _parse_categories_file
    raw = _parse_categories_file(categories_txt_path)
    result = {}
    for pid, paths in raw.items():
        if not paths:
            continue
        segments = [s.strip() for s in paths[0].split(",")]
        result[pid] = ", ".join(segments[:depth])
    return result


def build_alternative_and_weakly_similar_pairs(
    by_category: Dict[str, pd.DataFrame],
    target_alt: int,
    target_weak: int,
    seen_pairs: set,
    min_group_size: int = 5,
    candidate_pool_size: int = 500,
    alt_percentile: float = 90,
    weak_percentile: float = 50,
) -> tuple:
    """Replaces the fixed-threshold _sample_cross_brand_pairs for this
    general, multi-domain catalog. A fixed overlap cutoff (0.25) only
    worked on the narrow audio-earbuds dataset because those listings
    repeat heavy boilerplate spec language ("TWS earbuds", "Bluetooth
    5.3", "battery hours"). General categories (movies, books, generic
    electronics) don't repeat phrases like that -- verified: even the
    single biggest, most specific subcategory ("Movies & TV, Movies",
    49,999 items) averages only ~0.055 token overlap between random
    pairs. So instead, each category's own overlap distribution is
    sampled, and the TOP ~10% becomes SIMILAR_ALTERNATIVE, the next
    ~40% becomes WEAKLY_SIMILAR -- relative to what's actually possible
    within that category, not an absolute number tuned for a different
    dataset."""
    import numpy as np

    alternative_rows: List[Dict] = []
    weakly_similar_rows: List[Dict] = []

    categories = [c for c, g in by_category.items() if len(g) >= min_group_size]
    if not categories:
        return alternative_rows, weakly_similar_rows

    max_category_visits = max(len(categories) * 20, 2000)  # global safety cap
    visits = 0

    while (len(alternative_rows) < target_alt or len(weakly_similar_rows) < target_weak) and visits < max_category_visits:
        category = random.choice(categories)
        visits += 1
        group = by_category[category]

        pool = []
        attempts = 0
        max_attempts = candidate_pool_size * 5
        while len(pool) < candidate_pool_size and attempts < max_attempts:
            attempts += 1
            a = random.choice(group)
            b = random.choice(group)
            if a["id"] == b["id"]:
                continue
            brand_a = str(a.get("brand", "") or "").strip().lower()
            brand_b = str(b.get("brand", "") or "").strip().lower()
            if brand_a and brand_a == brand_b:
                continue  # want cross-brand, not same-brand variants
            pair_key = tuple(sorted([a["id"], b["id"]]))
            if pair_key in seen_pairs:
                continue
            overlap = token_overlap_ratio(a["_text"], b["_text"])
            pool.append((a, b, overlap, pair_key))

        if len(pool) < 10:
            continue

        overlaps = [p[2] for p in pool]
        alt_cutoff = np.percentile(overlaps, alt_percentile)
        weak_cutoff = np.percentile(overlaps, weak_percentile)

        for a, b, overlap, pair_key in pool:
            if overlap >= alt_cutoff and len(alternative_rows) < target_alt:
                seen_pairs.add(pair_key)
                alternative_rows.append({
                    "product1_id": a["id"], "product1_title": a["title"],
                    "product1_brand": a.get("brand", ""), "product1_description": a.get("description", ""),
                    "product2_id": b["id"], "product2_title": b["title"],
                    "product2_brand": b.get("brand", ""), "product2_description": b.get("description", ""),
                    "relationship_label": "SIMILAR_ALTERNATIVE", "source": "synthetic",
                })
            elif weak_cutoff <= overlap < alt_cutoff and len(weakly_similar_rows) < target_weak:
                seen_pairs.add(pair_key)
                weakly_similar_rows.append({
                    "product1_id": a["id"], "product1_title": a["title"],
                    "product1_brand": a.get("brand", ""), "product1_description": a.get("description", ""),
                    "product2_id": b["id"], "product2_title": b["title"],
                    "product2_brand": b.get("brand", ""), "product2_description": b.get("description", ""),
                    "relationship_label": "WEAKLY_SIMILAR", "source": "synthetic",
                })

    return alternative_rows, weakly_similar_rows


def _build_real_category_index(products: pd.DataFrame, category_col: str = "category") -> Dict[str, list]:
    """Same shape as generate_synthetic_pairs.py's _build_category_index,
    but uses a real category column instead of guessing via the
    categorize() keyword heuristic, and returns plain Python lists of
    dict records (not DataFrames) -- repeatedly calling pandas .sample()
    inside a tight Python loop (needed when cycling through categories
    many times to hit a target count) is far too slow at this scale."""
    products = products.copy().reset_index(drop=True)
    products["_text"] = products["title"].astype(str) + " " + products["description"].fillna("").astype(str)
    products["_category"] = products[category_col]
    result = {}
    for cat, group in products.groupby("_category"):
        if pd.notna(cat) and len(group) >= 2:
            result[cat] = group[["id", "title", "brand", "description", "_text"]].to_dict("records")
    return result


def build_exact_match_pairs(products: pd.DataFrame, target_count: int) -> List[Dict]:
    """Same product, reworded title/phrasing, NO attribute change -- the
    genuine EXACT_MATCH case, distinct from build_variant_pairs (which
    always changes one attribute for SAME_PRODUCT_DIFFERENT_VARIANT)."""
    rows = []
    shuffled = products.sample(frac=1, random_state=7).reset_index(drop=True)
    for _, product in shuffled.iterrows():
        if len(rows) >= target_count:
            break
        title = str(product["title"])
        brand = str(product.get("brand", "") or "")
        if not brand or brand.lower() not in title.lower():
            continue  # need a brand token we can safely move around
        # Simple, safe reword: "Brand Title" <-> "Title by Brand" / "Title - Brand"
        stripped_title = title.replace(brand, "").strip(" -,")
        reworded = random.choice([
            f"{stripped_title} by {brand}",
            f"{stripped_title} - {brand}",
            f"{brand}: {stripped_title}",
        ])
        rows.append({
            "product1_id": product["id"],
            "product1_title": title,
            "product1_brand": brand,
            "product1_description": product.get("description", ""),
            "product2_id": f"{product['id']}-reworded",
            "product2_title": reworded,
            "product2_brand": brand,
            "product2_description": product.get("description", ""),
            "relationship_label": "EXACT_MATCH",
            "source": "synthetic",
        })
    return rows


def build_unrelated_pairs(products: pd.DataFrame, target_count: int) -> List[Dict]:
    """Cross-category pairs -- genuinely different products in different
    categories. With real category diversity (Books vs Electronics vs
    Shoes vs ...), this is easy to sample and gives clean, unambiguous
    negatives."""
    rows = []
    category_groups = {cat: group for cat, group in products.groupby("category") if len(group) >= 1}
    categories = list(category_groups.keys())
    if len(categories) < 2:
        return rows

    attempts = 0
    max_attempts = target_count * 20
    while len(rows) < target_count and attempts < max_attempts:
        attempts += 1
        cat_a, cat_b = random.sample(categories, 2)
        a = category_groups[cat_a].sample(n=1).iloc[0]
        b = category_groups[cat_b].sample(n=1).iloc[0]
        rows.append({
            "product1_id": a["id"],
            "product1_title": a["title"],
            "product1_brand": a.get("brand", ""),
            "product1_description": a.get("description", ""),
            "product2_id": b["id"],
            "product2_title": b["title"],
            "product2_brand": b.get("brand", ""),
            "product2_description": b.get("description", ""),
            "relationship_label": "UNRELATED",
            "source": "synthetic",
        })
    return rows


def generate(input_path: str, out_path: str, target_per_class: int, categories_txt_path: Optional[str] = None,
             candidate_pool_size: int = 500, alt_percentile: float = 90, weak_percentile: float = 50) -> pd.DataFrame:
    products = pd.read_csv(input_path)
    products = products[products["title"].notna() & (products["title"].astype(str).str.len() > 0)]

    exact_match_rows = build_exact_match_pairs(products, target_per_class)
    variant_rows = build_variant_pairs(products, target_per_class)

    if categories_txt_path:
        subcat_lookup = _build_subcategory_lookup(categories_txt_path)
        products["_subcategory"] = products["id"].map(subcat_lookup).fillna(products["category"])
        category_col_for_sampling = "_subcategory"
    else:
        category_col_for_sampling = "category"

    by_category = _build_real_category_index(products, category_col=category_col_for_sampling)
    seen_pairs = set()
    alternative_rows, weakly_similar_rows = build_alternative_and_weakly_similar_pairs(
        by_category, target_alt=target_per_class, target_weak=target_per_class, seen_pairs=seen_pairs,
        candidate_pool_size=candidate_pool_size, alt_percentile=alt_percentile, weak_percentile=weak_percentile,
    )
    unrelated_rows = build_unrelated_pairs(products, target_per_class)

    all_rows = exact_match_rows + variant_rows + alternative_rows + weakly_similar_rows + unrelated_rows
    result = pd.DataFrame(all_rows)
    result.to_csv(out_path, index=False)

    print(f"EXACT_MATCH: {len(exact_match_rows)} | SAME_PRODUCT_DIFFERENT_VARIANT: {len(variant_rows)} | "
          f"SIMILAR_ALTERNATIVE: {len(alternative_rows)} | WEAKLY_SIMILAR: {len(weakly_similar_rows)} | "
          f"UNRELATED: {len(unrelated_rows)}")
    print(f"Total: {len(result)} rows -> {out_path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 5-class relationship pairs from a real-category catalog")
    parser.add_argument("--input", required=True, help="e.g. balanced_catalog.csv")
    parser.add_argument("--out", default="catalog_relationship_pairs.csv")
    parser.add_argument("--target-per-class", type=int, default=5000)
    parser.add_argument("--categories", default=None,
                         help="Optional path to the original categories.txt, used to re-derive a finer "
                              "sub-category grouping for SIMILAR_ALTERNATIVE/WEAKLY_SIMILAR sampling. "
                              "Without this, the coarse top-level category is used, which is too broad "
                              "for meaningful overlap (verified: ~0.06 mean overlap within 'Electronics').")
    parser.add_argument("--candidate-pool-size", type=int, default=500,
                         help="How many candidate pairs to sample per category before taking percentiles. "
                              "Raise this (e.g. 2000) if SIMILAR_ALTERNATIVE comes in short of target -- "
                              "larger pools give more absolute rows even at the same percentile cutoff.")
    parser.add_argument("--alt-percentile", type=float, default=90,
                         help="Percentile cutoff for SIMILAR_ALTERNATIVE (top X%% of each category's overlap "
                              "distribution). Lower this (e.g. 80) to get more rows at the cost of pair quality.")
    parser.add_argument("--weak-percentile", type=float, default=50,
                         help="Percentile cutoff for WEAKLY_SIMILAR (everything between this and alt-percentile).")
    args = parser.parse_args()

    generate(args.input, args.out, args.target_per_class, categories_txt_path=args.categories,
             candidate_pool_size=args.candidate_pool_size, alt_percentile=args.alt_percentile,
             weak_percentile=args.weak_percentile)