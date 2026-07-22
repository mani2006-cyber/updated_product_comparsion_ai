"""
ranking/candidate_retrieval.py
================================
Stage 6 (part 1) of the architecture: a cheap category/product-type
pre-filter over a candidate pool, run BEFORE the (comparatively expensive)
transformer similarity model scores anything.

Placeholder for the real classification/ module -- reuses the same
keyword-based categorize() heuristic already used in
generate_relationship_pairs.py, so retrieval and training-label generation
stay consistent. Swap both for the real trained category classifier
together, later -- nothing else in this file needs to change when you do.
"""

from typing import Dict, List

from generate_relationship_pairs import categorize


def build_candidate_text(product: Dict) -> str:
    return f"{product.get('title', '')} {product.get('description', '')}"


def retrieve_candidates(
    original_product: Dict,
    candidate_pool: List[Dict],
    max_candidates: int = 50,
) -> List[Dict]:
    """
    original_product / each item in candidate_pool: dict with at least
    'id', 'title', 'brand', 'description'.

    Returns a same-category shortlist, excluding the original product
    itself, capped at max_candidates. This is intentionally cheap (no
    model calls) -- it exists to avoid running the transformer over every
    candidate in a large catalog.
    """
    original_category = categorize(build_candidate_text(original_product))

    shortlist = []
    for candidate in candidate_pool:
        if candidate.get("id") == original_product.get("id"):
            continue
        candidate_category = categorize(build_candidate_text(candidate))
        if candidate_category == original_category:
            shortlist.append(candidate)

    return shortlist[:max_candidates]