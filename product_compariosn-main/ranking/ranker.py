"""
ranking/ranker.py
==================
Stage 6 (part 2): scores a category-filtered candidate shortlist against
one original product using the trained 5-class relationship model
(exact_match/inference.py's ProductComparer), and assembles the final
response matching the JSON shape from the original spec:

    {
      "original_product": "...",
      "exact_match": {...} | None,
      "similar_products": [
        {"title": ..., "similarity_score": ..., "relationship": ..., "reasons": [...]},
        ...
      ]
    }
"""

from typing import Dict, List, Optional

from ranking.candidate_retrieval import retrieve_candidates

ALTERNATIVE_RELATIONSHIPS = {"SAME_PRODUCT_DIFFERENT_VARIANT", "SIMILAR_ALTERNATIVE"}


def _build_reasons(original: Dict, candidate: Dict, relationship: str) -> List[str]:
    """Lightweight, explainable reasons -- placeholder for the richer
    attribute-diff reasons the attribute_extraction/ module will enable
    later (e.g. "Similar battery life: 50 hours")."""
    reasons = []
    original_brand = (original.get("brand") or "").strip().lower()
    candidate_brand = (candidate.get("brand") or "").strip().lower()
    if original_brand and candidate_brand:
        if original_brand == candidate_brand:
            reasons.append(f"Same brand: {candidate.get('brand')}")
        else:
            reasons.append(f"Different brand: {candidate.get('brand')} vs {original.get('brand')}")
    reasons.append(f"Relationship: {relationship.replace('_', ' ').title()}")
    return reasons


def rank_alternatives(
    original_product: Dict,
    candidate_pool: List[Dict],
    comparer=None,
    top_n: int = 5,
    include_weakly_similar: bool = False,
) -> Dict:
    """
    original_product: dict with 'id', 'title', 'brand', 'description'
    candidate_pool: list of such dicts to compare against
    comparer: a loaded ProductComparer (pass one in to avoid re-loading the
              model on every call). If None, a fresh one is loaded here --
              only do this for one-off calls, not in a loop.
    include_weakly_similar: if True, WEAKLY_SIMILAR candidates are included
              in similar_products as a fallback when better alternatives
              are scarce. Off by default since WEAKLY_SIMILAR is currently
              the model's weakest class (see evaluation notes).
    """
    if comparer is None:
        from exact_match.inference import ProductComparer
        comparer = ProductComparer()

    shortlist = retrieve_candidates(original_product, candidate_pool)

    exact_match = None
    scored = []

    for candidate in shortlist:
        result = comparer.compare(
            title_a=original_product.get("title", ""),
            brand_a=original_product.get("brand", ""),
            description_a=original_product.get("description", ""),
            title_b=candidate.get("title", ""),
            brand_b=candidate.get("brand", ""),
            description_b=candidate.get("description", ""),
        )
        relationship = result.relationship or result.prediction

        if relationship == "EXACT_MATCH":
            if exact_match is None or result.similarity_score > exact_match["similarity_score"]:
                exact_match = {"title": candidate.get("title"), "similarity_score": result.similarity_score}
            continue

        include = relationship in ALTERNATIVE_RELATIONSHIPS or (
            include_weakly_similar and relationship == "WEAKLY_SIMILAR"
        )
        if include:
            scored.append({
                "title": candidate.get("title"),
                "similarity_score": round(result.similarity_score / 100, 4),
                "relationship": relationship,
                "reasons": _build_reasons(original_product, candidate, relationship),
            })

    scored.sort(key=lambda r: r["similarity_score"], reverse=True)

    return {
        "original_product": original_product.get("title"),
        "exact_match": exact_match,
        "similar_products": scored[:top_n],
    }