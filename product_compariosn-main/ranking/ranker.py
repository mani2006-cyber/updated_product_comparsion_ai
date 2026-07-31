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


def _score_shortlist(comparer, original: Dict, shortlist: List[Dict]) -> List:
    """Scores the whole shortlist in one batched call when the comparer
    supports it, falling back to per-pair compare() when it does not.

    The fallback is not dead code: ranking/test_ranking.py and tests/test_api.py
    drive this with a stub exposing only compare(), and those tests are the
    only coverage this logic has. Requiring score_pairs() would break them.
    """
    if not shortlist:
        return []

    pairs = [{
        "title_a": original.get("title", ""),
        "brand_a": original.get("brand", ""),
        "description_a": original.get("description", ""),
        "title_b": c.get("title", ""),
        "brand_b": c.get("brand", ""),
        "description_b": c.get("description", ""),
    } for c in shortlist]

    scorer = getattr(comparer, "score_pairs", None)
    if callable(scorer):
        return scorer(pairs)

    return [comparer.compare(**p) for p in pairs]


def rank_alternatives(
    original_product: Dict,
    candidate_pool: List[Dict] = None,
    comparer=None,
    top_n: int = 5,
    include_weakly_similar: bool = False,
    index=None,
    retrieve_k: int = 50,
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

    # Two retrieval modes:
    #   index given -> embedding + FAISS search over a whole catalog, so the
    #                  caller does not have to know the candidates in advance.
    #                  Measured recall@50 = 100% on the current catalog, which
    #                  is the ceiling on everything downstream: a match that is
    #                  never retrieved is never re-ranked.
    #   no index     -> the original keyword category filter over a pool the
    #                  caller supplies. Kept because every existing test drives
    #                  this path, and because it needs no index to be built.
    if index is not None:
        shortlist = index.search(original_product, k=retrieve_k)
    else:
        shortlist = retrieve_candidates(original_product, candidate_pool)
    results = _score_shortlist(comparer, original_product, shortlist)

    # EVERY candidate the model calls a match is kept. The previous version had
    # a single `exact_match` slot and `continue`d past the rest, so any other
    # match was deleted from the response -- not ranked lower, absent. Measured
    # on three Garnier listings scoring 99.9878 / 99.9853 / 99.9848 (all
    # label=1): one was returned and two vanished, leaving similar_products
    # empty. At a 0.003 margin which one won the slot was arbitrary.
    #
    # That destroys the primary use case: retrieval hands this ~50 candidates,
    # and several merchants selling one product is the normal case for a price
    # comparison engine, not an edge case.
    matches: List[Dict] = []
    scored = []

    for candidate, result in zip(shortlist, results):
        # A BINARY model returns relationship=None. Its prediction string is
        # "Same Product"/"Different Product", which matches none of the 5-class
        # names, so the old `relationship or prediction` fallback silently
        # classified every candidate as neither an exact match nor an
        # alternative -- every response came back empty. The two model shapes
        # are therefore handled explicitly.
        if result.relationship is None:
            if result.label == 1:
                matches.append({"title": candidate.get("title"),
                                "similarity_score": result.similarity_score})
                continue
            # Everything else is ranked by P(same product) and truncated by
            # top_n. No extra cut-off is invented here: the shortlist is
            # already category-filtered, and an arbitrary threshold is exactly
            # the mistake that produced the unusable WEAKLY_SIMILAR class.
            scored.append({
                "title": candidate.get("title"),
                "similarity_score": round(result.similarity_score / 100, 4),
                "relationship": "SIMILAR_ALTERNATIVE",
                "reasons": _build_reasons(original_product, candidate, "SIMILAR_ALTERNATIVE"),
            })
            continue

        relationship = result.relationship
        if relationship == "EXACT_MATCH":
            matches.append({"title": candidate.get("title"),
                            "similarity_score": result.similarity_score})
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

    # The best match keeps the `exact_match` slot for backwards compatibility;
    # the rest go to `other_matches` instead of being discarded. Truncated by
    # top_n, which is the caller's own explicit bound -- unlike the silent drop
    # this replaces.
    matches.sort(key=lambda m: m["similarity_score"], reverse=True)

    return {
        "original_product": original_product.get("title"),
        "exact_match": matches[0] if matches else None,
        "other_matches": matches[1:][:top_n],
        "similar_products": scored[:top_n],
    }