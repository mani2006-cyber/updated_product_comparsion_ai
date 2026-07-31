"""
api/schemas.py
================
Pydantic models for the /compare endpoint, matching the original spec's
JSON shape:

    {
      "original_product": "...",
      "exact_match": {...} | null,
      "similar_products": [
        {"title": ..., "similarity_score": ..., "relationship": ..., "reasons": [...]}
      ]
    }
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class Product(BaseModel):
    id: str
    title: str
    brand: Optional[str] = ""
    description: Optional[str] = ""


class CompareRequest(BaseModel):
    product: Product
    candidates: List[Product] = Field(..., min_length=1)
    top_n: int = Field(default=5, ge=1, le=50)
    include_weakly_similar: bool = False


class ExactMatch(BaseModel):
    title: str
    similarity_score: float


class SimilarProduct(BaseModel):
    """A candidate the model judged NOT to be the same product.

    THIS IS NOT A RANKED LIST OF ALTERNATIVES, despite the name. The served
    model is binary -- it answers "is this the same purchasable product?" and
    nothing else. For every genuine alternative the answer is a confident no,
    so `similarity_score` collapses to near zero for all of them and the
    ordering between them carries no information.

    Measured, comparing a Philips air fryer against three candidates:

        Havells air fryer      0.02
        Morphy Richards OTG    0.01
        Logitech keyboard      0.01

    A competing air fryer and a Bluetooth keyboard land within 0.01 of each
    other. Do not present this order to a user as "most similar first", and do
    not threshold on the score to decide relatedness -- it measures the
    probability of being the SAME product, which is uniformly ~0 here.

    `relationship` is always "SIMILAR_ALTERNATIVE" on the binary model. That is
    the label given to anything that is not a match; it is not a judgment that
    the two items are similar.

    Ranking true alternatives needs a relatedness signal the cross-encoder does
    not produce. The bi-encoder behind /search does produce one (cosine
    similarity over embeddings) and is the intended basis for this later.
    """

    title: str
    similarity_score: float   # 0-100, same scale as ExactMatch
    relationship: str
    reasons: List[str]


class CompareResponse(BaseModel):
    original_product: str
    exact_match: Optional[ExactMatch] = None
    # Every OTHER candidate the model also judged to be the same product,
    # best-first. `exact_match` is a single slot, but several merchants listing
    # one product is the normal case for a price comparison engine -- these used
    # to be silently discarded. Defaults to [] so existing clients are
    # unaffected; the field is additive.
    other_matches: List[ExactMatch] = []
    similar_products: List[SimilarProduct]


class SearchQuery(BaseModel):
    """A product to look up. `id` is optional here, unlike Product: a caller
    searching an external listing against our catalog has no id of ours to
    give, and requiring a placeholder would be pointless ceremony."""
    title: str = Field(..., min_length=1)
    brand: Optional[str] = ""
    description: Optional[str] = ""
    id: Optional[str] = None


class SearchRequest(BaseModel):
    product: SearchQuery
    top_n: int = Field(default=5, ge=1, le=50)
    # How many neighbours the embedding stage retrieves before re-ranking.
    # This is the accuracy/latency dial: measured recall@10 was 85.4% and
    # recall@50 100%, while each extra candidate costs one cross-encoder
    # forward pass. Capped at 200 so a single request cannot force unbounded
    # work.
    retrieve_k: int = Field(default=50, ge=1, le=200)