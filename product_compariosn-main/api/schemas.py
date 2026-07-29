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
    title: str
    similarity_score: float
    relationship: str
    reasons: List[str]


class CompareResponse(BaseModel):
    original_product: str
    exact_match: Optional[ExactMatch] = None
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