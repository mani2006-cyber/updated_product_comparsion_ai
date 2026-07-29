"""
api/routes/compare.py
========================
POST /compare -- given one product and a candidate pool, returns the
exact match (if any) plus ranked similar-product alternatives.

Delegates entirely to ranking.ranker.rank_alternatives(), which already
does category filtering (ranking/candidate_retrieval.py), model scoring,
and exact-match separation. This route is just the HTTP wrapper + request
validation.
"""

from fastapi import APIRouter, HTTPException, Request

from api.schemas import CompareRequest, CompareResponse
from ranking.ranker import rank_alternatives
from utils import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Each candidate costs a transformer forward pass. `top_n` was capped but the
# candidate list was not, so a single request could submit an unbounded pool
# and occupy the service indefinitely -- a trivial denial-of-service vector.
MAX_CANDIDATES = 200


@router.post("/compare", response_model=CompareResponse)
def compare(payload: CompareRequest, request: Request):
    comparer = getattr(request.app.state, "comparer", None)
    if comparer is None:
        # Startup failed (missing/corrupt model directory). 503 says "try
        # later"; the previous behaviour was an AttributeError surfacing as an
        # opaque 500 with nothing in the logs.
        raise HTTPException(status_code=503, detail="Model is not loaded; service unavailable.")

    if len(payload.candidates) > MAX_CANDIDATES:
        raise HTTPException(
            status_code=413,
            detail=f"Too many candidates: {len(payload.candidates)} (max {MAX_CANDIDATES}).",
        )

    try:
        return rank_alternatives(
            original_product=payload.product.model_dump(),
            candidate_pool=[c.model_dump() for c in payload.candidates],
            comparer=comparer,
            top_n=payload.top_n,
            include_weakly_similar=payload.include_weakly_similar,
        )
    except Exception:
        # Log the traceback but do not leak internals to the caller. Without
        # this every failure was an unhandled 500 with no server-side record,
        # making production issues undiagnosable.
        logger.exception("compare failed for product id=%s with %d candidates",
                         payload.product.id, len(payload.candidates))
        raise HTTPException(status_code=500, detail="Comparison failed.")