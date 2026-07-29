"""
api/routes/search.py
====================
POST /search -- given ONE product, finds its match in the indexed catalog.

HOW THIS DIFFERS FROM /compare
------------------------------
/compare answers "which of these candidates I am handing you is the match?"
The caller must already know the candidates, which is fine for testing and
useless in production: nobody knows the shortlist in advance.

/search answers the real question -- "find this product in my catalog" --
by running both stages:

    embedding + FAISS over the whole catalog   -> ~50 nearest neighbours
    cross-encoder re-ranks just those          -> the actual match

Neither model can do this alone. Measured on the current catalog, the
bi-encoder's recall@1 is 4.9% (a poor matcher) while its recall@50 is 100%
(an excellent filter); the cross-encoder is accurate but far too slow to score
a whole catalog. Hence two stages.

Requires an index built by:
    python -m ranking.embedding_retrieval build --catalog <file> --out data/product_index
"""

from fastapi import APIRouter, HTTPException, Request

from api.schemas import CompareResponse, SearchRequest
from ranking.ranker import rank_alternatives
from utils import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.post("/search", response_model=CompareResponse)
def search(payload: SearchRequest, request: Request):
    comparer = getattr(request.app.state, "comparer", None)
    index = getattr(request.app.state, "index", None)

    if comparer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded; service unavailable.")
    if index is None:
        # Distinct from the model being missing, and distinctly actionable:
        # the index is a separate artefact the operator has to build.
        raise HTTPException(
            status_code=503,
            detail="No product index loaded. Build one with: python -m "
                   "ranking.embedding_retrieval build --catalog <file> --out data/product_index",
        )

    try:
        return rank_alternatives(
            original_product=payload.product.model_dump(),
            comparer=comparer,
            index=index,
            retrieve_k=payload.retrieve_k,
            top_n=payload.top_n,
        )
    except Exception:
        logger.exception("search failed for title=%r", payload.product.title)
        raise HTTPException(status_code=500, detail="Search failed.")
