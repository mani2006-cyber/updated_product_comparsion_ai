"""
api/main.py
=============
FastAPI entrypoint. Loads the trained ProductComparer ONCE at startup
(via lifespan), not per-request -- reloading a 286MB checkpoint on every
call would make this unusably slow.

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Then:
    POST http://localhost:8000/compare
    {
      "product": {"id": "P1", "title": "boAt Airdopes 300", "brand": "boAt",
                   "description": "50 hour battery, Bluetooth 5.3, AI-ENx"},
      "candidates": [
        {"id": "P2", "title": "GOBOULT W60", "brand": "GOBOULT", "description": "TWS earbuds, 40 hour battery"},
        {"id": "P3", "title": "MSI Modern 14 Laptop", "brand": "MSI", "description": "Core i7, 16GB RAM"}
      ]
    }
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

import config
from api.routes import compare, search
from exact_match.inference import ProductComparer
from utils import get_logger

logger = get_logger(__name__)

# Which checkpoint to serve, overridable with MODEL_DIR.
#
# There is deliberately NO fallback. An earlier version fell back to
# `trained_model/`, the original 5-class model trained on rule-generated
# labels, which scores 45.9 F1 against human-labelled pairs versus 75+ for the
# current one. Silently serving it would mean the service answered every
# request confidently and wrongly while reporting healthy -- strictly worse
# than refusing to start. A missing model now surfaces as 503.
_MODEL_CANDIDATES = [
    os.environ.get("MODEL_DIR"),
    os.path.join(config.ROOT_DIR, "trained_model_real"),
]


def resolve_model_dir() -> str:
    for path in _MODEL_CANDIDATES:
        if path and os.path.isfile(os.path.join(path, "config.json")):
            return path
    raise FileNotFoundError(
        "No usable model directory found. Set MODEL_DIR, or place a trained "
        f"model in {os.path.join(config.ROOT_DIR, 'trained_model_real')}."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup; reloading a 286MB checkpoint per request would make
    # the service unusable. A failure here leaves state.comparer unset, and the
    # route answers 503 rather than crashing every request with a 500.
    app.state.comparer = None
    app.state.model_dir = None
    app.state.index = None
    try:
        model_dir = resolve_model_dir()
        app.state.comparer = ProductComparer(model_dir=model_dir)
        app.state.model_dir = model_dir
        labels = app.state.comparer.model.config.id2label
        logger.info(f"Serving model from {model_dir} | classes: {labels}")
    except Exception:
        logger.exception("Model failed to load; /compare will return 503 until this is fixed.")

    # The catalog index is OPTIONAL and loaded separately. /compare works
    # without it (the caller supplies candidates); only /search needs it. A
    # missing index therefore degrades one endpoint rather than the service.
    index_dir = os.environ.get("INDEX_DIR", os.path.join(config.ROOT_DIR, "data", "product_index"))
    if os.path.isfile(os.path.join(index_dir, "index.faiss")):
        try:
            from ranking.embedding_retrieval import ProductIndex
            app.state.index = ProductIndex.load(index_dir)
            logger.info(f"Loaded product index from {index_dir} "
                        f"({len(app.state.index.products):,} products)")
        except Exception:
            logger.exception("Index failed to load; /search will return 503.")
    else:
        logger.warning(f"No product index at {index_dir}; /search will return 503. "
                       "Build one with: python -m ranking.embedding_retrieval build "
                       "--catalog <file> --out data/product_index")
    yield


app = FastAPI(title="Product Comparison API", lifespan=lifespan)
app.include_router(compare.router)
app.include_router(search.router)


@app.get("/health")
def health():
    """Reports whether the model actually loaded, not merely that the process
    is up -- a health check that always says "ok" cannot detect the one failure
    that matters here."""
    comparer = getattr(app.state, "comparer", None)
    if comparer is None:
        return {"status": "degraded", "model_loaded": False}
    index = getattr(app.state, "index", None)
    # Every field is read defensively. A health check that raises is worse
    # than one that reports "unknown": monitoring sees a 500 and cannot tell a
    # dead service from a introspection quirk.
    try:
        num_labels = len(comparer.model.config.id2label)
    except AttributeError:
        num_labels = None
    try:
        catalog_size = len(index.products) if index is not None else 0
    except AttributeError:
        catalog_size = None
    return {
        "status": "ok",
        "model_loaded": True,
        "model_dir": getattr(app.state, "model_dir", None),
        "num_labels": num_labels,
        "index_loaded": index is not None,
        "catalog_size": catalog_size,
    }