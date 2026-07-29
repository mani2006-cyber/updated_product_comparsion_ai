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
from api.routes import compare
from exact_match.inference import ProductComparer
from utils import get_logger

logger = get_logger(__name__)

# Which checkpoint to serve, overridable with MODEL_DIR.
#
# Default order matters. `trained_model/` holds the ORIGINAL 5-class model
# trained on rule-generated labels; measured against human-labelled WDC pairs
# it scores 45.9 F1. `trained_model_real/` is trained on 38k human-labelled
# pairs and beats published Ditto/RoBERTa baselines on 5 of 6 benchmarks. The
# better model is therefore preferred whenever it is present, with the old one
# kept only as a fallback so an existing deployment does not break.
_MODEL_CANDIDATES = [
    os.environ.get("MODEL_DIR"),
    os.path.join(config.ROOT_DIR, "trained_model_real"),
    config.TRAINED_MODEL_DIR,
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
    try:
        model_dir = resolve_model_dir()
        app.state.comparer = ProductComparer(model_dir=model_dir)
        app.state.model_dir = model_dir
        labels = app.state.comparer.model.config.id2label
        logger.info(f"Serving model from {model_dir} | classes: {labels}")
    except Exception:
        logger.exception("Model failed to load; /compare will return 503 until this is fixed.")
    yield


app = FastAPI(title="Product Comparison API", lifespan=lifespan)
app.include_router(compare.router)


@app.get("/health")
def health():
    """Reports whether the model actually loaded, not merely that the process
    is up -- a health check that always says "ok" cannot detect the one failure
    that matters here."""
    comparer = getattr(app.state, "comparer", None)
    if comparer is None:
        return {"status": "degraded", "model_loaded": False}
    return {
        "status": "ok",
        "model_loaded": True,
        "model_dir": getattr(app.state, "model_dir", None),
        "num_labels": len(comparer.model.config.id2label),
    }