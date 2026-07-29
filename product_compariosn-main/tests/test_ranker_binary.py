"""
Tests for the two changes that let the API serve the BINARY model
(trained_model_real) and score a shortlist in one batched call.

Both paths were previously uncovered:

  * A binary model returns relationship=None and prediction "Same Product".
    The old ranker did `relationship = result.relationship or result.prediction`
    and then compared against 5-class names, so nothing ever matched
    EXACT_MATCH or the alternative set -- every response came back empty while
    the service looked healthy.

  * Scoring looped over compare() one pair at a time: up to 50 sequential
    forward passes per request.

Everything here runs on stubs; no trained model is loaded.
"""

from dataclasses import dataclass
from typing import Dict, List

import pytest
from fastapi.testclient import TestClient

from ranking.ranker import rank_alternatives


@dataclass
class _Result:
    similarity_score: float
    prediction: str
    label: int
    relationship: str = None
    all_probabilities: dict = None


class _BinaryComparer:
    """Mimics the real binary model: relationship is always None."""

    def __init__(self, score_by_title: Dict[str, float], threshold: float = 50.0):
        self.score_by_title = score_by_title
        self.threshold = threshold
        self.batch_calls = 0

    def score_pairs(self, pairs: List[dict], **_):
        self.batch_calls += 1
        out = []
        for p in pairs:
            score = self.score_by_title[p["title_b"]]
            out.append(_Result(similarity_score=score,
                               prediction="Same Product" if score >= self.threshold else "Different Product",
                               label=int(score >= self.threshold)))
        return out


class _LegacyComparer:
    """Only exposes compare() -- the shape the older tests rely on."""

    def __init__(self, score_by_title):
        self.score_by_title = score_by_title
        self.compare_calls = 0

    def compare(self, title_a, title_b, **kwargs):
        self.compare_calls += 1
        score = self.score_by_title[title_b]
        return _Result(similarity_score=score,
                       prediction="Same Product" if score >= 50 else "Different Product",
                       label=int(score >= 50))


ORIGINAL = {"id": "P1", "title": "boAt Airdopes 300", "brand": "boAt",
            "description": "TWS earbuds, 50 hour battery"}

CANDIDATES = [
    {"id": "P2", "title": "GOBOULT W60", "brand": "GOBOULT", "description": "TWS earbuds, 40 hour battery"},
    {"id": "P3", "title": "NOISE Buds VS404", "brand": "NOISE", "description": "TWS earbuds, low latency"},
    {"id": "P4", "title": "Nike Air Max", "brand": "Nike", "description": "running shoe"},
    {"id": "P5", "title": "boAt Airdopes 300 Black", "brand": "boAt", "description": "TWS earbuds, 50 hour battery"},
]

SCORES = {"GOBOULT W60": 61.0, "NOISE Buds VS404": 42.0, "boAt Airdopes 300 Black": 97.0}


def test_binary_model_returns_an_exact_match():
    """The regression that would have shipped: empty responses for every call."""
    result = rank_alternatives(ORIGINAL, CANDIDATES, comparer=_BinaryComparer(SCORES))
    assert result["exact_match"] is not None, "binary model produced no exact match"
    assert result["exact_match"]["title"] == "boAt Airdopes 300 Black"


def test_binary_model_returns_ranked_alternatives():
    result = rank_alternatives(ORIGINAL, CANDIDATES, comparer=_BinaryComparer(SCORES))
    titles = [p["title"] for p in result["similar_products"]]
    assert titles, "binary model produced no alternatives"
    assert "boAt Airdopes 300 Black" not in titles      # the exact match is separated out
    assert "Nike Air Max" not in titles                 # category filter still applies
    scores = [p["similarity_score"] for p in result["similar_products"]]
    assert scores == sorted(scores, reverse=True)


def test_shortlist_is_scored_in_a_single_batched_call():
    comparer = _BinaryComparer(SCORES)
    rank_alternatives(ORIGINAL, CANDIDATES, comparer=comparer)
    assert comparer.batch_calls == 1, f"expected 1 batched call, got {comparer.batch_calls}"


def test_falls_back_to_compare_when_batching_is_unavailable():
    """Stubs without score_pairs must still work -- the older test suites
    depend on exactly that shape."""
    comparer = _LegacyComparer(SCORES)
    result = rank_alternatives(ORIGINAL, CANDIDATES, comparer=comparer)
    assert comparer.compare_calls > 0
    assert result["exact_match"]["title"] == "boAt Airdopes 300 Black"


def test_top_n_truncates_alternatives():
    result = rank_alternatives(ORIGINAL, CANDIDATES, comparer=_BinaryComparer(SCORES), top_n=1)
    assert len(result["similar_products"]) <= 1


def _client_with(comparer):
    from api.main import app
    app.state.comparer = comparer
    app.state.model_dir = "stub"
    return TestClient(app)


def _payload(n_candidates=2):
    return {
        "product": {"id": "P1", "title": "boAt Airdopes 300", "brand": "boAt", "description": "TWS"},
        "candidates": [{"id": f"C{i}", "title": "GOBOULT W60", "brand": "GOBOULT",
                        "description": "TWS earbuds"} for i in range(n_candidates)],
    }


def test_api_rejects_oversized_candidate_lists():
    """top_n was capped but the candidate list was not, so one request could
    force unbounded forward passes."""
    from api.routes.compare import MAX_CANDIDATES
    client = _client_with(_BinaryComparer({"GOBOULT W60": 61.0}))
    resp = client.post("/compare", json=_payload(MAX_CANDIDATES + 1))
    assert resp.status_code == 413


def test_api_returns_503_when_the_model_failed_to_load():
    """Previously an AttributeError surfaced as an opaque 500 on every call."""
    client = _client_with(None)
    resp = client.post("/compare", json=_payload())
    assert resp.status_code == 503


def test_health_reports_model_state():
    client = _client_with(None)
    body = client.get("/health").json()
    assert body["model_loaded"] is False
    assert body["status"] == "degraded"
