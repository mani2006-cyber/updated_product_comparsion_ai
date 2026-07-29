"""
Tests for POST /search -- catalog search via embedding retrieval + re-ranking.

/compare requires the caller to already know the candidate shortlist, which is
fine for testing and useless in production. /search answers the real question
("find this product in my catalog") by running both stages. These tests cover
the wiring and the failure modes; retrieval quality itself is measured
separately by `ranking.embedding_retrieval evaluate` (recall@k).

Everything here runs on stubs -- no model, no FAISS index, no downloads.
"""

from dataclasses import dataclass
from typing import Dict, List

from fastapi.testclient import TestClient

from api.main import app


@dataclass
class _Result:
    similarity_score: float
    prediction: str
    label: int
    relationship: str = None
    all_probabilities: dict = None


class _StubComparer:
    """Binary comparer: scores by title, like the real trained_model_real."""

    def __init__(self, score_by_title: Dict[str, float]):
        self.score_by_title = score_by_title

    def score_pairs(self, pairs: List[dict], **_):
        out = []
        for p in pairs:
            s = self.score_by_title.get(p["title_b"], 0.0)
            out.append(_Result(similarity_score=s,
                               prediction="Same Product" if s >= 50 else "Different Product",
                               label=int(s >= 50)))
        return out


class _StubIndex:
    """Stands in for ProductIndex, recording the k it was asked for so the
    retrieve_k wiring can be asserted rather than assumed."""

    def __init__(self, products: List[Dict]):
        self.products = products
        self.last_k = None

    def search(self, query: Dict, k: int = 50, exclude_self: bool = True):
        self.last_k = k
        return [dict(p, _retrieval_score=0.9) for p in self.products[:k]]


CATALOG = [
    {"id": "C1", "title": "boAt Airdopes 141 TWS Earphones - Bold Black 1 pc", "brand": "boAt", "description": ""},
    {"id": "C2", "title": "boAt Airdopes 148 Gen 2", "brand": "boAt", "description": ""},
    {"id": "C3", "title": "MSI Modern 14 Laptop", "brand": "MSI", "description": ""},
]

SCORES = {
    "boAt Airdopes 141 TWS Earphones - Bold Black 1 pc": 99.0,
    "boAt Airdopes 148 Gen 2": 12.0,
    "MSI Modern 14 Laptop": 0.1,
}


def _client(comparer=None, index=None):
    app.state.comparer = comparer
    app.state.index = index
    app.state.model_dir = "stub"
    return TestClient(app)


def _payload(**over):
    body = {"product": {"title": "boAt Airdopes 141 Bold Black"}, "top_n": 5}
    body.update(over)
    return body


def test_search_finds_the_match_without_caller_supplied_candidates():
    """The whole point of /search: no `candidates` field in the request."""
    client = _client(_StubComparer(SCORES), _StubIndex(CATALOG))
    resp = client.post("/search", json=_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["exact_match"] is not None
    assert body["exact_match"]["title"] == "boAt Airdopes 141 TWS Earphones - Bold Black 1 pc"


def test_search_returns_ranked_alternatives():
    client = _client(_StubComparer(SCORES), _StubIndex(CATALOG))
    body = client.post("/search", json=_payload()).json()
    titles = [p["title"] for p in body["similar_products"]]
    assert "boAt Airdopes 141 TWS Earphones - Bold Black 1 pc" not in titles  # separated out
    scores = [p["similarity_score"] for p in body["similar_products"]]
    assert scores == sorted(scores, reverse=True)


def test_retrieve_k_is_passed_through_to_the_index():
    """retrieve_k is the accuracy/latency dial; silently ignoring it would be
    invisible in the response."""
    index = _StubIndex(CATALOG)
    client = _client(_StubComparer(SCORES), index)
    client.post("/search", json=_payload(retrieve_k=7))
    assert index.last_k == 7


def test_search_503s_when_no_index_is_loaded():
    """Distinct from the model being missing: the index is a separate artefact
    the operator must build, so the error names the command."""
    client = _client(_StubComparer(SCORES), None)
    resp = client.post("/search", json=_payload())
    assert resp.status_code == 503
    assert "embedding_retrieval build" in resp.json()["detail"]


def test_search_503s_when_the_model_is_missing():
    client = _client(None, _StubIndex(CATALOG))
    assert client.post("/search", json=_payload()).status_code == 503


def test_search_rejects_an_empty_title():
    client = _client(_StubComparer(SCORES), _StubIndex(CATALOG))
    resp = client.post("/search", json={"product": {"title": ""}})
    assert resp.status_code == 422


def test_retrieve_k_is_capped():
    """An uncapped k would let one request force unbounded cross-encoder passes."""
    client = _client(_StubComparer(SCORES), _StubIndex(CATALOG))
    assert client.post("/search", json=_payload(retrieve_k=5000)).status_code == 422


def test_health_reports_index_state():
    client = _client(_StubComparer(SCORES), _StubIndex(CATALOG))
    body = client.get("/health").json()
    assert body["index_loaded"] is True
    assert body["catalog_size"] == len(CATALOG)

    body = _client(_StubComparer(SCORES), None).get("/health").json()
    assert body["index_loaded"] is False
