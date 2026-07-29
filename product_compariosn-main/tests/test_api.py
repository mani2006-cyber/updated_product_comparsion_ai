from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import compare as compare_route


@dataclass
class _FakeResult:
    similarity_score: float
    prediction: str
    label: int
    relationship: str = None
    all_probabilities: dict = None


class _FakeComparer:
    def compare(self, title_a, title_b, **kwargs):
        if "Laptop" in title_b:
            return _FakeResult(similarity_score=99.0, prediction="UNRELATED", label=0, relationship="UNRELATED")
        return _FakeResult(similarity_score=91.0, prediction="SIMILAR_ALTERNATIVE", label=0, relationship="SIMILAR_ALTERNATIVE")


def _make_client():
    app = FastAPI()
    app.include_router(compare_route.router)
    app.state.comparer = _FakeComparer()
    return TestClient(app)


PAYLOAD = {
    "product": {"id": "P1", "title": "boAt Airdopes 300", "brand": "boAt", "description": "TWS earbuds, 50 hour battery"},
    "candidates": [
        {"id": "P2", "title": "GOBOULT W60", "brand": "GOBOULT", "description": "TWS earbuds, 40 hour battery"},
        {"id": "P3", "title": "MSI Modern 14 Laptop", "brand": "MSI", "description": "Core i7 laptop"},
    ],
}


def test_compare_returns_200_and_expected_shape():
    client = _make_client()
    resp = client.post("/compare", json=PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["original_product"] == "boAt Airdopes 300"
    assert body["exact_match"] is None
    assert "similar_products" in body


def test_compare_excludes_different_category_candidate():
    client = _make_client()
    resp = client.post("/compare", json=PAYLOAD)
    titles = [p["title"] for p in resp.json()["similar_products"]]
    assert "MSI Modern 14 Laptop" not in titles


def test_compare_rejects_empty_candidates():
    client = _make_client()
    bad_payload = {**PAYLOAD, "candidates": []}
    resp = client.post("/compare", json=bad_payload)
    assert resp.status_code == 422  # Pydantic validation error, min_length=1