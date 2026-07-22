from dataclasses import dataclass
from typing import Dict

from ranking.ranker import rank_alternatives


@dataclass
class _FakeResult:
    similarity_score: float
    prediction: str
    label: int
    relationship: str = None
    all_probabilities: dict = None


class _FakeComparer:
    """Stubs ProductComparer.compare() with scripted relationships/scores,
    keyed by candidate title, so ranking logic can be tested without
    loading the real trained model."""

    def __init__(self, relationship_by_title: Dict[str, str], score_by_title: Dict[str, float]):
        self.relationship_by_title = relationship_by_title
        self.score_by_title = score_by_title

    def compare(self, title_a, title_b, **kwargs):
        relationship = self.relationship_by_title[title_b]
        score = self.score_by_title[title_b]
        return _FakeResult(similarity_score=score, prediction=relationship, label=0, relationship=relationship)


ORIGINAL = {"id": "P1", "title": "boAt Airdopes 300", "brand": "boAt", "description": "TWS earbuds, 50 hour battery"}

CANDIDATES = [
    {"id": "P2", "title": "GOBOULT W60", "brand": "GOBOULT", "description": "TWS earbuds, 40 hour battery"},
    {"id": "P3", "title": "NOISE Buds VS404", "brand": "NOISE", "description": "TWS earbuds, low latency"},
    {"id": "P4", "title": "Nike Air Max", "brand": "Nike", "description": "running shoe"},  # different category
    {"id": "P5", "title": "boAt Airdopes 300 Black", "brand": "boAt", "description": "TWS earbuds, 50 hour battery"},
]


def test_excludes_different_category_candidates():
    comparer = _FakeComparer(
        relationship_by_title={
            "GOBOULT W60": "SIMILAR_ALTERNATIVE",
            "NOISE Buds VS404": "SIMILAR_ALTERNATIVE",
            "boAt Airdopes 300 Black": "EXACT_MATCH",
        },
        score_by_title={"GOBOULT W60": 91.0, "NOISE Buds VS404": 85.0, "boAt Airdopes 300 Black": 99.0},
    )
    result = rank_alternatives(ORIGINAL, CANDIDATES, comparer=comparer)
    titles = [p["title"] for p in result["similar_products"]]
    assert "Nike Air Max" not in titles


def test_exact_match_is_separated_from_similar_products():
    comparer = _FakeComparer(
        relationship_by_title={
            "GOBOULT W60": "SIMILAR_ALTERNATIVE",
            "NOISE Buds VS404": "SIMILAR_ALTERNATIVE",
            "boAt Airdopes 300 Black": "EXACT_MATCH",
        },
        score_by_title={"GOBOULT W60": 91.0, "NOISE Buds VS404": 85.0, "boAt Airdopes 300 Black": 99.0},
    )
    result = rank_alternatives(ORIGINAL, CANDIDATES, comparer=comparer)
    assert result["exact_match"]["title"] == "boAt Airdopes 300 Black"
    titles = [p["title"] for p in result["similar_products"]]
    assert "boAt Airdopes 300 Black" not in titles


def test_similar_products_sorted_by_score_descending():
    comparer = _FakeComparer(
        relationship_by_title={
            "GOBOULT W60": "SIMILAR_ALTERNATIVE",
            "NOISE Buds VS404": "SIMILAR_ALTERNATIVE",
            "boAt Airdopes 300 Black": "EXACT_MATCH",
        },
        score_by_title={"GOBOULT W60": 91.0, "NOISE Buds VS404": 85.0, "boAt Airdopes 300 Black": 99.0},
    )
    result = rank_alternatives(ORIGINAL, CANDIDATES, comparer=comparer)
    scores = [p["similarity_score"] for p in result["similar_products"]]
    assert scores == sorted(scores, reverse=True)


def test_weakly_similar_excluded_by_default():
    comparer = _FakeComparer(
        relationship_by_title={
            "GOBOULT W60": "WEAKLY_SIMILAR",
            "NOISE Buds VS404": "UNRELATED",
            "boAt Airdopes 300 Black": "SAME_PRODUCT_DIFFERENT_VARIANT",
        },
        score_by_title={"GOBOULT W60": 60.0, "NOISE Buds VS404": 70.0, "boAt Airdopes 300 Black": 95.0},
    )
    result = rank_alternatives(ORIGINAL, CANDIDATES, comparer=comparer)
    titles = [p["title"] for p in result["similar_products"]]
    assert titles == ["boAt Airdopes 300 Black"]


def test_weakly_similar_included_when_requested():
    comparer = _FakeComparer(
        relationship_by_title={
            "GOBOULT W60": "WEAKLY_SIMILAR",
            "NOISE Buds VS404": "UNRELATED",
            "boAt Airdopes 300 Black": "SAME_PRODUCT_DIFFERENT_VARIANT",
        },
        score_by_title={"GOBOULT W60": 60.0, "NOISE Buds VS404": 70.0, "boAt Airdopes 300 Black": 95.0},
    )
    result = rank_alternatives(ORIGINAL, CANDIDATES, comparer=comparer, include_weakly_similar=True)
    titles = [p["title"] for p in result["similar_products"]]
    assert "GOBOULT W60" in titles
    assert "NOISE Buds VS404" not in titles  # UNRELATED stays excluded regardless of the flag