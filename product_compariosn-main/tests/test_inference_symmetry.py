"""Order-invariance of pair scoring.

"Is A the same product as B" is symmetric, but a cross-encoder is not a
symmetric function -- [CLS] a [SEP] b [SEP] and [CLS] b [SEP] a [SEP] are
different inputs with different token_type_ids. Nothing in training pushed back
on it: build_real_corpus.py dedups with tuple(sorted(...)), so every pair
appears in exactly one order and the model never saw the mirror.

Measured on the 190-pair Indian set before the fix: median difference 0.06 pp,
p95 35.9 pp, worst 91.8 pp, and 8 pairs (4.2%) flipped across the decision
threshold purely on argument order.

ProductComparer._canonical sorts the serialized texts so both call orders
become the same forward pass. These tests cover the helper directly -- the
end-to-end check needs a real checkpoint and lives outside the suite.
"""

from exact_match.inference import ProductComparer

_canon = ProductComparer._canonical


def test_canonical_is_order_invariant():
    a, b = "COL title VAL alpha", "COL title VAL beta"
    assert _canon(a, b) == _canon(b, a)


def test_canonical_preserves_both_texts():
    a, b = "COL title VAL zebra", "COL title VAL apple"
    assert set(_canon(a, b)) == {a, b}, "canonicalising must not drop or alter a side"


def test_canonical_handles_identical_texts():
    a = "COL title VAL same"
    assert _canon(a, a) == (a, a)


def test_canonical_is_stable_across_repeated_calls():
    a, b = "COL title VAL boAt Airdopes 141", "COL title VAL boAt Airdopes 141 Pro"
    first = _canon(a, b)
    assert all(_canon(a, b) == first and _canon(b, a) == first for _ in range(5))


def test_canonical_on_the_worst_measured_asymmetry():
    """The pair that differed by 91.8 pp between orders."""
    a = ("COL brand VAL Mamaearth COL title VAL Mamaearth Rice Face Wash With "
         "Rice Water & Niacinamide for Glass Skin - 150 ml")
    b = ("COL brand VAL Mamaearth COL title VAL Mamaearth Vitamin C Face Wash "
         "for Skin Illumination Turmeric Aloe Vera")
    assert _canon(a, b) == _canon(b, a)
