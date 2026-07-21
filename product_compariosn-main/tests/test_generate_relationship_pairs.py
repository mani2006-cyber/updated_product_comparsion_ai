import pandas as pd

from generate_relationship_pairs import label_pair


def _row(**kwargs):
    base = {
        "product1_id": "A1", "product1_title": "", "product1_brand": "", "product1_description": "",
        "product2_id": "B1", "product2_title": "", "product2_brand": "", "product2_description": "",
        "label": 0,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_identical_attributes_is_exact_match():
    row = _row(
        product1_title="Samsung Galaxy Watch7 44mm Black", product1_description="Bluetooth smartwatch",
        product2_title="Galaxy Watch7 44mm Black - Samsung", product2_description="Bluetooth smartwatch",
        label=1,
    )
    assert label_pair(row) == "EXACT_MATCH"


def test_differing_color_is_variant():
    row = _row(
        product1_title="MSI Modern 14 1TB SSD Silver", product1_description="8GB RAM",
        product2_title="MSI Modern 14 1TB SSD Black", product2_description="8GB RAM",
        label=1,
    )
    assert label_pair(row) == "SAME_PRODUCT_DIFFERENT_VARIANT"


def test_different_category_is_unrelated():
    row = _row(
        product1_title="boAt Airdopes 300 TWS earbuds", product1_description="50 hour battery",
        product2_title="Nike running shoes", product2_description="Comfortable sneaker",
        label=0,
    )
    assert label_pair(row) == "UNRELATED"


def test_same_category_high_overlap_is_similar_alternative():
    row = _row(
        product1_title="boAt Airdopes 300 TWS earbuds Bluetooth", product1_description="AI-ENx low latency",
        product2_title="Noise Buds VS404 TWS earbuds Bluetooth", product2_description="AI-ENx low latency",
        label=0,
    )
    assert label_pair(row) == "SIMILAR_ALTERNATIVE"


def test_same_category_low_overlap_is_weakly_similar():
    row = _row(
        product1_title="boAt Airdopes 300 TWS earbuds", product1_description="basic model",
        product2_title="Sony WF earbuds", product2_description="premium noise cancelling flagship",
        label=0,
    )
    assert label_pair(row) == "WEAKLY_SIMILAR"