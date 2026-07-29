"""
Regression tests for train/val/test leakage.

Guards two defects measured on the real dataset:

  1. ENTITY LEAKAGE -- 29.48% of test rows involved a product that also
     appeared in a training pair, because split_data() split at the pair-row
     level. Fixed by grouping pairs into connected components of the product
     graph and moving whole components together.

  2. ORDER-DEPENDENT DEDUP -- clean_dataframe() deduplicated on
     (text_a, text_b), so (A,B) and (B,A) both survived and could land in
     different splits. 328 such mirrored pairs existed; 2 leaked into test.

These run on small synthetic frames -- no trained model, no data files.
"""

import pandas as pd
import pytest

import config
from exact_match import preprocessing


def _pair_frame(n_pairs: int = 60) -> pd.DataFrame:
    """Builds clusters of three products wired together by three pairs.

    Products deliberately RECUR across pairs, mirroring the real dataset where
    one listing appears in several pairs. A fixture of fully disjoint pairs
    would pass even under the old row-level split, since there would be no
    shared product to leak -- it would test nothing.
    """
    rows = []
    for cluster in range((n_pairs + 2) // 3):
        a = f"product a{cluster} | brand x{cluster}"
        b = f"product b{cluster} | brand y{cluster}"
        c = f"product c{cluster} | brand z{cluster}"
        rows += [(a, b, cluster % 5), (b, c, (cluster + 1) % 5), (a, c, (cluster + 2) % 5)]
    return pd.DataFrame(rows[:n_pairs], columns=["text_a", "text_b", "label"])


def _entities(df: pd.DataFrame) -> set:
    return set(df["text_a"]) | set(df["text_b"])


def test_no_entity_overlap_between_splits():
    train, val, test = preprocessing.split_data(_pair_frame(120))
    assert not (_entities(train) & _entities(test)), "product leaked train -> test"
    assert not (_entities(train) & _entities(val)), "product leaked train -> val"
    assert not (_entities(val) & _entities(test)), "product leaked val -> test"


def test_no_exact_pair_overlap_between_splits():
    train, val, test = preprocessing.split_data(_pair_frame(120))

    def keys(df):
        return {tuple(sorted([a, b])) for a, b in zip(df["text_a"], df["text_b"])}

    assert not (keys(train) & keys(test))
    assert not (keys(train) & keys(val))
    assert not (keys(val) & keys(test))


def test_connected_products_stay_in_one_split():
    """(A,B) and (B,C) share B, so all three must land together."""
    df = pd.concat([
        _pair_frame(60),
        pd.DataFrame({"text_a": ["shared x"], "text_b": ["shared y"], "label": [0]}),
        pd.DataFrame({"text_a": ["shared y"], "text_b": ["shared z"], "label": [1]}),
    ], ignore_index=True)

    train, val, test = preprocessing.split_data(df)
    homes = [name for name, part in (("train", train), ("val", val), ("test", test))
             if "shared y" in _entities(part)]
    assert len(homes) == 1, f"chained component split across {homes}"


def test_split_ratios_stay_near_target():
    df = _pair_frame(600)
    train, val, test = preprocessing.split_data(df)
    assert len(train) + len(val) + len(test) == len(df)
    # Whole components move together, so exact ratios are impossible; a wide
    # band still catches the 63/23/14 drift a group-count split produced.
    assert 0.60 <= len(train) / len(df) <= 0.80
    assert 0.08 <= len(test) / len(df) <= 0.25


def test_dedup_is_order_invariant():
    """(A,B) and (B,A) with the same label are one pair, not two."""
    original = config.NUM_LABELS
    config.NUM_LABELS = 5
    try:
        df = pd.DataFrame({
            "product1_id": ["1", "2"], "product2_id": ["2", "1"],
            "product1_title": ["Alpha Buds", "Beta Case"],
            "product1_brand": ["Alpha", "Beta"],
            "product1_description": ["earbuds", "case"],
            "product2_title": ["Beta Case", "Alpha Buds"],
            "product2_brand": ["Beta", "Alpha"],
            "product2_description": ["case", "earbuds"],
            "relationship_label": ["UNRELATED", "UNRELATED"],
        })
        cleaned = preprocessing.clean_dataframe(df)
        assert len(cleaned) == 1, f"mirrored pair not deduplicated: {len(cleaned)} rows"
    finally:
        config.NUM_LABELS = original


def test_entity_split_is_deterministic_for_a_seed():
    a = preprocessing.split_data(_pair_frame(120), seed=123)[2]
    b = preprocessing.split_data(_pair_frame(120), seed=123)[2]
    assert list(a["text_a"]) == list(b["text_a"])


def test_falls_back_when_too_few_groups():
    """Tiny frames cannot be split group-wise; it must degrade, not crash."""
    df = _pair_frame(4)
    train, val, test = preprocessing.split_data(df)
    assert len(train) + len(val) + len(test) == len(df)
