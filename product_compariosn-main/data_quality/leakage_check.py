"""
data_quality/leakage_check.py
==============================
Data-leakage analysis for the train/val/test split produced by
preprocessing.split_data.

preprocessing.split_data() does a stratified train_test_split at the ROW
(pair) level, after clean_dataframe() has already dropped exact-duplicate
(text_a, text_b, label) rows. That dedup only guards against the SAME
pair appearing twice -- it does nothing to stop the SAME product
appearing as one side of a pair in train and as one side of a DIFFERENT
pair in test. This script measures that directly instead of assuming
either way.

Checks:
  1. Exact pair leakage across splits (sanity check on the split itself --
     should be 0; a nonzero number means the dedup step itself is broken).
  2. Near-duplicate pair leakage: pairs in different splits whose
     text_a/text_b match after aggressive normalization.
  3. Entity-level leakage: how many individual products (by their cleaned
     text_a/text_b string) appear in both train and test.
  4. % of test ROWS where at least one side of the pair was already seen
     during training.

Read-only diagnostic -- does not modify preprocessing.py or the split
logic. If the numbers show severe leakage, the fix (splitting by
entity/product id before pairing, not by row after pairing) is a
follow-up change to preprocessing.split_data, not bundled here.

Run with:
    python -m data_quality.leakage_check
    python -m data_quality.leakage_check --data data/relationship_pairs_final.csv --num_labels 5
"""

import argparse
import re

import pandas as pd

import config
from exact_match import preprocessing
from utils import get_logger

logger = get_logger(__name__)

_NORM_RE = re.compile(r"[^a-z0-9]+")


def _norm(text: str) -> str:
    return _NORM_RE.sub("", str(text).lower())


def _entity_set(df: pd.DataFrame) -> set:
    return set(df["text_a"]).union(set(df["text_b"]))


def _pair_key_set(df: pd.DataFrame, normalized: bool = False) -> set:
    keys = set()
    for _, row in df.iterrows():
        a, b = row["text_a"], row["text_b"]
        if normalized:
            a, b = _norm(a), _norm(b)
        keys.add(tuple(sorted([a, b])))
    return keys


def check_leakage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    report = {}

    train_pairs = _pair_key_set(train_df)
    val_pairs = _pair_key_set(val_df)
    test_pairs = _pair_key_set(test_df)
    report["exact_pair_overlap_train_test"] = len(train_pairs & test_pairs)
    report["exact_pair_overlap_train_val"] = len(train_pairs & val_pairs)
    report["exact_pair_overlap_val_test"] = len(val_pairs & test_pairs)

    train_norm = _pair_key_set(train_df, normalized=True)
    test_norm = _pair_key_set(test_df, normalized=True)
    report["near_duplicate_pair_overlap_train_test"] = len(train_norm & test_norm) - report["exact_pair_overlap_train_test"]

    train_entities = _entity_set(train_df)
    val_entities = _entity_set(val_df)
    test_entities = _entity_set(test_df)
    shared_train_test = train_entities & test_entities
    shared_train_val = train_entities & val_entities
    report["shared_entities_train_test"] = len(shared_train_test)
    report["shared_entities_train_val"] = len(shared_train_val)
    report["test_entities_total"] = len(test_entities)
    report["pct_test_entities_also_in_train"] = round(
        100 * len(shared_train_test) / len(test_entities), 2
    ) if test_entities else 0.0

    seen_mask = test_df["text_a"].isin(train_entities) | test_df["text_b"].isin(train_entities)
    report["pct_test_rows_with_a_seen_entity"] = round(100 * seen_mask.mean(), 2)

    return report, shared_train_test


def main():
    parser = argparse.ArgumentParser(description="Check train/val/test splits for leakage.")
    parser.add_argument("--data", default=config.RAW_DATA_PATH)
    parser.add_argument("--num_labels", type=int, default=config.NUM_LABELS,
                         help="2 for the binary exact-match schema, 5 for the relationship schema. "
                              "Must be set here explicitly -- this runs as its own process.")
    parser.add_argument("--sample_shared", type=int, default=10,
                         help="How many example shared entities to print")
    args = parser.parse_args()
    config.NUM_LABELS = args.num_labels

    train_df, val_df, test_df = preprocessing.load_clean_split(args.data)
    report, shared_entities = check_leakage(train_df, val_df, test_df)

    logger.info("=== LEAKAGE REPORT ===")
    for k, v in report.items():
        logger.info(f"{k}: {v}")

    if shared_entities:
        sample = list(shared_entities)[: args.sample_shared]
        logger.info(f"\nExample products seen in BOTH train and test ({len(sample)} of {len(shared_entities)}):")
        for text in sample:
            logger.info(f"  - {text}")

    if report["exact_pair_overlap_train_test"] > 0:
        logger.warning(
            f"{report['exact_pair_overlap_train_test']} EXACT duplicate pairs leaked across "
            "train/test despite clean_dataframe's dedup step -- investigate "
            "clean_dataframe's drop_duplicates key before anything else."
        )

    if report["pct_test_rows_with_a_seen_entity"] > 20:
        logger.warning(
            f"{report['pct_test_rows_with_a_seen_entity']}% of test rows involve a product "
            "also seen during training. The current split is pair-level, not "
            "entity-level -- reported test accuracy may be inflated relative to "
            "truly unseen products."
        )


if __name__ == "__main__":
    main()