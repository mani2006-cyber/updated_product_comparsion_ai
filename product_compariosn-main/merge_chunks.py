"""
merge_chunks.py
===============
Merges all chunked CSV files into one final dataset.
Also deduplicates and shuffles.

Usage:
    python merge_chunks.py --input-dir ./data_chunks --output data/products_combined.csv

Optional - create train/val/test split directly:
    python merge_chunks.py --input-dir ./data_chunks --output data/products_combined.csv --split
"""

import argparse
import csv
import os
import random

import pandas as pd
from sklearn.model_selection import train_test_split

random.seed(42)

HEADER = [
    "product1_id", "product1_title", "product1_brand", "product1_description",
    "product2_id", "product2_title", "product2_brand", "product2_description",
    "label",
]


def find_chunk_files(input_dir: str):
    files = sorted([f for f in os.listdir(input_dir) if f.startswith("products_chunk_") and f.endswith(".csv")])
    return [os.path.join(input_dir, f) for f in files]


def merge_chunks(input_dir: str, output_path: str, dedupe: bool = True, shuffle: bool = True):
    chunk_files = find_chunk_files(input_dir)
    if not chunk_files:
        raise ValueError(f"No chunk files found in {input_dir}")

    print(f"Found {len(chunk_files)} chunk files")

    # Read all
    all_rows = []
    for path in chunk_files:
        df = pd.read_csv(path)
        all_rows.append(df)
        print(f"  Loaded {len(df)} rows from {os.path.basename(path)}")

    combined = pd.concat(all_rows, ignore_index=True)
    print(f"\nCombined: {len(combined)} rows")

    # Validate columns
    missing = [c for c in HEADER if c not in combined.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Drop rows with empty titles
    combined = combined.dropna(subset=["product1_title", "product2_title", "label"])

    # Deduplicate
    if dedupe:
        before = len(combined)
        combined["_key"] = combined.apply(
            lambda r: tuple(sorted([str(r["product1_title"]).strip().lower(), 
                                     str(r["product2_title"]).strip().lower()])), axis=1
        )
        combined = combined.drop_duplicates(subset=["_key"]).drop(columns=["_key"]).reset_index(drop=True)
        print(f"Deduplicated: {before} -> {len(combined)} rows")

    # Shuffle
    if shuffle:
        combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    combined[HEADER].to_csv(output_path, index=False)

    n_pos = int((combined["label"] == 1).sum())
    n_neg = len(combined) - n_pos
    print(f"\nSaved to {output_path}")
    print(f"Final: {len(combined)} rows | Positives: {n_pos} | Negatives: {n_neg}")
    print(f"Label distribution: {dict(combined['label'].value_counts().sort_index())}")

    return combined


def split_and_save(df: pd.DataFrame, output_dir: str, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001

    stratify = df["label"] if len(df) > 100 else None
    train_val, test = train_test_split(df, test_size=test_ratio, random_state=seed, stratify=stratify)

    stratify2 = train_val["label"] if len(train_val) > 100 else None
    relative_val = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(train_val, test_size=relative_val, random_state=seed, stratify=stratify2)

    os.makedirs(output_dir, exist_ok=True)
    train[HEADER].to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val[HEADER].to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test[HEADER].to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"\nSplits saved to {output_dir}:")
    print(f"  train.csv: {len(train)} rows")
    print(f"  val.csv:   {len(val)} rows")
    print(f"  test.csv:  {len(test)} rows")


def main():
    parser = argparse.ArgumentParser(description="Merge chunked dataset files")
    parser.add_argument("--input-dir", required=True, help="Directory with chunk files")
    parser.add_argument("--output", default="data/products_combined.csv", help="Merged output file")
    parser.add_argument("--no-dedupe", action="store_true", help="Skip deduplication")
    parser.add_argument("--no-shuffle", action="store_true", help="Skip shuffling")
    parser.add_argument("--split", action="store_true", help="Also create train/val/test splits")
    parser.add_argument("--split-dir", default="data/splits", help="Directory for splits")
    args = parser.parse_args()

    df = merge_chunks(args.input_dir, args.output, dedupe=not args.no_dedupe, shuffle=not args.no_shuffle)

    if args.split:
        split_and_save(df, args.split_dir)


if __name__ == "__main__":
    main()
