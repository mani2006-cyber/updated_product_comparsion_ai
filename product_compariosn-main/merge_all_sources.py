"""
merge_all_sources.py
=======================
Merges all relationship_pairs.csv sources into one final training file:
  - your audio-specific real data (relationship_pairs_with_nearmiss_v2.csv)
  - your general-category real data (catalog_relationship_pairs.csv)
  - the friend's synthetic data, AFTER running it through
    generate_relationship_pairs.py to get 5-class labels
    (synthetic_relationship_pairs.csv)

Each input file must already be in the 5-class relationship_pairs schema
(product1_*, product2_*, relationship_label, source). Tags each row with
which file it came from (in a new 'origin' column) so you can inspect or
filter by source later without losing that information.

Usage:
    python merge_all_sources.py \
        --inputs data/relationship_pairs_with_nearmiss_v2.csv data/catalog_relationship_pairs.csv data/synthetic_relationship_pairs.csv \
        --out data/relationship_pairs_final.csv
"""

import argparse
import os

import pandas as pd

REQUIRED_COLUMNS = [
    "product1_id", "product1_title", "product1_brand", "product1_description",
    "product2_id", "product2_title", "product2_brand", "product2_description",
    "relationship_label",
]


def merge_all_sources(input_paths: list, out_path: str) -> pd.DataFrame:
    frames = []
    for path in input_paths:
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}. "
                              f"Run it through generate_relationship_pairs.py first if it's still binary-labeled.")
        df = df[REQUIRED_COLUMNS + (["source"] if "source" in df.columns else [])].copy()
        df["origin"] = os.path.basename(path)
        frames.append(df)
        print(f"Loaded {len(df)} rows from {path}")

    combined = pd.concat(frames, ignore_index=True)

    before = len(combined)
    combined = combined.drop_duplicates(subset=["product1_title", "product2_title", "relationship_label"])
    print(f"Deduplicated: {before} -> {len(combined)} rows")

    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    combined.to_csv(out_path, index=False)

    print(f"\nWrote {len(combined)} rows to {out_path}")
    print("\nBy relationship_label:")
    print(combined["relationship_label"].value_counts())
    print("\nBy origin file:")
    print(combined["origin"].value_counts())
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all relationship-pair sources into one final training file")
    parser.add_argument("--inputs", nargs="+", required=True, help="Space-separated list of relationship_pairs CSVs")
    parser.add_argument("--out", default="data/relationship_pairs_final.csv")
    args = parser.parse_args()

    merge_all_sources(args.inputs, args.out)