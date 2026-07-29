"""
build_balanced_catalog.py
============================
Caps oversized categories (Books, Music, Movies & TV, etc.) down to a
target ceiling, while keeping smaller categories (Electronics, Shoes,
Clothing, etc.) fully intact -- so downstream category-classifier
training and pair generation aren't dominated by whichever category
happened to have the most raw listings.

Usage:
    python build_balanced_catalog.py \
        --input full_joined_catalog.csv \
        --out balanced_catalog.csv \
        --max-per-category 75000
"""

import argparse

import pandas as pd


def build_balanced_catalog(input_path: str, out_path: str, max_per_category: int, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    before_counts = df["category"].value_counts()

    sampled_frames = []
    for category, group in df.groupby("category"):
        if len(group) > max_per_category:
            sampled_frames.append(group.sample(n=max_per_category, random_state=seed))
        else:
            sampled_frames.append(group)

    balanced = pd.concat(sampled_frames, ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle
    balanced.to_csv(out_path, index=False)

    after_counts = balanced["category"].value_counts()
    print(f"Before: {len(df)} rows across {len(before_counts)} categories")
    print(before_counts.head(15))
    print()
    print(f"After:  {len(balanced)} rows across {len(after_counts)} categories (cap={max_per_category})")
    print(after_counts.head(15))
    print(f"\nWrote {out_path}")
    return balanced


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cap oversized categories to balance the catalog")
    parser.add_argument("--input", required=True, help="e.g. full_joined_catalog.csv")
    parser.add_argument("--out", default="balanced_catalog.csv")
    parser.add_argument("--max-per-category", type=int, default=75000,
                         help="Categories larger than this are randomly downsampled to this size; "
                              "smaller categories are kept fully intact")
    args = parser.parse_args()

    build_balanced_catalog(args.input, args.out, args.max_per_category)