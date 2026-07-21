"""
merge_datasets.py
==================
Combines multiple datasets that already use the structured schema
(product1_id, product1_title, product1_brand, product1_description,
 product2_id, product2_title, product2_brand, product2_description, label)
into a single deduplicated training file.

This is how you combine real-world data (e.g. products_amazon_google.csv)
with synthetic data (e.g. products_structured.csv) so the model gets both
real-world phrasing diversity AND deliberately-constructed hard negatives
(same brand, different product) that real data rarely provides on its own.

Run:
    python merge_datasets.py \\
        --inputs data/products_amazon_google.csv data/products_structured.csv \\
        --out data/products_combined.csv
"""

import argparse
import random

import pandas as pd

random.seed(42)

REQUIRED_COLUMNS = [
    "product1_id", "product1_title", "product1_brand", "product1_description",
    "product2_id", "product2_title", "product2_brand", "product2_description",
    "label",
]


def _dedupe_key(row) -> tuple:
    return (str(row["product1_title"]).strip().lower(), str(row["product2_title"]).strip().lower())


def merge(input_paths: list, out_path: str):
    frames = []
    for path in input_paths:
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"{path} is missing columns {missing} -- it doesn't match the structured schema. "
                f"Regenerate it with the structured/full schema before merging."
            )
        df["__source"] = path
        frames.append(df[REQUIRED_COLUMNS + ["__source"]])
        print(f"Loaded {len(df)} rows from {path} "
              f"(positives={int((df['label'] == 1).sum())}, negatives={int((df['label'] == 0).sum())})")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["product1_title", "product2_title", "label"])

    before = len(combined)
    seen = set()
    keep_mask = []
    for _, row in combined.iterrows():
        k = _dedupe_key(row)
        k_rev = (k[1], k[0])
        if k in seen or k_rev in seen or k[0] == k[1]:
            keep_mask.append(False)
            continue
        seen.add(k)
        keep_mask.append(True)
    combined = combined[keep_mask].reset_index(drop=True)
    print(f"Deduplicated: {before} -> {len(combined)} rows "
          f"(removed {before - len(combined)} duplicates/near-duplicates across sources)")

    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)
    combined[REQUIRED_COLUMNS].to_csv(out_path, index=False)

    n_pos = int((combined["label"] == 1).sum())
    n_neg = len(combined) - n_pos
    print(f"Wrote {len(combined)} rows to {out_path} (positives={n_pos}, negatives={n_neg})")

    print("\nSource breakdown in final merged file:")
    print(combined["__source"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True,
                         help="Two or more structured-schema CSV files to merge")
    parser.add_argument("--out", default="data/products_combined.csv")
    args = parser.parse_args()

    if len(args.inputs) < 2:
        raise ValueError("Provide at least 2 input files to merge.")

    merge(args.inputs, args.out)


if __name__ == "__main__":
    main()