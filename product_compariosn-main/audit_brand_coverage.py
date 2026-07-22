"""
audit_brand_coverage.py
=========================
Checks how well a list of brands/product-lines you actually care about
(your real target catalog) is represented in your source product data --
run this BEFORE sourcing new data or retraining, so you know exactly
which brands are missing or thin, instead of finding out one bad
inference result at a time (like GOBOULT).

Usage:
    python audit_brand_coverage.py --input data/products_combined.csv \
        --brands GOBOULT boAt Noise pTron Boult Zebronics Realme JBL Sony

    # or from a file, one brand per line:
    python audit_brand_coverage.py --input data/products_combined.csv \
        --brands-file my_target_brands.txt
"""

import argparse

import pandas as pd


def collect_text_columns(df: pd.DataFrame) -> pd.Series:
    """Works with either the pair format (product1_title, product2_title, ...)
    or a plain single-product format (title, description, brand)."""
    text_cols = [c for c in df.columns if c.endswith("_title") or c.endswith("_description") or c.endswith("_brand")]
    if not text_cols:
        text_cols = [c for c in ["title", "description", "brand"] if c in df.columns]
    if not text_cols:
        raise ValueError("Could not find title/description/brand-like columns in this file.")
    combined = pd.concat([df[c].astype(str) for c in text_cols], ignore_index=True).str.lower()
    return combined


def audit(input_path: str, brands: list, thin_threshold: int = 10) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    text = collect_text_columns(df)

    rows = []
    for brand in brands:
        keyword = brand.strip().lower()
        count = int(text.str.contains(keyword, regex=False).sum())
        if count == 0:
            status = "MISSING"
        elif count < thin_threshold:
            status = "THIN"
        else:
            status = "OK"
        rows.append({"brand": brand, "mentions": count, "status": status})

    status_order = {"MISSING": 0, "THIN": 1, "OK": 2}
    report = pd.DataFrame(rows)
    report["_sort"] = report["status"].map(status_order)
    report = report.sort_values(["_sort", "mentions"]).drop(columns="_sort").reset_index(drop=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit brand/product-line coverage in source data")
    parser.add_argument("--input", required=True, help="e.g. data/products_combined.csv")
    parser.add_argument("--brands", nargs="*", default=None, help="Space-separated brand names")
    parser.add_argument("--brands-file", default=None, help="Path to a text file, one brand per line")
    parser.add_argument("--thin-threshold", type=int, default=10,
                         help="Below this many mentions (but above 0), flag as THIN")
    args = parser.parse_args()

    brands = list(args.brands) if args.brands else []
    if args.brands_file:
        with open(args.brands_file) as f:
            brands += [line.strip() for line in f if line.strip()]

    if not brands:
        raise SystemExit("Provide --brands (space-separated) or --brands-file (one per line).")

    report = audit(args.input, brands, thin_threshold=args.thin_threshold)
    print(report.to_string(index=False))

    missing = report[report["status"] == "MISSING"]["brand"].tolist()
    thin = report[report["status"] == "THIN"]["brand"].tolist()
    print(f"\nMISSING entirely ({len(missing)}): {missing}")
    print(f"THIN coverage, <{args.thin_threshold} mentions ({len(thin)}): {thin}")