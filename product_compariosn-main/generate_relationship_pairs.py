"""
generate_relationship_pairs.py
================================
Phase 1 / Pass 1 of the migration plan: turns your existing binary
(product1/product2/label) pair data into the 5-class relationship dataset
(EXACT_MATCH / SAME_PRODUCT_DIFFERENT_VARIANT / SIMILAR_ALTERNATIVE /
WEAKLY_SIMILAR / UNRELATED) using rule-derived signals -- no manual
labeling required for this pass.

This is a STARTING POINT, not a final labeler: it re-uses your existing
label==1/0 pairs and refines them into 5 classes using lightweight
attribute extraction + a keyword-based category heuristic. Both of those
are placeholders for the real `attribute_extraction/` and `classification/`
modules described in the architecture -- swap them in later without
changing this script's structure.

Any pair where either product is flagged HIGH severity by
data_quality.validator is dropped before labeling (Phase 0 gate).

Usage:
    python generate_relationship_pairs.py \
        --input data/products_combined.csv \
        --out data/relationship_pairs.csv
"""

import argparse
import re
from typing import Dict, Optional

import pandas as pd

from data_quality.contradiction_rules import TEXT_ONLY_RULES

# --------------------------------------------------------------------------
# Lightweight attribute extraction (placeholder for attribute_extraction/)
# --------------------------------------------------------------------------
COLOR_WORDS = [
    "black", "white", "silver", "gold", "blue", "red", "green", "grey",
    "gray", "titanium", "rose gold", "purple", "yellow", "pink", "bronze",
]
RAM_RE = re.compile(r"\b(\d{1,3})\s*gb\s*ram\b", re.I)
STORAGE_RE = re.compile(r"\b(\d{1,4})\s*(gb|tb)\b(?!\s*ram)", re.I)
BLUETOOTH_RE = re.compile(r"\bbluetooth\s*v?(\d\.\d)\b", re.I)


def extract_attributes(text: str) -> Dict[str, Optional[str]]:
    text_l = text.lower()
    attrs: Dict[str, Optional[str]] = {}

    color_match = next((c for c in COLOR_WORDS if c in text_l), None)
    attrs["color"] = color_match

    ram_match = RAM_RE.search(text_l)
    attrs["ram"] = ram_match.group(1) if ram_match else None

    storage_match = STORAGE_RE.search(text_l)
    attrs["storage"] = f"{storage_match.group(1)}{storage_match.group(2)}" if storage_match else None

    bt_match = BLUETOOTH_RE.search(text_l)
    attrs["bluetooth_version"] = bt_match.group(1) if bt_match else None

    return attrs


# --------------------------------------------------------------------------
# Lightweight category heuristic (placeholder for classification/)
# --------------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    "LAPTOP": ["laptop", "notebook", "macbook", "ultrabook"],
    "SMARTPHONE": ["smartphone", "iphone", "galaxy s", "pixel phone"],
    "TWS_EARBUDS": ["earbuds", "airdopes", "tws", "buds"],
    "CAMERA": ["camera", "dslr", "mirrorless", "lens"],
    "FOOTWEAR": ["shoe", "sneaker", "boot", "sandal"],
    "SOFTWARE": ["software", "license", "cd-rom", "diskette"],
    "NETWORKING": ["router", "modem", "switch", "access point"],
}


def categorize(text: str) -> str:
    text_l = text.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in text_l for kw in keywords):
            return category
    return "OTHER"


def is_high_severity(text: str) -> bool:
    for rule in TEXT_ONLY_RULES:
        issue = rule(text)
        if issue is not None and issue.severity == "high":
            return True
    return False


def token_overlap_ratio(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"[a-z0-9]+", a.lower()))
    tokens_b = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def label_pair(row: pd.Series) -> str:
    text_a = f"{row['product1_title']} {row.get('product1_description', '')}"
    text_b = f"{row['product2_title']} {row.get('product2_description', '')}"
    attrs_a = extract_attributes(text_a)
    attrs_b = extract_attributes(text_b)
    cat_a = categorize(text_a)
    cat_b = categorize(text_b)

    if row["label"] == 1:
        # Same product per the original binary label -- decide exact vs. variant
        differing_attrs = [
            k for k in attrs_a
            if attrs_a[k] is not None and attrs_b[k] is not None and attrs_a[k] != attrs_b[k]
        ]
        if differing_attrs:
            return "SAME_PRODUCT_DIFFERENT_VARIANT"
        return "EXACT_MATCH"

    # label == 0: different products -- decide alternative vs. weak vs. unrelated
    if cat_a != cat_b or cat_a == "OTHER":
        return "UNRELATED"

    overlap = token_overlap_ratio(text_a, text_b)
    if overlap >= 0.25:
        return "SIMILAR_ALTERNATIVE"
    return "WEAKLY_SIMILAR"


def generate(input_path: str, output_path: str, sample: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)

    # Phase 0 gate: drop pairs where either side has a HIGH severity issue
    keep_mask = []
    for _, row in df.iterrows():
        text_a = f"{row['product1_title']} {row.get('product1_description', '')}"
        text_b = f"{row['product2_title']} {row.get('product2_description', '')}"
        keep_mask.append(not (is_high_severity(text_a) or is_high_severity(text_b)))
    dropped = len(df) - sum(keep_mask)
    df = df[keep_mask].reset_index(drop=True)

    df["relationship_label"] = df.apply(label_pair, axis=1)
    df["source"] = "derived"

    out_cols = [
        "product1_id", "product1_title", "product1_brand", "product1_description",
        "product2_id", "product2_title", "product2_brand", "product2_description",
        "relationship_label", "source",
    ]
    result = df[out_cols]
    result.to_csv(output_path, index=False)

    print(f"Input pairs: {len(df) + dropped} | dropped (high-severity data quality): {dropped} "
          f"| written: {len(result)}")
    print(result["relationship_label"].value_counts())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the 5-class relationship_pairs.csv")
    parser.add_argument("--input", required=True, help="Existing binary pair CSV, e.g. data/products_combined.csv")
    parser.add_argument("--out", default="data/relationship_pairs.csv")
    parser.add_argument("--sample", type=int, default=None,
                         help="Optional: only process N random rows (for a quick sanity check)")
    args = parser.parse_args()

    generate(args.input, args.out, sample=args.sample)