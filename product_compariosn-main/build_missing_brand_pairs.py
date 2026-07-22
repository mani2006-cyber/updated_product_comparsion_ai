"""
build_missing_brand_pairs.py
===============================
One-off script: generates data/products_missing_brands.csv in the same
schema as your other source files (product1_id, product1_title,
product1_brand, product1_description, product2_id, product2_title,
product2_brand, product2_description, label), using REAL products from
the brands audit_brand_coverage.py flagged as MISSING: GOBOULT (formerly
Boult Audio -- same company, so both names are included), pTron, and
Zebronics -- all TWS-earbuds-heavy, all absent from products_combined.csv.

Specs below are sourced from each brand's own product pages (battery
hours, Bluetooth version, feature names -- factual specs, not copied
marketing copy).

label=1 rows: two phrasing/color variants of the SAME product (refines
    into EXACT_MATCH / SAME_PRODUCT_DIFFERENT_VARIANT downstream).
label=0 rows: different products, same category -- both same-brand
    (hard negatives) and cross-brand (refines into SIMILAR_ALTERNATIVE /
    WEAKLY_SIMILAR downstream, since they're all TWS earbuds).

Run:
    python build_missing_brand_pairs.py --out data/products_missing_brands.csv
"""

import argparse
import itertools

import pandas as pd

# (id, brand, model_title, description, category_tag)
PRODUCTS = [
    ("MB001", "GOBOULT", "GOBOULT Z40 Pro TWS Earbuds", "TWS earbuds with 100 hours total playtime and quad-mic ENC", "earbuds"),
    ("MB002", "GOBOULT", "Boult AirBass K60 TWS Earbuds", "TWS earbuds with touch controls and extra ear tips included", "earbuds"),
    ("MB003", "GOBOULT", "Boult Z40 Earbuds", "True wireless earbuds with USB-C charging case", "earbuds"),
    ("MB004", "pTron", "pTron Bassbuds Astra TWS Earbuds", "TWS earbuds with 34 hours playtime, Bluetooth 5.3, and a companion app with custom EQ", "earbuds"),
    ("MB005", "pTron", "pTron Bassbuds Indie TWS Earbuds", "TWS earbuds with 28 hours total playtime and Bluetooth 5.0", "earbuds"),
    ("MB006", "pTron", "pTron Bassbuds Sports TWS Earbuds", "TWS earbuds with 32 hours total playback and a secure hook-style fit", "earbuds"),
    ("MB007", "pTron", "pTron Bassbuds Pixel TWS Earbuds", "TWS earbuds with Bluetooth 5.1, passive noise cancellation, and dual mic", "earbuds"),
    ("MB008", "pTron", "pTron Bassbuds Duo TWS Earbuds", "TWS earbuds with 8 hours single-charge playback and Bluetooth 5.1", "earbuds"),
    ("MB009", "Zebronics", "Zebronics Sound Bomb 1 TWS Earbuds", "TWS earbuds with 12 hours playback with case and splash-proof design", "earbuds"),
    ("MB010", "Zebronics", "Zebronics Sound Bomb Q Pro TWS Earbuds", "TWS earbuds with 35 hours playtime, Bluetooth 5.0, and IPX7 waterproofing", "earbuds"),
    ("MB011", "Zebronics", "Zebronics Sound Bomb 7 TWS Earbuds", "TWS earbuds with up to 60 hours playtime, Bluetooth 5.2, ENC, and a gaming mode", "earbuds"),
    ("MB012", "Zebronics", "Zebronics Sound Bomb 4 TWS Earbuds", "TWS earbuds with 20 hours playtime and 13mm drivers", "earbuds"),
]

# (product_id, alternate title phrasing/color, alternate description) -- same product, label=1
VARIANTS = {
    "MB001": ("Non-Touch GOBOULT Z40 Pro Earbuds Black", "GOBOULT Z40 Pro: quad-mic ENC, 100 hour total playtime, TWS earbuds"),
    "MB004": ("Bassbuds Astra by pTron, Black", "pTron Bassbuds Astra: Bluetooth 5.3, 34 hour playtime, custom EQ app"),
    "MB009": ("Zeb-Sound Bomb 1 Earbuds White", "Sound Bomb 1 by Zebronics: splash-proof, 12 hour playback with case"),
}


def build(out_path: str):
    rows = []

    # ---- Positive pairs: same product, different phrasing (label=1) ----
    for product_id, (alt_title, alt_description) in VARIANTS.items():
        base = next(p for p in PRODUCTS if p[0] == product_id)
        rows.append({
            "product1_id": base[0], "product1_title": base[2], "product1_brand": base[1], "product1_description": base[3],
            "product2_id": f"{base[0]}-alt", "product2_title": alt_title, "product2_brand": base[1], "product2_description": alt_description,
            "label": 1,
        })

    # ---- Negative pairs: different products, same category (label=0) ----
    # Same-brand hard negatives + cross-brand alternatives -- both are
    # valuable: same-brand teaches "same brand != same product", cross-brand
    # gives the SIMILAR_ALTERNATIVE refinement step real examples for
    # exactly the brands that were missing.
    for (id_a, brand_a, title_a, desc_a, _), (id_b, brand_b, title_b, desc_b, _) in itertools.combinations(PRODUCTS, 2):
        rows.append({
            "product1_id": id_a, "product1_title": title_a, "product1_brand": brand_a, "product1_description": desc_a,
            "product2_id": id_b, "product2_title": title_b, "product2_brand": brand_b, "product2_description": desc_b,
            "label": 0,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path} "
          f"(positives={int((df['label'] == 1).sum())}, negatives={int((df['label'] == 0).sum())})")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build real GOBOULT/pTron/Zebronics pairs to close the coverage gap")
    parser.add_argument("--out", default="data/products_missing_brands.csv")
    args = parser.parse_args()
    build(args.out)