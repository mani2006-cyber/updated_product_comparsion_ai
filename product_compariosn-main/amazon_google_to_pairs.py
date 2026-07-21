"""
amazon_google_to_pairs.py
==========================
Converts the standard Amazon-Google Products entity-resolution benchmark
into training pairs for this project, using the STRUCTURED schema
(id, title, brand, description per side) -- matching config.TEXT_COLUMNS_FULL.
Every positive pair here is a REAL, human-verified match (from the
dataset's "perfect mapping" gold standard) between two independently
written product listings, with manufacturer used as the brand field.

Expected input files (from the Leipzig benchmark, or a Kaggle mirror):
    Amazon.csv                          columns: id, title, description, manufacturer, price
    GoogleProducts.csv                  columns: id, name, description, manufacturer, price
    Amzon_GoogleProducts_perfectMapping.csv   columns: idAmazon, idGoogleBase

Run:
    python amazon_google_to_pairs.py \\
        --amazon Amazon.csv \\
        --google GoogleProducts.csv \\
        --mapping Amzon_GoogleProducts_perfectMapping.csv \\
        --out data/products_amazon_google.csv \\
        --neg_ratio 1.5
"""

import argparse
import csv
import random

import pandas as pd

random.seed(42)


def _clean(text) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def load_data(amazon_path: str, google_path: str, mapping_path: str):
    amazon = pd.read_csv(amazon_path, encoding="latin-1")
    google = pd.read_csv(google_path, encoding="latin-1")
    mapping = pd.read_csv(mapping_path, encoding="latin-1")

    for col in ("title", "description", "manufacturer"):
        if col in amazon.columns:
            amazon[col] = amazon[col].apply(_clean)
    for col in ("name", "description", "manufacturer"):
        if col in google.columns:
            google[col] = google[col].apply(_clean)

    amazon = amazon.set_index("id")
    google = google.set_index("id")
    return amazon, google, mapping


def _amazon_fields(amazon, a_id):
    row = amazon.loc[a_id]
    title = row.get("title", "")
    brand = row.get("manufacturer", "") or ""
    desc = row.get("description", "") or ""
    return str(a_id), title, brand, desc


def _google_fields(google, g_id):
    row = google.loc[g_id]
    title = row.get("name", "")
    brand = row.get("manufacturer", "") or ""
    desc = row.get("description", "") or ""
    return str(g_id), title, brand, desc


def generate_pairs(amazon: pd.DataFrame, google: pd.DataFrame, mapping: pd.DataFrame, neg_ratio: float):
    rows = []
    seen = set()

    def _add_unique(side_a, side_b, label):
        title_a, title_b = side_a[1], side_b[1]
        if not title_a or not title_b or title_a.lower() == title_b.lower():
            return False
        key, key_rev = (title_a.lower(), title_b.lower()), (title_b.lower(), title_a.lower())
        if key in seen or key_rev in seen:
            return False
        seen.add(key)
        rows.append((*side_a, *side_b, label))
        return True

    # ---- Positives: real gold-standard matches ----
    n_pos = 0
    for _, m in mapping.sample(frac=1, random_state=1).iterrows():
        a_id, g_id = m["idAmazon"], m["idGoogleBase"]
        if a_id not in amazon.index or g_id not in google.index:
            continue
        if _add_unique(_amazon_fields(amazon, a_id), _google_fields(google, g_id), 1):
            n_pos += 1

    # ---- Negatives: random non-matching Amazon x Google pairs ----
    n_neg_target = int(n_pos * neg_ratio)
    amazon_ids = amazon.index.tolist()
    google_ids = google.index.tolist()
    mapping_pairs = set(zip(mapping["idAmazon"], mapping["idGoogleBase"]))

    added, attempts = 0, 0
    while added < n_neg_target and attempts < n_neg_target * 20:
        attempts += 1
        a_id = random.choice(amazon_ids)
        g_id = random.choice(google_ids)
        if (a_id, g_id) in mapping_pairs:
            continue
        if _add_unique(_amazon_fields(amazon, a_id), _google_fields(google, g_id), 0):
            added += 1

    random.shuffle(rows)
    return rows, n_pos, added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--amazon", required=True)
    parser.add_argument("--google", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--out", default="data/products_amazon_google.csv")
    parser.add_argument("--neg_ratio", type=float, default=1.5,
                         help="negatives generated per positive (dataset has far more possible "
                              "negatives than the ~1300 gold positives)")
    args = parser.parse_args()

    amazon, google, mapping = load_data(args.amazon, args.google, args.mapping)
    print(f"Loaded {len(amazon)} Amazon products, {len(google)} Google products, "
          f"{len(mapping)} gold-standard matches")

    rows, n_pos, n_neg = generate_pairs(amazon, google, mapping, args.neg_ratio)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "product1_id", "product1_title", "product1_brand", "product1_description",
            "product2_id", "product2_title", "product2_brand", "product2_description",
            "label",
        ])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out} (positives={n_pos}, negatives={n_neg})")


if __name__ == "__main__":
    main()