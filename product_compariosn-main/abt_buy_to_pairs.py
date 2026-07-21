"""
abt_buy_to_pairs.py
====================
Converts the Abt-Buy entity-resolution benchmark (from the same Leipzig
benchmark suite as Amazon-Google Products) into structured training pairs
-- a second REAL e-commerce data source, adding diversity beyond a single
catalog pairing.

Expected input files (from the Leipzig benchmark "Abt-Buy" download, or a
Kaggle mirror of the same dataset):
    Abt.csv                     columns: id, name, description, price
    Buy.csv                     columns: id, name, description, manufacturer, price
    abt_buy_perfectMapping.csv  columns: idAbt, idBuy

Run:
    python abt_buy_to_pairs.py \\
        --abt Abt.csv \\
        --buy Buy.csv \\
        --mapping abt_buy_perfectMapping.csv \\
        --out data/products_abt_buy.csv \\
        --neg_ratio 1.5
"""

import argparse
import csv
import random

import pandas as pd

random.seed(43)


def _clean(text) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.split()).strip()


def load_data(abt_path: str, buy_path: str, mapping_path: str):
    abt = pd.read_csv(abt_path, encoding="latin-1")
    buy = pd.read_csv(buy_path, encoding="latin-1")
    mapping = pd.read_csv(mapping_path, encoding="latin-1")

    for col in ("name", "description"):
        if col in abt.columns:
            abt[col] = abt[col].apply(_clean)
    for col in ("name", "description", "manufacturer"):
        if col in buy.columns:
            buy[col] = buy[col].apply(_clean)

    abt = abt.set_index("id")
    buy = buy.set_index("id")
    return abt, buy, mapping


def _abt_fields(abt, a_id):
    row = abt.loc[a_id]
    title = row.get("name", "")
    # Abt.csv has no manufacturer column -- brand isn't reliably available,
    # leave blank rather than guessing (build_product_text handles empty
    # brand gracefully).
    brand = ""
    desc = row.get("description", "") or ""
    return str(a_id), title, brand, desc


def _buy_fields(buy, b_id):
    row = buy.loc[b_id]
    title = row.get("name", "")
    brand = row.get("manufacturer", "") or ""
    desc = row.get("description", "") or ""
    return str(b_id), title, brand, desc


def generate_pairs(abt: pd.DataFrame, buy: pd.DataFrame, mapping: pd.DataFrame, neg_ratio: float):
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
        a_id, b_id = m["idAbt"], m["idBuy"]
        if a_id not in abt.index or b_id not in buy.index:
            continue
        if _add_unique(_abt_fields(abt, a_id), _buy_fields(buy, b_id), 1):
            n_pos += 1

    # ---- Negatives: random non-matching Abt x Buy pairs ----
    n_neg_target = int(n_pos * neg_ratio)
    abt_ids = abt.index.tolist()
    buy_ids = buy.index.tolist()
    mapping_pairs = set(zip(mapping["idAbt"], mapping["idBuy"]))

    added, attempts = 0, 0
    while added < n_neg_target and attempts < n_neg_target * 20:
        attempts += 1
        a_id = random.choice(abt_ids)
        b_id = random.choice(buy_ids)
        if (a_id, b_id) in mapping_pairs:
            continue
        if _add_unique(_abt_fields(abt, a_id), _buy_fields(buy, b_id), 0):
            added += 1

    random.shuffle(rows)
    return rows, n_pos, added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--abt", required=True)
    parser.add_argument("--buy", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--out", default="data/products_abt_buy.csv")
    parser.add_argument("--neg_ratio", type=float, default=1.5)
    args = parser.parse_args()

    abt, buy, mapping = load_data(args.abt, args.buy, args.mapping)
    print(f"Loaded {len(abt)} Abt products, {len(buy)} Buy products, "
          f"{len(mapping)} gold-standard matches")

    rows, n_pos, n_neg = generate_pairs(abt, buy, mapping, args.neg_ratio)

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