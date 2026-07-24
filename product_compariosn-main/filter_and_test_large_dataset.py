"""
filter_and_test_large_dataset.py
===================================
Filters the large (categories.txt / titles.txt / brands.txt /
description.txt) dataset down to your relevant categories (electronics /
audio / wearables etc.), then optionally runs a sample of the filtered
products through the trained ProductComparer as a real-world spot-check.

IMPORTANT: this was written without seeing your actual files (only a
pasted sample earlier), so the parsing assumptions below may not match
exactly. ALWAYS run --preview first on Kaggle to confirm parsing looks
right before processing the full ~7M lines.

Assumed format (confirm with --preview):
  titles.txt / brands.txt / description.txt : "<product_id>\t<value>" one per line
  categories.txt : "<product_id>\t<category_path_1>\t<category_path_2>\t..."
                   (multiple tab-separated category hierarchy strings per id,
                   each hierarchy comma-separated, e.g. "Electronics, Audio, Headphones")

Usage (on Kaggle, paths under /kaggle/input/<your-dataset-name>/):
    # Step 1 -- ALWAYS do this first
    python filter_and_test_large_dataset.py \
        --categories /kaggle/input/xxx/categories.txt \
        --titles /kaggle/input/xxx/titles.txt \
        --brands /kaggle/input/xxx/brands.txt \
        --descriptions /kaggle/input/xxx/description.txt \
        --preview 10

    # Step 2 -- once preview looks right, filter for real
    python filter_and_test_large_dataset.py \
        --categories /kaggle/input/xxx/categories.txt \
        --titles /kaggle/input/xxx/titles.txt \
        --brands /kaggle/input/xxx/brands.txt \
        --descriptions /kaggle/input/xxx/description.txt \
        --keywords electronics audio earbuds headphone wireless speaker \
        --out data/large_dataset_filtered.csv

    # Step 3 -- spot-check the filtered products against the trained model
    python filter_and_test_large_dataset.py --test-only \
        --filtered data/large_dataset_filtered.csv --sample-size 20 \
        --model-dir trained_model
"""

import argparse
import random

import pandas as pd


def _parse_id_value_file(path: str, max_lines: int = None) -> dict:
    """Parses '<id>\t<value>' lines into {id: value}. Falls back to
    splitting on the first run of whitespace if there's no literal tab,
    in case the real file uses spaces instead."""
    result = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
            else:
                parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            product_id, value = parts
            result[product_id.strip()] = value.strip()
    return result


def _parse_description_file(path: str, max_lines: int = None) -> dict:
    """Parses the REAL description.txt block format (confirmed against the
    actual uploaded file):

        product/productId: B0027DQHA0
        product/description: Conducted by John Neschling since 1997, ...
        <blank line>
        product/productId: 0756400120
        product/description: ...

    NOT a simple 'id value' file like titles.txt/brands.txt -- each record
    is 3 lines (id, description, blank separator), CRLF line endings."""
    result = {}
    current_id = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            line = line.rstrip("\r\n")
            if line.startswith("product/productId:"):
                current_id = line.split(":", 1)[1].strip()
            elif line.startswith("product/description:"):
                if current_id is not None:
                    result[current_id] = line.split(":", 1)[1].strip()
    return result


def _parse_categories_file(path: str, max_lines: int = None) -> dict:
    """Parses the REAL categories.txt block format (confirmed against the
    actual uploaded file -- NOT the tab-separated single-line format
    originally assumed):

        B0027DQHA0
          Movies & TV, TV
          Music, Classical
        0756400120
          Books, Literature & Fiction, Anthologies & Literary Collections, General
          ...

    An unindented line is a product id; the indented lines that follow it
    (until the next unindented line) are its category paths."""
    result = {}
    current_id = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            if raw[0] in (" ", "\t"):
                if current_id is not None:
                    result.setdefault(current_id, []).append(raw.strip())
            else:
                current_id = raw.strip()
                result.setdefault(current_id, [])
    return result


def preview(args):
    print("=" * 70)
    print("PREVIEW MODE -- confirming parsing before processing the full files")
    print("=" * 70)
    for label, path, parser in [
        ("titles", args.titles, _parse_id_value_file),
        ("brands", args.brands, _parse_id_value_file),
        ("descriptions", args.descriptions, _parse_description_file),
        ("categories", args.categories, _parse_categories_file),
    ]:
        if not path:
            continue
        parsed = parser(path, max_lines=args.preview)
        print(f"\n--- {label} ({path}) -- parsed {len(parsed)} of first {args.preview} lines ---")
        for pid, val in list(parsed.items())[:5]:
            print(f"  {pid!r} -> {str(val)[:120]}")
    print("\nIf these look wrong (e.g. id/value not split correctly, or "
          "category paths not separated), the file's actual delimiter differs "
          "from what this script assumes -- tell me what you see and I'll fix the parser.")


def filter_dataset(args):
    print("Parsing categories.txt (this is the biggest file -- may take a while)...")
    categories = _parse_categories_file(args.categories)
    print(f"Parsed {len(categories)} category entries.")

    keywords = [k.lower() for k in args.keywords]
    matching_ids = set()
    for product_id, paths in categories.items():
        combined = " ".join(paths).lower()
        if any(kw in combined for kw in keywords):
            matching_ids.add(product_id)
    print(f"{len(matching_ids)} products match keywords {keywords}")

    print("Parsing titles.txt...")
    titles = _parse_id_value_file(args.titles)
    print("Parsing brands.txt...")
    brands = _parse_id_value_file(args.brands)
    print("Parsing description.txt...")
    descriptions = _parse_description_file(args.descriptions)

    rows = []
    for product_id in matching_ids:
        if product_id not in titles:
            continue
        rows.append({
            "id": product_id,
            "title": titles.get(product_id, ""),
            "brand": brands.get(product_id, ""),
            "description": descriptions.get(product_id, ""),
            "category": " | ".join(categories.get(product_id, [])),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} filtered rows to {args.out}")
    return df


def test_against_model(args):
    from exact_match.inference import ProductComparer

    df = pd.read_csv(args.filtered)
    if len(df) < 2:
        raise SystemExit(f"Only {len(df)} rows in {args.filtered} -- need at least 2 to compare.")

    sample = df.sample(n=min(args.sample_size, len(df)), random_state=42).reset_index(drop=True)
    comparer = ProductComparer(model_dir=args.model_dir)

    print(f"Running {len(sample) - 1} sequential comparisons against a real out-of-training-data sample...\n")
    for i in range(len(sample) - 1):
        a, b = sample.iloc[i], sample.iloc[i + 1]
        result = comparer.compare(
            title_a=a["title"], brand_a=a.get("brand", ""), description_a=a.get("description", ""),
            title_b=b["title"], brand_b=b.get("brand", ""), description_b=b.get("description", ""),
        )
        print(f"[{a['title'][:50]}] vs [{b['title'][:50]}]")
        print(f"  -> {result.relationship or result.prediction} ({result.similarity_score:.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter the large dataset and spot-check it against the trained model")
    parser.add_argument("--categories", default=None)
    parser.add_argument("--titles", default=None)
    parser.add_argument("--brands", default=None)
    parser.add_argument("--descriptions", default=None)
    parser.add_argument("--keywords", nargs="*", default=["electronics", "audio", "headphone", "earbud", "wireless", "speaker"])
    parser.add_argument("--out", default="data/large_dataset_filtered.csv")
    parser.add_argument("--preview", type=int, default=0, help="Preview the first N lines of each file instead of processing everything")
    parser.add_argument("--test-only", action="store_true", help="Skip filtering, just run the model against an already-filtered CSV")
    parser.add_argument("--filtered", default="data/large_dataset_filtered.csv", help="Path to a filtered CSV (used with --test-only)")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--model-dir", default="trained_model")
    args = parser.parse_args()

    if args.preview:
        preview(args)
    elif args.test_only:
        test_against_model(args)
    else:
        filter_dataset(args)
        test_against_model(argparse.Namespace(filtered=args.out, sample_size=args.sample_size, model_dir=args.model_dir))