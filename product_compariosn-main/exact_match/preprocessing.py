"""
preprocessing.py
=================
Step 1 (load), Step 2 (clean), and Step 5 (train/val split) of the
pipeline. Kept separate from dataset.py (which handles tokenization /
torch Dataset wrapping) so each file has one job.

Supports two input schemas transparently:

  A) CSV, title-only:
        product1, product2, label

  B) JSON / JSONL, title + specs:
        product1_title, product1_specs,
        product2_title, product2_specs, label
"""

import json
import os
import re
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Step 1: Load
# --------------------------------------------------------------------------
def load_raw_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Loads CSV, JSON (list of records), or JSONL into a DataFrame."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext == ".jsonl":
        records = [json.loads(line) for line in open(path, "r", encoding="utf-8") if line.strip()]
        df = pd.DataFrame(records)
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data if isinstance(data, list) else [data])
    else:
        raise ValueError(f"Unsupported data file extension: {ext}")

    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def _detect_schema(df: pd.DataFrame) -> str:
    if set(config.TEXT_COLUMNS_FULL).issubset(df.columns):
        return "full"
    if set(config.TEXT_COLUMNS_WITH_SPECS).issubset(df.columns):
        return "with_specs"
    if set(config.TEXT_COLUMNS_TITLE_ONLY).issubset(df.columns):
        return "title_only"
    raise ValueError(
        "Dataset does not match any supported schema.\n"
        f"  title_only schema needs columns: {config.TEXT_COLUMNS_TITLE_ONLY}\n"
        f"  with_specs schema needs columns: {config.TEXT_COLUMNS_WITH_SPECS}\n"
        f"  full schema needs columns: {config.TEXT_COLUMNS_FULL}\n"
        f"  Found columns: {list(df.columns)}"
    )


# --------------------------------------------------------------------------
# Step 2: Clean
# --------------------------------------------------------------------------
_WHITESPACE_RE = re.compile(r"\s+")
_UNIT_SPACING_RE = re.compile(r"(\d)\s*(gb|tb|mb|mp|inch|in|hz|w|mah|ghz)\b", re.IGNORECASE)


def clean_text(text: str) -> str:
    """
    Normalizes messy e-commerce text so that trivially-different strings
    that mean the same thing ("128GB" vs "128 GB") don't confuse the
    tokenizer as much:
      - lowercasing
      - collapsing whitespace
      - normalizing "128 GB" -> "128gb" style unit spacing
      - stripping stray punctuation noise (but keeping - and . which
        matter for model numbers like "A16" / "6.1")
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    text = text.strip().lower()
    text = _UNIT_SPACING_RE.sub(lambda m: f"{m.group(1)}{m.group(2).lower()}", text)
    text = re.sub(r"[^a-z0-9.\-\s]", " ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def build_product_text(title: str, brand: str = "", specs: str = "", description: str = "") -> str:
    """Combines a product's title, brand, and specs/description into one
    string. `specs` and `description` are treated as equivalent free-text
    detail fields (kept as separate params for call-site clarity)."""
    parts = [clean_text(title)]

    brand_c = clean_text(brand)
    if brand_c:
        parts.append(f"brand {brand_c}")

    detail = specs or description
    detail_c = clean_text(detail)
    if detail_c:
        parts.append(detail_c)

    return " | ".join(p for p in parts if p)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Detects schema (title-only vs title+specs)
    - Drops rows with missing/empty required fields or invalid labels
    - Normalizes into two canonical columns: text_a, text_b, label
    """
    schema = _detect_schema(df)
    logger.info(f"Detected dataset schema: {schema}")

    df = df.copy()
    if config.NUM_LABELS == 5:
        df = df.dropna(subset=["relationship_label"])
        df["label"] = df["relationship_label"].map(config.RELATIONSHIP_LABEL_MAP)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)
    else:
        df = df.dropna(subset=[config.LABEL_COLUMN])
        df[config.LABEL_COLUMN] = pd.to_numeric(df[config.LABEL_COLUMN], errors="coerce")
        df = df.dropna(subset=[config.LABEL_COLUMN])
        df = df[df[config.LABEL_COLUMN].isin([0, 1])]
        df[config.LABEL_COLUMN] = df[config.LABEL_COLUMN].astype(int)

    if schema == "full":
        df["text_a"] = df.apply(
            lambda r: build_product_text(
                r.get("product1_title", ""), brand=r.get("product1_brand", ""),
                description=r.get("product1_description", "")
            ),
            axis=1,
        )
        df["text_b"] = df.apply(
            lambda r: build_product_text(
                r.get("product2_title", ""), brand=r.get("product2_brand", ""),
                description=r.get("product2_description", "")
            ),
            axis=1,
        )
    elif schema == "with_specs":
        df["text_a"] = df.apply(
            lambda r: build_product_text(r.get("product1_title", ""), r.get("product1_specs", "")),
            axis=1,
        )
        df["text_b"] = df.apply(
            lambda r: build_product_text(r.get("product2_title", ""), r.get("product2_specs", "")),
            axis=1,
        )
    else:  # title_only
        df["text_a"] = df["product1"].apply(build_product_text)
        df["text_b"] = df["product2"].apply(build_product_text)

    before = len(df)
    df = df[(df["text_a"].str.len() > 0) & (df["text_b"].str.len() > 0)]
    dropped = before - len(df)
    if dropped:
        logger.info(f"Dropped {dropped} rows with empty text after cleaning")

    df = df.drop_duplicates(subset=["text_a", "text_b", config.LABEL_COLUMN]).reset_index(drop=True)
    logger.info(f"Clean dataset size: {len(df)} rows | label counts: "
                f"{df[config.LABEL_COLUMN].value_counts().to_dict()}")
    return df[["text_a", "text_b", config.LABEL_COLUMN]].rename(columns={config.LABEL_COLUMN: "label"})


# --------------------------------------------------------------------------
# Step 5: Train / Validation / Test split
# --------------------------------------------------------------------------
def split_data(
    df: pd.DataFrame,
    val_ratio: float = config.VAL_SPLIT_RATIO,
    test_ratio: float = config.TEST_SPLIT_RATIO,
    seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits into train / val / test. Stratifies on the label when the
    dataset is large enough for every split to keep both classes
    (falls back to a plain random split on tiny datasets).
    """
    stratify_col = df["label"] if config.STRATIFY_SPLITS else None

    try:
        train_val_df, test_df = train_test_split(
            df, test_size=test_ratio, random_state=seed, stratify=stratify_col
        )
        strat2 = train_val_df["label"] if config.STRATIFY_SPLITS else None
        relative_val = val_ratio / (1 - test_ratio)
        train_df, val_df = train_test_split(
            train_val_df, test_size=relative_val, random_state=seed, stratify=strat2
        )
    except ValueError:
        # Not enough samples per class to stratify -- fall back gracefully.
        logger.warning("Stratified split failed (too few samples per class); using random split.")
        train_val_df, test_df = train_test_split(df, test_size=test_ratio, random_state=seed)
        relative_val = val_ratio / (1 - test_ratio)
        train_df, val_df = train_test_split(train_val_df, test_size=relative_val, random_state=seed)

    logger.info(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def load_clean_split(path: str = config.RAW_DATA_PATH):
    """Convenience one-shot: load -> clean -> split."""
    raw_df = load_raw_data(path)
    clean_df = clean_dataframe(raw_df)
    return split_data(clean_df)