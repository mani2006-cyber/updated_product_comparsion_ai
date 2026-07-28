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

import numpy as np
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

    _sorted = df.apply(lambda r: tuple(sorted([r["text_a"], r["text_b"]])), axis=1)
    df["_dedup_a"], df["_dedup_b"] = zip(*_sorted)
    df = (
    df.drop_duplicates(subset=["_dedup_a", "_dedup_b", config.LABEL_COLUMN])
      .drop(columns=["_dedup_a", "_dedup_b"])
      .reset_index(drop=True)
    )
    logger.info(f"Clean dataset size: {len(df)} rows | label counts: "
                f"{df[config.LABEL_COLUMN].value_counts().to_dict()}")
    return df[["text_a", "text_b", config.LABEL_COLUMN]].rename(columns={config.LABEL_COLUMN: "label"})


# --------------------------------------------------------------------------
# Step 5: Train / Validation / Test split
# --------------------------------------------------------------------------
# Splitting at the pair-row level lets the SAME product appear in a training
# pair and a different test pair. Measured on relationship_pairs_final.csv,
# 29.48% of test rows involved a product already seen in training, which
# inflates reported accuracy relative to genuinely unseen products.
#
# Entity-level splitting fixes this by grouping pairs into connected
# components of the product graph: if (A,B) and (B,C) are both pairs, all of
# A, B and C must land in the same split, or B leaks. Measured component
# structure on the same file: 47,678 components, largest 6.2% of rows, 44,754
# singletons -- comfortably splittable at a 70/15/15 ratio.
ENTITY_LEVEL_SPLIT = True

# Below this many groups, group-wise splitting cannot hit the target ratios
# and we fall back to the old stratified row split (with a warning).
_MIN_GROUPS_FOR_ENTITY_SPLIT = 10


def _entity_group_ids(df: pd.DataFrame) -> pd.Series:
    """Assigns each pair the id of its connected component in the product
    graph, using union-find with path compression."""
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:            # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in zip(df["text_a"], df["text_b"]):
        union(a, b)
    # Roots must be read only after every union is applied.
    return df["text_a"].map(find)


def split_data(
    df: pd.DataFrame,
    val_ratio: float = config.VAL_SPLIT_RATIO,
    test_ratio: float = config.TEST_SPLIT_RATIO,
    seed: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits into train / val / test.

    With ENTITY_LEVEL_SPLIT (default), splits by product entity so no product
    appears in more than one split. Label stratification is not possible
    alongside group constraints -- whole components move together -- so the
    resulting per-split label balance is logged instead, and skew is reported.

    Falls back to the previous stratified row split when there are too few
    groups for group-wise splitting to be meaningful.
    """
    if ENTITY_LEVEL_SPLIT:
        groups = _entity_group_ids(df)
        n_groups = groups.nunique()
        if n_groups >= _MIN_GROUPS_FOR_ENTITY_SPLIT:
            # Greedy size-aware bin packing rather than GroupShuffleSplit.
            # GroupShuffleSplit allocates a fraction of GROUPS, but components
            # range from 1 to 4,131 rows here, so equal group counts gave very
            # unequal row counts (measured 63/23/14 against a 70/15/15 target).
            # Walking components largest-first and giving each to whichever
            # split is furthest below its row quota keeps rows on target while
            # still moving whole components together.
            # Components are label-homogeneous (the generators emit long chains
            # of one class), so packing purely by size sent every large
            # SIMILAR/WEAKLY component to train and left val/test with
            # singletons: measured 11.6% WEAKLY_SIMILAR in train vs 2.3% in
            # test. Assignment therefore scores BOTH row quota and label
            # balance, choosing the split that keeps each class closest to its
            # global proportion.
            sizes = groups.value_counts()
            comp_labels = (pd.crosstab(groups, df["label"])
                           .reindex(sizes.index)
                           .to_numpy(dtype=float))
            label_cols = sorted(df["label"].unique())
            global_frac = (df["label"].value_counts(normalize=True)
                           .reindex(label_cols).to_numpy(dtype=float))

            rng = np.random.RandomState(seed)
            tie_break = rng.permutation(len(sizes))
            order = sorted(range(len(sizes)), key=lambda i: (-sizes.iloc[i], tie_break[i]))

            n = len(df)
            splits = ["train", "val", "test"]
            quota = np.array([n * (1 - val_ratio - test_ratio), n * val_ratio, n * test_ratio])
            filled = np.zeros(3)
            filled_labels = np.zeros((3, len(label_cols)))

            assignment = {}
            for i in order:
                comp_vec, comp_size = comp_labels[i], sizes.iloc[i]
                best, best_cost = 0, None
                for s in range(3):
                    if filled[s] + comp_size > quota[s] * 1.02 and filled.sum() < n:
                        continue                      # respect the row quota
                    after = filled_labels[s] + comp_vec
                    total = after.sum()
                    # L1 distance from the global class distribution.
                    div = np.abs(after / total - global_frac).sum() if total else 0.0
                    # Prefer the emptier split when divergence is comparable.
                    cost = div + 0.5 * (filled[s] / quota[s])
                    if best_cost is None or cost < best_cost:
                        best, best_cost = s, cost
                if best_cost is None:                 # every split at quota
                    best = int(np.argmax(quota - filled))
                assignment[sizes.index[i]] = splits[best]
                filled[best] += comp_size
                filled_labels[best] += comp_vec

            split_of = groups.map(assignment)
            train_df = df[split_of == "train"]
            val_df = df[split_of == "val"]
            test_df = df[split_of == "test"]

            logger.info(f"Entity-level split over {n_groups:,} connected components")
            logger.info(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")
            for name, part in (("train", train_df), ("val", val_df), ("test", test_df)):
                dist = (part["label"].value_counts(normalize=True).sort_index() * 100).round(1).to_dict()
                logger.info(f"  {name} label %: {dist}")
            return (train_df.reset_index(drop=True),
                    val_df.reset_index(drop=True),
                    test_df.reset_index(drop=True))
        logger.warning(
            f"Only {n_groups} entity groups (<{_MIN_GROUPS_FOR_ENTITY_SPLIT}); "
            "falling back to a row-level stratified split -- entity leakage is NOT prevented.")

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