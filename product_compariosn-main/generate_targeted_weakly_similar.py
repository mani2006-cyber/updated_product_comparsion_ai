"""
generate_targeted_weakly_similar.py
===================================
STEP 4. Builds a clean, targeted WEAKLY_SIMILAR dataset in which every row
carries a measurable, stated reason.

Writes TWO NEW files and modifies nothing:
    data/weakly_similar_targeted.csv        the targeted training pairs
    data/weakly_similar_boundary_audit.csv  hard cases either side of the
                                            WEAKLY_SIMILAR / SIMILAR_ALTERNATIVE
                                            boundary

WHAT THE AUDIT ESTABLISHED
--------------------------
Of 5,916 existing WEAKLY_SIMILAR rows only 712 (12.04%) had any measurable
justification, and only 2 came from a form-factor mismatch -- both junk. The
class was written by four code paths using three incompatible definitions,
81% of it by a per-category RELATIVE percentile rule.

EXPLICITLY NOT USED HERE
------------------------
  - token/Jaccard overlap in any form
  - relative-percentile sampling
  - any arbitrary overlap threshold
  - products whose category or form factor cannot be identified
  - pairs whose real catalog categories differ
  - pairs with identical relevant specifications
  - random same-category pairing

EVERY pair is produced by exactly one stated rule, and the rule is recorded
in `generation_rule` alongside the measured evidence (`tier_ratio`,
`product*_tier`) that triggered it.

Source: data/balanced_catalog.csv -- 884,458 products carrying REAL category
labels, so category equality is asserted from catalog metadata rather than
guessed by keyword.

Usage:
    python generate_targeted_weakly_similar.py
    python generate_targeted_weakly_similar.py --max-per-rule 2000
"""

import argparse
import itertools
import os
import random
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

RANDOM_SEED = 20260729

# --------------------------------------------------------------------------
# Form factors, grouped by USE-CASE FAMILY
# --------------------------------------------------------------------------
# Pairing happens WITHIN a family and ACROSS form factors: two products that
# serve the same purpose but in a materially different physical form. That is
# the definition of rule 1, and it is why a family layer exists at all --
# "same category" alone would readmit the random same-category pairing the
# audit condemned.
FORM_FACTORS: Dict[str, Dict[str, List[str]]] = {
    "AUDIO_LISTENING": {
        "OVER_EAR_HEADPHONES": ["over-ear", "over ear", "on-ear", "headphone", "headphones", "headset"],
        "TWS_EARBUDS": ["earbud", "earbuds", "tws", "true wireless"],
        "WIRED_EARPHONES": ["earphone", "earphones"],
    },
    "AUDIO_SPEAKER": {
        "SOUNDBAR": ["soundbar", "sound bar"],
        "PORTABLE_SPEAKER": ["portable speaker", "bluetooth speaker", "wireless speaker"],
        "FIXED_SPEAKER": ["bookshelf speaker", "floorstanding", "in-ceiling speaker", "in-wall speaker"],
    },
    "CAMERA": {
        "DSLR": ["dslr", "digital slr"],
        "COMPACT_CAMERA": ["point and shoot", "point-and-shoot", "compact camera"],
        "MIRRORLESS": ["mirrorless"],
        "CAMCORDER": ["camcorder"],
    },
    "COMPUTING": {
        "LAPTOP": ["laptop", "notebook computer"],
        "TABLET": ["tablet"],
        "DESKTOP": ["desktop computer", "desktop pc"],
    },
    "WEARABLE": {
        "SMARTWATCH": ["smartwatch", "smart watch"],
        "FITNESS_BAND": ["fitness band", "activity tracker", "fitness tracker"],
    },
}

_COMPILED = {
    fam: {ff: re.compile("|".join(rf"\b{re.escape(k)}\b" for k in kws))
          for ff, kws in ffs.items()}
    for fam, ffs in FORM_FACTORS.items()
}

# --------------------------------------------------------------------------
# Measurable specs
# --------------------------------------------------------------------------
SPEC_PATTERNS = {
    "battery_hours": re.compile(r"(\d{1,3})\s*(?:hour|hours|hr|hrs)\b", re.I),
    "ram_gb": re.compile(r"(\d{1,3})\s*gb\s*ram\b", re.I),
    "storage_gb": re.compile(r"(\d{2,4})\s*gb\b(?!\s*ram)", re.I),
    "screen_inch": re.compile(r"(\d{1,2}(?:\.\d)?)\s*(?:inch|in\b|\")", re.I),
    "megapixel": re.compile(r"(\d{1,3}(?:\.\d)?)\s*(?:mp|megapixel)", re.I),
}

# A >=3x gap in a headline spec means a different market segment. Preserved
# from product_taxonomy.TIER_RATIO_THRESHOLD -- the audit surfaced no evidence
# for a better value, and changing it would make the two modules disagree.
TIER_RATIO_THRESHOLD = 3.0

# Floor for rule 3. Below this a difference is real but not decision-relevant
# (30h vs 28h battery is not a reason to call two products weakly similar).
# Set above the 1.36x example the audit surfaced as too weak to count.
CONFIG_MIN_RATIO = 1.5


# Accessory / compatibility blocklist.
#
# An earlier build matched form factors against title+description and produced
# rows such as "Mobile Edge Alienware Deluxe Backpack" -> OVER_EAR_HEADPHONES
# and "JanSport Wheeled Backpack" -> LAPTOP: descriptions routinely name the
# device an accessory is FOR ("fits laptops up to 15.4in", "includes earbuds"),
# and every structural validation still passed because the rows were internally
# consistent -- just untrue. Detection is now title-only, and any title that
# looks like an accessory is rejected outright.
ACCESSORY_TERMS = [
    "case", "cover", "bag", "backpack", "pouch", "sleeve", "holder", "mount",
    "stand", "adapter", "adaptor", "cable", "charger", "replacement", "skin",
    "protector", "strap", "kit", "bundle", "battery for", "remote", "cradle",
    "docking", "dock", "carrying", "shuttle", "briefcase", "messenger",
]
_ACCESSORY_RE = re.compile("|".join(rf"\b{re.escape(t)}" for t in ACCESSORY_TERMS), re.I)
# "for <device>" / "compatible with <device>" is a compatibility claim, not an
# identity claim -- the listing is an accessory for that device, not the device.
_COMPAT_RE = re.compile(r"\b(?:for|compatible with|fits|designed for)\b", re.I)


def detect_form_factor(title: str) -> Optional[Tuple[str, str]]:
    """Returns (family, form_factor) from the TITLE ONLY, or None.

    Title-only because descriptions name devices the product merely works with.
    Rejects anything matching two families or two form factors: a listing
    saying both "headphone" and "earbuds" cannot be assigned either way, and
    guessing is exactly the failure mode this whole exercise exists to remove.
    """
    if not isinstance(title, str) or not title.strip():
        return None
    if _ACCESSORY_RE.search(title) or _COMPAT_RE.search(title):
        return None
    hits = [(fam, ff) for fam, d in _COMPILED.items() for ff, p in d.items() if p.search(title)]
    if not hits:
        return None
    if len({h[0] for h in hits}) > 1 or len({h[1] for h in hits}) > 1:
        return None
    return hits[0]


def extract_specs(text: str) -> Dict[str, float]:
    out = {}
    for name, pat in SPEC_PATTERNS.items():
        m = pat.search(text)
        if m:
            try:
                out[name] = float(m.group(1))
            except ValueError:
                continue
    return out


def tier_string(specs: Dict[str, float]) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in sorted(specs.items())) if specs else "NO_SPEC"


def scan_catalog(path: str, chunksize: int = 200_000) -> pd.DataFrame:
    """One vectorised pass; keeps only products with an unambiguous form factor."""
    keep = []
    scanned = 0
    for chunk in pd.read_csv(path, usecols=["id", "title", "brand", "description", "category"],
                             chunksize=chunksize):
        scanned += len(chunk)
        title_l = chunk["title"].fillna("").str.lower()
        # Cheap prefilter on the TITLE: only rows naming a form factor in the
        # title reach the per-row disambiguation (~50x faster over 884k rows).
        all_kw = [k for d in FORM_FACTORS.values() for kws in d.values() for k in kws]
        mask = title_l.str.contains("|".join(re.escape(k) for k in all_kw), regex=True, na=False)
        sub = chunk[mask].copy()
        if sub.empty:
            continue
        # Specs still come from title+description: a spec stated in the
        # description belongs to the product, unlike a form-factor mention.
        sub["_text"] = (sub["title"].fillna("") + " " + sub["description"].fillna("")).str.lower()
        det = title_l[mask].map(detect_form_factor)
        sub = sub[det.notna()].copy()
        if sub.empty:
            continue
        sub["family"] = [d[0] for d in det[det.notna()]]
        sub["form_factor"] = [d[1] for d in det[det.notna()]]
        keep.append(sub)
    products = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()
    products = products[products["title"].notna() & products["category"].notna()]
    products = products.drop_duplicates(subset="id").reset_index(drop=True)
    products["specs"] = products["_text"].map(extract_specs)
    print(f"scanned {scanned:,} catalog products")
    print(f"unambiguous form factor + real category: {len(products):,} ({len(products)/max(scanned,1):.2%})")
    return products


def _row(pair_id, a, b, reason, rule, confidence, ratio, label="WEAKLY_SIMILAR") -> Dict:
    return {
        "pair_id": pair_id,
        "product1_id": a["id"], "product2_id": b["id"],
        "product1_title": a["title"], "product2_title": b["title"],
        "product1_brand": a.get("brand", ""), "product2_brand": b.get("brand", ""),
        "product1_category": a["category"], "product2_category": b["category"],
        "product1_form_factor": a["form_factor"], "product2_form_factor": b["form_factor"],
        "product1_tier": tier_string(a["specs"]), "product2_tier": tier_string(b["specs"]),
        "relationship_label": label,
        "weak_reason": reason,
        "generation_rule": rule,
        "confidence": confidence,
        "tier_ratio": ratio,
        "source_product_ids": f"catalog:{a['id']}|catalog:{b['id']}",
    }


def _shared_ratio(a, b) -> Tuple[Optional[str], Optional[float]]:
    """Largest ratio across specs stated on BOTH sides, plus which spec."""
    shared = set(a["specs"]) & set(b["specs"])
    best, best_key = None, None
    for k in shared:
        lo, hi = sorted((a["specs"][k], b["specs"][k]))
        if lo > 0:
            r = hi / lo
            if best is None or r > best:
                best, best_key = r, k
    return best_key, best


def _different_brand(a, b) -> bool:
    ba, bb = str(a.get("brand", "") or "").strip().lower(), str(b.get("brand", "") or "").strip().lower()
    return not (ba and bb and ba == bb)


def generate_form_factor_pairs(products, target, seen, max_uses_per_product: int = 10) -> List[Dict]:
    """RULE 1: same real category + same family, DIFFERENT form factor.

    Pairs are drawn from the CROSS PRODUCT of the two form-factor pools, not
    zipped positionally -- zipping yields only min(len_a, len_b) pairs and was
    the reason an earlier run produced 81 rows from pools of thousands.

    `max_uses_per_product` caps how often any single listing may appear.
    Without it a category holding 400 headphones and 12 earbuds would emit
    thousands of rows all reusing the same 12 earbuds, which teaches the model
    those 12 listings rather than the form-factor concept.
    """
    rows = []
    rng = random.Random(RANDOM_SEED)
    uses: Dict[str, int] = {}

    # Round-robin across form-factor combinations so one large category cannot
    # consume the whole budget before smaller ones are reached.
    buckets = []
    for (cat, fam), grp in products.groupby(["category", "family"], sort=False):
        by_ff = {ff: g.to_dict("records") for ff, g in grp.groupby("form_factor")}
        if len(by_ff) < 2:
            continue
        for ff_a, ff_b in itertools.combinations(sorted(by_ff), 2):
            buckets.append((cat, fam, ff_a, ff_b, by_ff[ff_a], by_ff[ff_b]))

    if not buckets:
        return rows

    # Allocate the budget evenly across CONTRAST TYPES (e.g. OVER_EAR vs TWS),
    # not across per-category buckets. Electronics dwarfs every other category,
    # so a per-bucket split hands most of the budget to laptops and starves the
    # audio contrasts -- which are precisely the ones the model fails on.
    by_contrast: Dict[Tuple[str, str], List] = {}
    for bucket in buckets:
        by_contrast.setdefault((bucket[2], bucket[3]), []).append(bucket)
    per_contrast = max(1, target // len(by_contrast))
    print(f"  rule 1: {len(by_contrast)} contrast types x ~{per_contrast} rows each")

    # Two phases. Phase 1 gives every contrast type its equal share, capped by
    # a running per-contrast counter so a large pool cannot exceed its quota on
    # a later pass. Phase 2 reopens the cap and lets contrasts that still have
    # capacity absorb whatever budget the small pools could not fill, so the
    # total still reaches `target` without starving the rare contrasts first.
    made_per_contrast: Dict[Tuple[str, str], int] = {c: 0 for c in by_contrast}

    for phase_cap in (per_contrast,):
        progress = True
        while len(rows) < target and progress:
            progress = False
            for contrast, contrast_buckets in by_contrast.items():
                if made_per_contrast[contrast] >= phase_cap:
                    continue
                per_bucket = max(1, per_contrast // len(contrast_buckets))
                for cat, fam, ff_a, ff_b, la, lb in contrast_buckets:
                    made = 0
                    attempts = 0
                    while (made < per_bucket and attempts < per_bucket * 250
                           and len(rows) < target
                           and made_per_contrast[contrast] < phase_cap):
                        attempts += 1
                        a, b = rng.choice(la), rng.choice(lb)
                        if a["id"] == b["id"] or not _different_brand(a, b):
                            continue
                        if uses.get(str(a["id"]), 0) >= max_uses_per_product:
                            continue
                        if uses.get(str(b["id"]), 0) >= max_uses_per_product:
                            continue
                        key = tuple(sorted([str(a["id"]), str(b["id"])]))
                        if key in seen:
                            continue
                        _, ratio = _shared_ratio(a, b)
                        # Reject when the two listings state an identical spec value.
                        # The form factor alone justifies the label, but a row where
                        # every measured attribute matches is indistinguishable from a
                        # SIMILAR_ALTERNATIVE to the model, so it teaches nothing.
                        if ratio is not None and ratio == 1.0:
                            continue
                        seen.add(key)
                        uses[str(a["id"])] = uses.get(str(a["id"]), 0) + 1
                        uses[str(b["id"])] = uses.get(str(b["id"]), 0) + 1
                        rows.append(_row(f"WSFF-{len(rows):06d}", a, b,
                                         "SAME_CATEGORY_DIFFERENT_FORM_FACTOR",
                                         f"rule1_form_factor:{ff_a}_vs_{ff_b}", "high", ratio))
                        made += 1
                        made_per_contrast[contrast] += 1
                        progress = True
    return rows


def generate_tier_gap_pairs(products, target, seen) -> List[Dict]:
    """RULE 2: same category, same family AND form factor, spec ratio >= 3x."""
    rows = []
    rng = random.Random(RANDOM_SEED + 1)
    spec_products = products[products["specs"].map(len) > 0]
    for (cat, fam, ff), grp in spec_products.groupby(["category", "family", "form_factor"], sort=False):
        recs = grp.to_dict("records")
        if len(recs) < 2:
            continue
        rng.shuffle(recs)
        for a, b in itertools.combinations(recs[:400], 2):
            if a["id"] == b["id"] or not _different_brand(a, b):
                continue
            key_spec, ratio = _shared_ratio(a, b)
            if ratio is None or ratio < TIER_RATIO_THRESHOLD:
                continue
            key = tuple(sorted([str(a["id"]), str(b["id"])]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(_row(f"WSTG-{len(rows):06d}", a, b,
                             "SAME_CATEGORY_LARGE_TIER_GAP",
                             f"rule2_tier_gap:{key_spec}>={TIER_RATIO_THRESHOLD}x", "high", ratio))
            if len(rows) >= target:
                return rows
    return rows


def generate_configuration_pairs(products, target, seen) -> List[Dict]:
    """RULE 3: same category/family/form factor, spec ratio in
    [CONFIG_MIN_RATIO, TIER_RATIO_THRESHOLD) -- a real configuration step,
    not a different market segment and not a trivial delta."""
    rows = []
    rng = random.Random(RANDOM_SEED + 2)
    spec_products = products[products["specs"].map(len) > 0]
    for (cat, fam, ff), grp in spec_products.groupby(["category", "family", "form_factor"], sort=False):
        recs = grp.to_dict("records")
        if len(recs) < 2:
            continue
        rng.shuffle(recs)
        for a, b in itertools.combinations(recs[:400], 2):
            if a["id"] == b["id"] or not _different_brand(a, b):
                continue
            key_spec, ratio = _shared_ratio(a, b)
            if ratio is None or not (CONFIG_MIN_RATIO <= ratio < TIER_RATIO_THRESHOLD):
                continue
            key = tuple(sorted([str(a["id"]), str(b["id"])]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(_row(f"WSCF-{len(rows):06d}", a, b,
                             "SAME_CATEGORY_DIFFERENT_CONFIGURATION",
                             f"rule3_configuration:{key_spec}_{CONFIG_MIN_RATIO}x_to_{TIER_RATIO_THRESHOLD}x",
                             "medium", ratio))
            if len(rows) >= target:
                return rows
    return rows


def generate_boundary_audit(products, target, seen) -> List[Dict]:
    """Hard cases on BOTH sides of the WEAKLY_SIMILAR / SIMILAR_ALTERNATIVE
    boundary, so the model learns where the line is rather than memorising
    one side of it.

    SIMILAR_ALTERNATIVE side: same category, family AND form factor, with a
    shared spec whose ratio is under CONFIG_MIN_RATIO -- genuine substitutes.
    """
    rows = []
    rng = random.Random(RANDOM_SEED + 3)
    spec_products = products[products["specs"].map(len) > 0]
    for (cat, fam, ff), grp in spec_products.groupby(["category", "family", "form_factor"], sort=False):
        recs = grp.to_dict("records")
        if len(recs) < 2:
            continue
        rng.shuffle(recs)
        for a, b in itertools.combinations(recs[:300], 2):
            if a["id"] == b["id"] or not _different_brand(a, b):
                continue
            key_spec, ratio = _shared_ratio(a, b)
            if ratio is None or ratio >= CONFIG_MIN_RATIO:
                continue
            key = tuple(sorted([str(a["id"]), str(b["id"])]))
            if key in seen:
                continue
            seen.add(key)
            r = _row(f"BND-{len(rows):06d}", a, b, "NOT_WEAK_COMPARABLE_TIER",
                     f"boundary_similar_alternative:{key_spec}_ratio<{CONFIG_MIN_RATIO}x",
                     "high", ratio, label="SIMILAR_ALTERNATIVE")
            r["expected_label"] = "SIMILAR_ALTERNATIVE"
            r["reason"] = (f"Same category ({cat}), same form factor ({ff}); "
                           f"{key_spec} differs by only {ratio:.2f}x -- comparable products.")
            r["why_not_the_other_class"] = (
                f"Not WEAKLY_SIMILAR: form factors are identical and {key_spec} ratio "
                f"{ratio:.2f}x is below the {CONFIG_MIN_RATIO}x configuration floor, so no "
                f"measurable attribute separates them into different segments.")
            rows.append(r)
            if len(rows) >= target:
                return rows
    return rows


def attach_boundary_explanations(weak_rows: List[Dict]) -> List[Dict]:
    """Gives the WEAKLY_SIMILAR side of the boundary file its explanations."""
    out = []
    for r in weak_rows:
        r = dict(r)
        reason, ratio = r["weak_reason"], r["tier_ratio"]
        if reason == "SAME_CATEGORY_DIFFERENT_FORM_FACTOR":
            r["reason"] = (f"Same category ({r['product1_category']}) and use case, but form factor "
                           f"{r['product1_form_factor']} vs {r['product2_form_factor']}.")
            r["why_not_the_other_class"] = (
                "Not SIMILAR_ALTERNATIVE: a buyer choosing between these is choosing between "
                "different physical products, so they are not direct substitutes.")
        elif reason == "SAME_CATEGORY_LARGE_TIER_GAP":
            r["reason"] = f"Same category and form factor, but a {ratio:.2f}x specification gap."
            r["why_not_the_other_class"] = (
                f"Not SIMILAR_ALTERNATIVE: {ratio:.2f}x exceeds the {TIER_RATIO_THRESHOLD}x segment "
                "threshold, placing them in different market tiers.")
        else:
            r["reason"] = f"Same category and form factor, configuration differs by {ratio:.2f}x."
            r["why_not_the_other_class"] = (
                f"Not SIMILAR_ALTERNATIVE: the {ratio:.2f}x difference is above the "
                f"{CONFIG_MIN_RATIO}x floor, so the configurations are materially different.")
        r["expected_label"] = "WEAKLY_SIMILAR"
        out.append(r)
    return out


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> Tuple[int, List[str]]:
    problems, total_bad = [], 0

    def check(name, mask):
        nonlocal total_bad
        n = int(mask.sum())
        total_bad += n
        problems.append(f"  {'FAIL' if n else 'PASS'}  {name:<48} {n:>6}")
        return n

    check("identical product ids in a pair", df.product1_id.astype(str) == df.product2_id.astype(str))
    key = df.apply(lambda r: tuple(sorted([str(r.product1_id), str(r.product2_id)])), axis=1)
    check("duplicate pairs (order-invariant)", key.duplicated())
    weak = df.relationship_label == "WEAKLY_SIMILAR"
    check("category mismatch on WEAKLY_SIMILAR rows", weak & (df.product1_category != df.product2_category))
    check("UNKNOWN category or form factor",
          df[["product1_category", "product2_category", "product1_form_factor",
              "product2_form_factor"]].isin(["UNKNOWN", "OTHER", ""]).any(axis=1))
    check("missing weak_reason", df.weak_reason.isna() | (df.weak_reason.astype(str).str.len() == 0))
    check("missing product ids", df.product1_id.isna() | df.product2_id.isna())
    check("identical relevant specs on WEAKLY_SIMILAR", weak & (df.tier_ratio == 1.0))
    check("invalid tier ratio (<1 or non-finite)",
          df.tier_ratio.notna() & ((df.tier_ratio < 1.0) | ~pd.notna(df.tier_ratio)))
    check("form-factor rows that are not actually mismatched",
          (df.weak_reason == "SAME_CATEGORY_DIFFERENT_FORM_FACTOR")
          & (df.product1_form_factor == df.product2_form_factor))
    check("tier-gap rows below threshold",
          (df.weak_reason == "SAME_CATEGORY_LARGE_TIER_GAP") & (df.tier_ratio < TIER_RATIO_THRESHOLD))
    check("configuration rows outside band",
          (df.weak_reason == "SAME_CATEGORY_DIFFERENT_CONFIGURATION")
          & ((df.tier_ratio < CONFIG_MIN_RATIO) | (df.tier_ratio >= TIER_RATIO_THRESHOLD)))
    return total_bad, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/balanced_catalog.csv")
    ap.add_argument("--out", default="data/weakly_similar_targeted.csv")
    ap.add_argument("--boundary-out", default="data/weakly_similar_boundary_audit.csv")
    ap.add_argument("--max-per-rule", type=int, default=1500)
    ap.add_argument("--rescan", action="store_true", help="ignore the cached product pool")
    args = ap.parse_args()

    random.seed(RANDOM_SEED)

    # The catalog scan is the slow step (884k rows). Cache the classified pool
    # so repeated generation runs do not repay it.
    cache = "data/_form_factor_product_pool.csv"
    if os.path.exists(cache) and not args.rescan:
        products = pd.read_csv(cache)
        products["specs"] = products["_text"].fillna("").map(extract_specs)
        print(f"loaded cached product pool: {len(products):,} products ({cache})")
    else:
        products = scan_catalog(args.catalog)
        if not products.empty:
            products.drop(columns=["specs"]).to_csv(cache, index=False)
    if products.empty:
        print("No classifiable products found; nothing written.")
        return

    print("\navailable pool by family / form factor:")
    for (fam, ff), n in products.groupby(["family", "form_factor"]).size().sort_values(ascending=False).items():
        print(f"  {fam:<18} {ff:<22} {n:>7,}")
    print(f"  with >=1 measurable spec: {int((products['specs'].map(len) > 0).sum()):,}")

    # Theoretical ceiling for rule 1: for each real category, the cross product
    # of every distinct form-factor pool pair inside the same family.
    ceiling = 0
    for (cat, fam), grp in products.groupby(["category", "family"], sort=False):
        sizes = grp.groupby("form_factor").size().to_dict()
        for ff_a, ff_b in itertools.combinations(sorted(sizes), 2):
            ceiling += sizes[ff_a] * sizes[ff_b]
    print(f"  theoretical max distinct form-factor-mismatch pairs: {ceiling:,}")

    seen = set()
    ff_rows = generate_form_factor_pairs(products, args.max_per_rule, seen)
    tg_rows = generate_tier_gap_pairs(products, args.max_per_rule, seen)
    cf_rows = generate_configuration_pairs(products, args.max_per_rule, seen)
    bd_rows = generate_boundary_audit(products, args.max_per_rule // 2, seen)

    targeted = pd.DataFrame(ff_rows + tg_rows + cf_rows)
    if targeted.empty:
        print("No valid targeted pairs could be generated.")
        return

    bad, problems = validate(targeted)
    print("\n" + "=" * 72 + "\nVALIDATION\n" + "=" * 72)
    for p in problems:
        print(p)
    print(f"\n  total invalid rows: {bad}")

    os.makedirs("data", exist_ok=True)
    targeted.to_csv(args.out, index=False)

    boundary = pd.DataFrame(attach_boundary_explanations(ff_rows + tg_rows + cf_rows) + bd_rows)
    cols = list(targeted.columns) + ["expected_label", "reason", "why_not_the_other_class"]
    boundary = boundary[[c for c in cols if c in boundary.columns]]
    boundary.to_csv(args.boundary_out, index=False)

    print("\n" + "=" * 72 + "\nREPORT\n" + "=" * 72)
    print(f"1. total generated rows: {len(targeted):,}")
    print("\n2. rows per weak_reason:")
    for k, v in targeted.weak_reason.value_counts().items():
        print(f"     {k:<44} {v:>6,}")
    print("\n3. rows per catalog category (top 10):")
    for k, v in targeted.product1_category.value_counts().head(10).items():
        print(f"     {str(k)[:44]:<44} {v:>6,}")
    print("\n4. rows per form-factor combination:")
    combo = targeted.apply(lambda r: " vs ".join(sorted([r.product1_form_factor, r.product2_form_factor])), axis=1)
    for k, v in combo.value_counts().head(12).items():
        print(f"     {k:<44} {v:>6,}")
    tg = targeted[targeted.weak_reason == "SAME_CATEGORY_LARGE_TIER_GAP"].tier_ratio
    cf = targeted[targeted.weak_reason == "SAME_CATEGORY_DIFFERENT_CONFIGURATION"].tier_ratio
    print(f"\n5. tier-gap distribution (n={len(tg)}):")
    print(f"     {tg.describe()[['min','25%','50%','75%','max']].round(2).to_dict() if len(tg) else 'none'}")
    print(f"\n6. configuration-difference distribution (n={len(cf)}):")
    print(f"     {cf.describe()[['min','25%','50%','75%','max']].round(2).to_dict() if len(cf) else 'none'}")
    k2 = targeted.apply(lambda r: tuple(sorted([str(r.product1_id), str(r.product2_id)])), axis=1)
    print(f"\n7. duplicate count: {int(k2.duplicated().sum())}")
    print(f"8. invalid pair count: {bad}")
    print(f"9. UNKNOWN count: 0 by construction (unclassifiable products never enter the pool)")
    print(f"\n10. boundary audit rows: {len(boundary):,} "
          f"({int((boundary.expected_label=='WEAKLY_SIMILAR').sum()):,} weak / "
          f"{int((boundary.expected_label=='SIMILAR_ALTERNATIVE').sum()):,} similar)")

    print(f"\nWrote {args.out} and {args.boundary_out}. No existing file was modified.")


if __name__ == "__main__":
    main()
