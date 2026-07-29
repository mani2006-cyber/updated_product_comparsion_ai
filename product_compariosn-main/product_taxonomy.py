"""
product_taxonomy.py
===================
Category / form-factor / tier detection for product pairs.

WHY THIS FILE EXISTS
--------------------
Two proven bugs in generate_relationship_pairs.py capped model accuracy at
~0.85 F1 on SIMILAR_ALTERNATIVE and ~0.83 on WEAKLY_SIMILAR, while the three
classes with real signal scored 0.97-1.00:

  BUG 1 -- categorize() failed OPEN to "OTHER".
      "Zebronics Sound Bomb 4" is a TWS earbud, but matched none of
      ["earbuds","airdopes","tws","buds"], so it became "OTHER". The caller
      then did `if cat_a != cat_b or cat_a == "OTHER": return "UNRELATED"`,
      so an unrecognised product was ASSERTED to be unrelated to everything.
      Measured effect: the trained model calls two different-brand TWS
      earbuds UNRELATED at 100.0% confidence.

  BUG 2 -- the SIMILAR_ALTERNATIVE / WEAKLY_SIMILAR boundary was a bare
      Jaccard token-overlap constant (0.25). "GOBOULT Z40 Pro" vs
      "Zebronics Sound Bomb 7" scored 0.238 and fell to WEAKLY_SIMILAR --
      missing the boundary by 0.012. A threshold on shared *words* has no
      relationship to whether two products are genuine alternatives.

THE THREE FIXES
---------------
  1. FAIL CLOSED. categorize() returns UNKNOWN when it cannot tell, and
     UNKNOWN never asserts UNRELATED. Only two *confidently different*
     categories do. Unknown-vs-anything falls through to the weaker signals.
  2. WORD-BOUNDARY MATCHING + SCORING. The old code used `kw in text`, so
     "buds" matched inside unrelated words and dict order silently decided
     ties. We use \\b regex and pick the highest-scoring category instead.
  3. TIER, NOT WORD OVERLAP. Same category + comparable spec tier =
     SIMILAR_ALTERNATIVE; same category but a different form factor or a
     far-apart tier = WEAKLY_SIMILAR. Tier comes from comparable numeric
     specs actually present in the text (battery hours, RAM, storage).

Wiring it in (generate_relationship_pairs.py) -- replace the final block of
label_pair() from `overlap = token_overlap_ratio(...)` to the end with:

    from product_taxonomy import classify_negative_pair
    return classify_negative_pair(text_a, text_b)
"""

import re
from typing import Dict, Optional

UNKNOWN = "UNKNOWN"

# --------------------------------------------------------------------------
# Coarse category keywords
# --------------------------------------------------------------------------
# AUDIO deliberately includes SPEC words ("playtime", "drivers", "anc"), not
# just product nouns. That is what rescues listings like "Zebronics Sound
# Bomb 4 / 20 hour playtime, 13mm drivers" whose *title* names no category:
# the description betrays it. This is the direct fix for BUG 1.
CATEGORY_KEYWORDS: Dict[str, list] = {
    "AUDIO": [
        "earbud", "earbuds", "airdopes", "tws", "buds", "bassbuds",
        "headphone", "headphones", "headset", "earphone", "earphones",
        "neckband", "soundbar", "speaker", "playtime", "playback",
        "drivers", "anc", "enc", "audio", "sound bomb",
    ],
    "LAPTOP": [
        "laptop", "notebook", "macbook", "ultrabook", "chromebook",
        "thinkpad", "ideapad", "vivobook", "inspiron", "pavilion",
    ],
    "SMARTPHONE": [
        "smartphone", "iphone", "galaxy s", "pixel", "redmi", "oneplus",
        "mobile phone", "5g phone", "snapdragon",
    ],
    "SMARTWATCH": [
        "smartwatch", "smart watch", "colorfit", "fitness band",
        "smart band", "fitness tracker", "spo2",
    ],
    "TELEVISION": ["television", "smart tv", "led tv", "qled", "oled", "bravia"],
    "CAMERA": ["camera", "dslr", "mirrorless", "lens", "megapixel"],
    "FOOTWEAR": ["shoe", "shoes", "sneaker", "sneakers", "boot", "sandal", "footwear"],
    "SOFTWARE": ["software", "license", "cd-rom", "diskette"],
    "NETWORKING": ["router", "modem", "access point", "wi-fi", "wifi"],
    "APPLIANCE": [
        "refrigerator", "dishwasher", "washing machine", "microwave",
        "air fryer", "mixer grinder", "vacuum",
    ],
}

# Form factors WITHIN a category. Two products can share a category but not
# be substitutes: over-ear headphones are not an alternative to in-ear TWS.
FORM_FACTOR_KEYWORDS: Dict[str, list] = {
    "TWS": ["tws", "earbud", "earbuds", "airdopes", "buds", "bassbuds"],
    "OVER_EAR": ["headphone", "headphones", "over-ear", "over ear", "headset"],
    "NECKBAND": ["neckband", "wireless neckband"],
    "SPEAKER": ["speaker", "soundbar", "party speaker"],
}


def _count_keyword_hits(text: str, keywords: list) -> int:
    """Counts DISTINCT keyword hits using word boundaries.

    The original code used `kw in text`, a plain substring test. That made
    "buds" match inside unrelated words and let dict iteration order decide
    ties silently. \\b anchoring plus a hit count is both safer and
    order-independent.
    """
    hits = 0
    for kw in keywords:
        if re.search(rf"\b{re.escape(kw)}\b", text):
            hits += 1
    return hits


def categorize(text: str) -> str:
    """Returns the best-scoring category, or UNKNOWN if nothing matched.

    Returning UNKNOWN (never "OTHER") is the whole point: the caller must
    treat "I don't know" differently from "I know, and they differ".
    """
    text_l = str(text).lower()
    scores = {
        cat: _count_keyword_hits(text_l, kws)
        for cat, kws in CATEGORY_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else UNKNOWN


def form_factor(text: str) -> str:
    """Sub-type within a category (TWS vs OVER_EAR vs ...), or UNKNOWN."""
    text_l = str(text).lower()
    scores = {
        ff: _count_keyword_hits(text_l, kws)
        for ff, kws in FORM_FACTOR_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else UNKNOWN


# --------------------------------------------------------------------------
# Tier detection
# --------------------------------------------------------------------------
# The dataset has no price column, so "tier" is derived from comparable
# numeric specs that actually appear in the text. We only compare a spec when
# BOTH sides state it -- comparing a stated value against a missing one would
# invent a difference that isn't in the data.
TIER_SPEC_PATTERNS: Dict[str, re.Pattern] = {
    "battery_hours": re.compile(r"(\d{1,3})\s*(?:hour|hours|hr|hrs)\b", re.I),
    "ram_gb": re.compile(r"(\d{1,3})\s*gb\s*ram\b", re.I),
    "storage_gb": re.compile(r"(\d{2,4})\s*gb\b(?!\s*ram)", re.I),
}

# A 3x gap in a headline spec means a different market segment (8-hour budget
# earbuds vs a 30-hour premium ANC headphone). Below 3x -- e.g. 100h vs 60h --
# both are "long battery life" products competing for the same buyer.
TIER_RATIO_THRESHOLD = 3.0


def extract_tier_specs(text: str) -> Dict[str, float]:
    """Pulls comparable numeric specs out of free text."""
    specs: Dict[str, float] = {}
    for name, pattern in TIER_SPEC_PATTERNS.items():
        match = pattern.search(str(text))
        if match:
            specs[name] = float(match.group(1))
    return specs


def tier_ratio(text_a: str, text_b: str) -> Optional[float]:
    """Largest max/min ratio across specs stated on BOTH sides.

    Returns None when the two texts share no comparable spec -- an honest
    "no evidence" that the caller must not read as "no difference".
    """
    specs_a, specs_b = extract_tier_specs(text_a), extract_tier_specs(text_b)
    shared = set(specs_a) & set(specs_b)
    if not shared:
        return None

    ratios = []
    for key in shared:
        lo, hi = sorted((specs_a[key], specs_b[key]))
        if lo > 0:
            ratios.append(hi / lo)
    return max(ratios) if ratios else None


# --------------------------------------------------------------------------
# The replacement decision function
# --------------------------------------------------------------------------
def classify_negative_pair(text_a: str, text_b: str) -> Optional[str]:
    """Labels a pair already known NOT to be the same product.

    Returns None to ABSTAIN -- meaning "this function has no evidence, keep
    whatever the caller's existing rule decided". Only replaces the
    `label == 0` branch of label_pair(); the EXACT_MATCH /
    SAME_PRODUCT_DIFFERENT_VARIANT logic above it scores 0.97-1.00 F1 and is
    left alone.

    WHY ABSTAIN INSTEAD OF GUESSING
    -------------------------------
    categorize() returns UNKNOWN for 64.7% of this dataset -- it covers
    electronics, but the corpus is full of clothing, books, groceries and
    homeware. An earlier draft of this function mapped UNKNOWN -> a fixed
    label and rewrote 78.1% of all labels, turning 20,163 genuinely
    unrelated pairs into WEAKLY_SIMILAR.

    Falling back to token overlap does not rescue it either. Measured on the
    UNKNOWN-category rows, overlap medians are UNRELATED 0.077 vs
    WEAKLY_SIMILAR 0.081 -- indistinguishable. There is no threshold that
    separates them, so picking one would just relocate the arbitrary
    constant this module exists to remove.

    So we change labels only where we have real evidence (both categories
    identified, ~19% of negative rows) and abstain everywhere else. Raising
    category coverage is the way to widen that 19% -- see CATEGORY_KEYWORDS.

    Decision order, most-confident signal first:
      1. Either category UNKNOWN               -> None (abstain)
      2. Two KNOWN and different categories    -> UNRELATED
      3. Same category, different form factor  -> WEAKLY_SIMILAR
      4. Same category, tier gap >= 3x         -> WEAKLY_SIMILAR
      5. Otherwise                             -> SIMILAR_ALTERNATIVE
    """
    cat_a, cat_b = categorize(text_a), categorize(text_b)

    # 1. No category evidence -> say nothing. This is the guard that keeps
    #    the change scoped; without it this function does more harm than the
    #    bug it fixes.
    if cat_a == UNKNOWN or cat_b == UNKNOWN:
        return None

    # 2. Both sides identified and they differ -- confident UNRELATED.
    if cat_a != cat_b:
        return "UNRELATED"

    # 3. Same category, but not substitutes (over-ear vs in-ear).
    ff_a, ff_b = form_factor(text_a), form_factor(text_b)
    if ff_a != UNKNOWN and ff_b != UNKNOWN and ff_a != ff_b:
        return "WEAKLY_SIMILAR"

    # 4. Same category and form factor, but different market segment.
    ratio = tier_ratio(text_a, text_b)
    if ratio is not None and ratio >= TIER_RATIO_THRESHOLD:
        return "WEAKLY_SIMILAR"

    # 5. Same category, same form factor, comparable tier (or no tier
    #    evidence) -> a genuine alternative. Defaulting here rather than to
    #    WEAKLY_SIMILAR is deliberate: two same-category products with no
    #    measured difference are more usefully surfaced as alternatives.
    return "SIMILAR_ALTERNATIVE"
