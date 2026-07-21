"""
data_quality/contradiction_rules.py
====================================
Rule definitions for detecting contradictory or impossible specs inside a
SINGLE product's title + description (e.g. a description that mentions both
"Core i7" and "Ryzen 5 5600H", or two different RAM values).

Each rule is a function: (text: str) -> Optional[Issue]
Add new rules by writing a function and appending it to ALL_RULES.
"""

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional


@dataclass
class Issue:
    code: str                 # short machine-readable identifier
    severity: str              # "high" | "medium" | "low"
    message: str                 # human-readable explanation
    evidence: List[str] = field(default_factory=list)  # the conflicting matches found


# --------------------------------------------------------------------------
# Known value families used for conflict detection
# --------------------------------------------------------------------------
CPU_FAMILIES = {
    "intel_core": r"\bcore\s*i[3579]\b",
    "amd_ryzen": r"\bryzen\s*[3579]\b",
    "apple_silicon": r"\bm[1-4]\s*(pro|max|ultra)?\b",
    "snapdragon": r"\bsnapdragon\s*\d{3,4}\b",
    "mediatek": r"\b(mediatek|dimensity)\s*\d{3,4}\b",
}

BLUETOOTH_VERSION_RE = r"\bbluetooth\s*v?(\d\.\d)\b"
RAM_RE = r"\b(\d{1,3})\s*gb\s*ram\b"
STORAGE_RE = r"\b(\d{1,4})\s*(gb|tb)\s*(ssd|hdd|storage)?\b"
VARIANT_KEYWORDS = re.compile(r"\b(options?|variants?|available in|choose|configurations?)\b", re.I)


def _distinct_matches(pattern: str, text: str) -> List[str]:
    return sorted(set(m.group(0) for m in re.finditer(pattern, text, re.I)))


def rule_cpu_family_conflict(text: str) -> Optional[Issue]:
    found = {}
    for family, pattern in CPU_FAMILIES.items():
        matches = _distinct_matches(pattern, text)
        if matches:
            found[family] = matches
    if len(found) >= 2:
        evidence = [f"{fam}: {vals}" for fam, vals in found.items()]
        return Issue(
            code="cpu_family_conflict",
            severity="high",
            message="Description mentions processors from more than one family/vendor.",
            evidence=evidence,
        )
    return None


def rule_ram_conflict(text: str) -> Optional[Issue]:
    matches = _distinct_matches(RAM_RE, text)
    if len(matches) >= 2:
        return Issue(
            code="ram_conflict",
            severity="high",
            message="Multiple distinct RAM values found in the same listing.",
            evidence=matches,
        )
    return None


def rule_bluetooth_version_conflict(text: str) -> Optional[Issue]:
    matches = _distinct_matches(BLUETOOTH_VERSION_RE, text)
    if len(matches) >= 2:
        return Issue(
            code="bluetooth_version_conflict",
            severity="medium",
            message="Multiple distinct Bluetooth versions found in the same listing.",
            evidence=matches,
        )
    return None


def rule_storage_conflict(text: str) -> Optional[Issue]:
    # Skip if the text explicitly signals multiple purchasable variants
    # ("available in 128GB/256GB") -- that's a legitimate multi-SKU listing,
    # not a contradiction.
    if VARIANT_KEYWORDS.search(text):
        return None
    matches = _distinct_matches(STORAGE_RE, text)
    if len(matches) >= 3:
        return Issue(
            code="storage_conflict",
            severity="medium",
            message="Three or more distinct storage values found without variant language.",
            evidence=matches,
        )
    return None


def rule_brand_mismatch(text: str, declared_brand: Optional[str]) -> Optional[Issue]:
    if not declared_brand:
        return None
    if declared_brand.strip().lower() not in text.lower():
        return Issue(
            code="brand_mismatch",
            severity="low",
            message=f"Declared brand '{declared_brand}' does not appear in title/description text.",
            evidence=[declared_brand],
        )
    return None


# Rules that only need the combined text
TEXT_ONLY_RULES: List[Callable[[str], Optional[Issue]]] = [
    rule_cpu_family_conflict,
    rule_ram_conflict,
    rule_bluetooth_version_conflict,
    rule_storage_conflict,
]

# Rules that also need a declared_brand argument
BRAND_RULES: List[Callable[[str, Optional[str]], Optional[Issue]]] = [
    rule_brand_mismatch,
]