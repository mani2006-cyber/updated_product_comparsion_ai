"""
exactmatch/llm_arbiter.py
==========================
Escalates the hard near-miss slice to an actual LLM judgment instead of
trusting the small cross-encoder.

WHY THIS EXISTS
---------------
The cross-encoder's hard-slice weakness (model-suffix near-misses: "VS104"
vs "VS104 Max", "141" vs "141 Pro") is not a LOW-CONFIDENCE problem, it's a
WRONG-AND-CONFIDENT one. Live-measured this session: "Noise Buds VS104" vs
"VS104 Max" scored 99.8% MATCH from deberta-v3-small -- and every checkpoint
this project has trained (v7 through this one) makes the same class of
error at similarly high confidence. This has capped hard-slice precision at
12-17% across seven checkpoints; it is architectural, not a training-recipe
problem -- a 142M-parameter classifier pattern-matches text, it does not
reason about what a "Max" suffix or a third-party "Compatible with X"
listing implies the way an LLM does.

A confidence-band escalation ("only ask when the cross-encoder is unsure")
would never even see this failure, because the cross-encoder isn't unsure.
So escalation here is triggered STRUCTURALLY -- does this pair look like a
near-miss shape (near-identical title, one side carrying a qualifier the
other doesn't) -- not by confidence alone. A generic mid-confidence band is
kept too, as a second, independent trigger for ordinary ambiguous cases.

Nothing here replaces the cross-encoder for the bulk of traffic: only pairs
that trip a trigger pay the extra LLM latency/cost. Everything else is
scored exactly as before.
"""
import os
import re
from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel, Field

from exactmatch.inference import ExactMatcher, MatchResult

# ---------------------------------------------------------------------------
# Trigger 1: structural near-miss shape
# ---------------------------------------------------------------------------

_STOPWORDS = {"the", "a", "an", "with", "for", "of", "in", "and", "or", "by"}


def _tokens(title: str) -> set:
    """Word tokens, PLUS the letter/digit sub-parts of each token, so
    "215TWS" also contributes "215" and "tws" and matches against a title
    where the same model number appears with a space or punctuation
    boundary instead ("215 TWS", "215-TWS"). Real listing titles tokenize
    the same model number inconsistently across merchants; without this,
    bag-of-words overlap silently undercounts identical model numbers as
    different tokens.
    """
    words = re.findall(r"[a-z0-9]+", str(title or "").lower())
    toks = set()
    for w in words:
        if w in _STOPWORDS:
            continue
        toks.add(w)
        for part in re.findall(r"[a-z]+|\d+", w):
            if len(part) > 1 and part not in _STOPWORDS:
                toks.add(part)
    return toks


def _digit_runs(title: str) -> set:
    """Numeric substrings of 2+ digits -- the part of a model code that
    actually identifies the product ("VS104" -> "104", "215TWS" -> "215",
    "Gen 2" contributes nothing since "2" is one digit and too generic to
    be a meaningful signal on its own)."""
    return set(re.findall(r"\d{2,}", str(title or "")))


def is_near_miss_candidate(title_a: str, title_b: str, exempt_overlap: float = 0.85) -> bool:
    """True when two titles share a model number but aren't a near-total
    textual match -- exactly the shape where a differing suffix ("Max",
    "Gen 2"), form factor ("TWS" vs neckband), or third-party knockoff is
    easy to miss. Measured live this session: every one of these patterns
    scored 97-100% MATCH from deberta-v3-small when it should not have.

    A shape check, not a confidence check -- the cross-encoder isn't
    uncertain on these pairs, it's confidently wrong, so a confidence-band
    trigger alone would never catch them (see is_low_confidence below,
    kept as an independent second trigger for genuinely ambiguous pairs).

    exempt_overlap: above this bag-of-words overlap, treat the pair as a
    plain wording paraphrase of the identical listing (which the
    cross-encoder already handles well) rather than a near-miss worth an
    LLM call. Deliberately loose -- over-triggering only costs an extra
    LLM round trip on an easy pair; under-triggering reproduces the actual
    accuracy bug this module exists to fix, so the two failure costs are
    not symmetric and the threshold is tuned toward escalating too much
    rather than too little.
    """
    nums_a, nums_b = _digit_runs(title_a), _digit_runs(title_b)
    shared = nums_a & nums_b
    if not shared:
        # Catch "215" vs "215tws" (one run is a prefix of the other inside
        # a longer alnum token) even when the exact strings don't match.
        shared = {na for na in nums_a for nb in nums_b if na in nb or nb in na}
    if not shared:
        return False

    ta, tb = _tokens(title_a), _tokens(title_b)
    if not ta or not tb:
        return False
    overlap = len(ta & tb) / max(len(ta | tb), 1)
    return overlap < exempt_overlap


# ---------------------------------------------------------------------------
# Trigger 2: generic mid-confidence band (catches ordinary ambiguous cases
# the structural check above doesn't cover)
# ---------------------------------------------------------------------------

DEFAULT_LOW = 0.15
DEFAULT_HIGH = 0.85


def is_low_confidence(same_prob: float, low: float = DEFAULT_LOW,
                      high: float = DEFAULT_HIGH) -> bool:
    return low < same_prob < high


# ---------------------------------------------------------------------------
# The LLM call
# ---------------------------------------------------------------------------

class ArbiterVerdict(BaseModel):
    is_match: bool = Field(description="True if the two listings are the exact same product.")
    confidence: int = Field(ge=0, le=100, description="0-100 confidence in the verdict.")
    reasoning: str = Field(description="One or two sentences: the specific evidence that decided it.")


_PROMPT_TEMPLATE = """Are these two e-commerce listings the EXACT same product, or different products \
(different model/variant, or a third-party knockoff trading on a brand name)?

Product A:
  Title: {title_a}
  Brand: {brand_a}
  Description: {description_a}

Product B:
  Title: {title_b}
  Brand: {brand_b}
  Description: {description_b}

Pay close attention to:
- Model-suffix words (Max, Pro, Plus, Gen 2, etc.) present on only one side -- usually a different SKU, not a paraphrase.
- Third-party or "Compatible with X" / "For X" listings that trade on a brand name without being the genuine product.
- Spec differences (capacity, wattage, size) that look similar but aren't identical.
- Genuine paraphrases of the SAME product (different merchant wording, unit formatting like "128GB" vs "128 GB") -- these ARE matches."""


@dataclass
class ArbiterResult(MatchResult):
    escalated: bool = False
    reasoning: Optional[str] = None
    cross_encoder_confidence: Optional[float] = None

    def __str__(self) -> str:
        base = super().__str__()
        if self.escalated:
            return f"{base} [LLM-arbitrated: {self.reasoning}]"
        return base


class ArbiterMatcher:
    """Wraps ExactMatcher with an LLM escalation path for the hard slice.

    Falls back to the cross-encoder's own verdict whenever the LLM call
    can't be made (no credentials, network error, refusal) -- an
    escalation failure must never take the whole pipeline down.
    """

    def __init__(self, model_dir: str = None, llm_model: str = "claude-opus-5",
                low_conf: float = DEFAULT_LOW, high_conf: float = DEFAULT_HIGH):
        self.matcher = ExactMatcher(model_dir=model_dir)
        self.llm_model = llm_model
        self.low_conf = low_conf
        self.high_conf = high_conf
        self._client = None  # lazy -- don't require credentials just to import this module

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()  # resolves credentials from env / ant profile
        return self._client

    def _should_escalate(self, title_a: str, title_b: str, same_prob: float) -> bool:
        return (is_near_miss_candidate(title_a, title_b)
                or is_low_confidence(same_prob, self.low_conf, self.high_conf))

    def _ask_llm(self, title_a: str, brand_a: str, description_a: str,
                title_b: str, brand_b: str, description_b: str) -> Optional[ArbiterVerdict]:
        try:
            client = self._get_client()
            prompt = _PROMPT_TEMPLATE.format(
                title_a=title_a, brand_a=brand_a or "(not given)",
                description_a=description_a or "(not given)",
                title_b=title_b, brand_b=brand_b or "(not given)",
                description_b=description_b or "(not given)",
            )
            response = client.messages.parse(
                model=self.llm_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
                output_format=ArbiterVerdict,
            )
            if getattr(response, "stop_reason", None) == "refusal":
                return None
            return response.parsed_output
        except Exception:  # noqa: BLE001 -- any failure here must degrade, not crash
            import logging
            logging.getLogger("exactmatch.llm_arbiter").exception(
                "LLM arbiter call failed; falling back to cross-encoder verdict")
            return None

    def compare(self, title_a: str, title_b: str, brand_a: str = "", brand_b: str = "",
               description_a: str = "", description_b: str = "") -> ArbiterResult:
        base = self.matcher.compare(
            title_a=title_a, title_b=title_b, brand_a=brand_a, brand_b=brand_b,
            description_a=description_a, description_b=description_b,
        )
        same_prob = (base.confidence / 100) if base.is_match else (1 - base.confidence / 100)

        if not self._should_escalate(title_a, title_b, same_prob):
            return ArbiterResult(is_match=base.is_match, confidence=base.confidence,
                                 escalated=False, cross_encoder_confidence=base.confidence)

        verdict = self._ask_llm(title_a, brand_a, description_a, title_b, brand_b, description_b)
        if verdict is None:
            return ArbiterResult(is_match=base.is_match, confidence=base.confidence,
                                 escalated=False, cross_encoder_confidence=base.confidence)

        return ArbiterResult(
            is_match=verdict.is_match, confidence=float(verdict.confidence),
            escalated=True, reasoning=verdict.reasoning,
            cross_encoder_confidence=base.confidence,
        )

    def compare_batch(self, pairs: List[dict]) -> List[ArbiterResult]:
        """Same shape as ExactMatcher.compare_batch, but per-pair -- LLM
        escalation isn't batchable the way the cross-encoder is, so this
        costs one extra round trip per escalated pair, not per pair overall.
        """
        return [self.compare(**p) for p in pairs]
