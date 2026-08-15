"""
exactmatch/inference.py
========================
ONE decision, nothing else: is product A the exact same product as
product B. No relationship classes, no "similar alternative", no ranked
list, no generated reasons. If you want those, they live in
ranking/ranker.py on top of a comparer's score -- this module doesn't
produce them and doesn't import anything from ranking/.

    from exactmatch.inference import ExactMatcher
    m = ExactMatcher()
    result = m.compare(title_a="...", title_b="...")
    result.is_match      # bool
    result.confidence     # 0-100

CLI:
    python -m exactmatch.inference --title_a "..." --title_b "..."
"""
import argparse
import json
import os
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import exactmatch.config as config


@dataclass
class MatchResult:
    is_match: bool
    confidence: float  # 0-100, P(same product) if match else P(different product)

    def __str__(self) -> str:
        verdict = "MATCH" if self.is_match else "NO MATCH"
        return f"{verdict} ({self.confidence:.1f}% confidence)"


def _serialize(title: str, brand: str = "", description: str = "",
              description_words: int = 20) -> str:
    """COL/VAL layout, same as the corpus this model trains on. Changing
    this without retraining silently destroys recall (trap 2.3)."""
    def words(v, n):
        return " ".join(str(v or "").split(" ")[:n]).strip()
    return (f"COL brand VAL {words(brand, 5)} "
            f"COL title VAL {words(title, 50)} "
            f"COL description VAL {words(description, description_words)}").strip()


class ExactMatcher:
    """Loads a checkpoint and answers exactly one question: same product or not."""

    def __init__(self, model_dir: str = None, device: str = None, threshold: float = None):
        model_dir = model_dir or config.TRAINED_MODEL_DIR
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

        if len(self.model.config.id2label) != 2:
            raise ValueError(
                f"{model_dir} has {len(self.model.config.id2label)} labels. "
                "ExactMatcher only serves binary same/different checkpoints."
            )

        meta = self._read_metadata(model_dir)
        self.description_words = meta.get("description_words", 20)
        self.threshold = threshold if threshold is not None else meta.get(
            "inference_threshold", config.INFERENCE_THRESHOLD)

    @staticmethod
    def _read_metadata(model_dir: str) -> dict:
        path = os.path.join(model_dir, "training_metadata.json")
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _canonical(a: str, b: str):
        """Sorts the pair into a fixed order so compare(A, B) == compare(B, A).
        A cross-encoder is not a symmetric function even when the question is
        -- see this project's own measured order-sensitivity (median gap
        0.06pp, worst-case 91.8pp, 4.2% of decisions flip on argument order
        alone). Costs nothing; makes both call orders literally the same
        forward pass."""
        return (a, b) if a <= b else (b, a)

    @torch.no_grad()
    def compare(self, title_a: str, title_b: str, brand_a: str = "", brand_b: str = "",
               description_a: str = "", description_b: str = "") -> MatchResult:
        text_a = _serialize(title_a, brand_a, description_a, self.description_words)
        text_b = _serialize(title_b, brand_b, description_b, self.description_words)
        text_a, text_b = self._canonical(text_a, text_b)

        enc = self.tokenizer(text_a, text_b, truncation=True,
                            max_length=config.MAX_SEQ_LENGTH, return_tensors="pt").to(self.device)
        probs = torch.softmax(self.model(**enc).logits, dim=-1).squeeze(0)
        p_match = probs[1].item()
        is_match = p_match >= self.threshold
        confidence = p_match if is_match else (1 - p_match)
        return MatchResult(is_match=is_match, confidence=confidence * 100)

    @torch.no_grad()
    def compare_batch(self, pairs: list, batch_size: int = 32) -> list:
        """pairs: list of dicts with title_a/title_b (+ optional brand_a/b,
        description_a/b). Real batching -- one forward pass per batch."""
        if not pairs:
            return []

        texts_a, texts_b = [], []
        for p in pairs:
            ta = _serialize(p.get("title_a", ""), p.get("brand_a", ""),
                           p.get("description_a", ""), self.description_words)
            tb = _serialize(p.get("title_b", ""), p.get("brand_b", ""),
                           p.get("description_b", ""), self.description_words)
            ta, tb = self._canonical(ta, tb)
            texts_a.append(ta)
            texts_b.append(tb)

        results = []
        for start in range(0, len(pairs), batch_size):
            enc = self.tokenizer(
                texts_a[start:start + batch_size], texts_b[start:start + batch_size],
                truncation=True, max_length=config.MAX_SEQ_LENGTH, padding=True,
                return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu()
            for row in probs:
                p_match = float(row[1])
                is_match = p_match >= self.threshold
                results.append(MatchResult(
                    is_match=is_match,
                    confidence=(p_match if is_match else (1 - p_match)) * 100,
                ))
        return results


def _parse_args():
    ap = argparse.ArgumentParser(description="Exact-match check: same product or not, nothing else.")
    ap.add_argument("--title_a", required=True)
    ap.add_argument("--title_b", required=True)
    ap.add_argument("--brand_a", default="")
    ap.add_argument("--brand_b", default="")
    ap.add_argument("--description_a", default="")
    ap.add_argument("--description_b", default="")
    ap.add_argument("--model_dir", default=None)
    return ap.parse_args()


def main():
    args = _parse_args()
    matcher = ExactMatcher(model_dir=args.model_dir)
    result = matcher.compare(
        title_a=args.title_a, title_b=args.title_b,
        brand_a=args.brand_a, brand_b=args.brand_b,
        description_a=args.description_a, description_b=args.description_b,
    )
    print(result)


if __name__ == "__main__":
    main()
