"""
inference.py
============
Step 24: Inference. Loads the exported `trained_model/` and compares
two products, printing a similarity score + Same/Different prediction.

Library usage:
    from inference import ProductComparer
    comparer = ProductComparer()
    result = comparer.compare(
        title_a="iPhone 15 128GB Black",
        specs_a="Apple A16\n128GB Storage\n6.1 OLED\n48MP Camera",
        title_b="Apple iPhone 15 Black 128 GB",
        specs_b="A16 Bionic\n128 GB\n6.1-inch OLED\n48 MP",
    )
    print(result)

CLI usage:
    python inference.py \\
        --title_a "iPhone 15 128GB Black" \\
        --specs_a "Apple A16, 128GB Storage, 6.1 OLED, 48MP Camera" \\
        --title_b "Apple iPhone 15 Black 128 GB" \\
        --specs_b "A16 Bionic, 128 GB, 6.1-inch OLED, 48 MP"

    # or interactively:
    python inference.py --interactive
"""

import argparse
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from .preprocessing import build_product_text
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class ComparisonResult:
    similarity_score: float          # 0-100 (%) -- confidence in the predicted class
    prediction: str                  # human-readable prediction
    label: int                       # binary compat: 1/0 same/different (for 5-class: 1 iff EXACT_MATCH)
    relationship: str = None         # 5-class label name, e.g. "SIMILAR_ALTERNATIVE" (None for binary models)
    all_probabilities: dict = None   # {label_name: prob} for every class (only set for 5-class models)

    def __str__(self) -> str:
        text = f"Similarity Score: {self.similarity_score:.1f}%\nPrediction:\n{self.prediction}"
        if self.relationship:
            text += f"\nRelationship: {self.relationship}"
        return text

def _serialize_colval(title: str, brand: str = "", description: str = "",
                      description_words: int = 100) -> str:
    """Ditto-style `COL attr VAL value` used by the WDC / ER-Magellan corpora.

    description_words defaults to 100 -- the budget every checkpoint before
    August 2026 was trained with. It is NOT read from config here on purpose:
    the budget belongs to the checkpoint, exactly like `serialization`, because
    serving a model a different budget than it trained on is a silent accuracy
    loss rather than an error (trap 2.3). ProductComparer reads it from
    training_metadata.json and passes it in.
    """
    def words(value, limit):
        return " ".join(str(value or "").split(" ")[:limit]).strip()
    return (f"COL brand VAL {words(brand, 5)} "
            f"COL title VAL {words(title, 50)} "
            f"COL description VAL {words(description, description_words)}").strip()


class ProductComparer:
    """Loads a trained checkpoint and scores product pairs.

    SERIALIZATION MUST MATCH TRAINING
    ---------------------------------
    A model only understands the text layout it was trained on. Feeding the
    wrong one does not raise; it silently destroys accuracy. Measured on this
    project's own model, three unmistakable matches scored 49.8% / 20.6% / 4.4%
    under the `title | brand x | description` layout and 100% / 99.9% / 100%
    under `COL brand VAL ... COL title VAL ...`. Recall on real matches was
    zero, while the service reported healthy.

    So the layout is read from the checkpoint's training_metadata.json
    ("serialization": "colval" | "pipeline") rather than assumed. Checkpoints
    predating that field fall back to "pipeline", which is what the original
    5-class model was trained with, and it can always be forced explicitly.
    """

    def __init__(self, model_dir: str = config.TRAINED_MODEL_DIR, device: str = None,
                 serialization: str = None, threshold: float = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading trained model from {model_dir} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.serialization = serialization or self._detect_serialization(model_dir)
        self.threshold = threshold if threshold is not None else self._detect_threshold(model_dir)
        self.description_words = self._detect_description_words(model_dir)
        logger.info(f"Input serialization: {self.serialization} | "
                    f"decision threshold: {self.threshold} | "
                    f"description budget: {self.description_words}w")

    @staticmethod
    def _detect_description_words(model_dir: str) -> int:
        """How many description words this checkpoint was trained on.

        Measured on validation: 100w F1 84.38, 50w 84.48, 20w 83.91, 0w 81.84,
        against a bootstrap SE of 0.89. So 20 words costs nothing measurable
        while cutting mean tokens 111.7 -> 84.9 and dropping the fraction of
        pairs hitting the 256 cap from 5.3% to 0.1%; dropping the description
        entirely costs 2.54 F1, which is ~2.9 SE and real.

        Absent means 100: every checkpoint trained before this was measured.
        Guessing the new default for an old model would silently change its
        input and cost accuracy without raising.
        """
        import json
        import os
        path = os.path.join(model_dir, "training_metadata.json")
        try:
            with open(path, encoding="utf-8") as fh:
                value = json.load(fh).get("description_words")
            if value is not None and int(value) >= 0:
                return int(value)
        except Exception:  # noqa: BLE001
            pass
        return 100

    @staticmethod
    def _detect_threshold(model_dir: str) -> float:
        """Reads the decision threshold recorded by calibrate_threshold.py.

        0.5 is only optimal if the training positive rate happens to match the
        serving one. It did not: moving from the 38k corpus (26.8% positive) to
        the 78k LSPC corpus (20.6%) shifted the probability distribution far
        enough that the best cut-point on WDC-UNSEEN was 0.94, and scoring at
        0.5 cost 1.8 F1 there while precision fell 0.669 -> 0.612. The
        threshold belongs to the checkpoint, not to a global constant, for the
        same reason `serialization` does -- a mismatch does not raise, it just
        quietly makes the model over-predict matches.
        """
        import json
        import os
        path = os.path.join(model_dir, "training_metadata.json")
        try:
            with open(path, encoding="utf-8") as fh:
                value = json.load(fh).get("inference_threshold")
            if value is not None and 0.0 < float(value) < 1.0:
                return float(value)
        except Exception:  # noqa: BLE001
            pass
        logger.warning(
            "No 'inference_threshold' recorded in %s; falling back to "
            "config.INFERENCE_THRESHOLD=%s. Run calibrate_threshold.py --write to fit one.",
            path, config.INFERENCE_THRESHOLD)
        return float(config.INFERENCE_THRESHOLD)

    @staticmethod
    def _detect_serialization(model_dir: str) -> str:
        import json
        import os
        path = os.path.join(model_dir, "training_metadata.json")
        try:
            with open(path, encoding="utf-8") as fh:
                value = json.load(fh).get("serialization")
            if value in ("colval", "pipeline"):
                return value
        except Exception:  # noqa: BLE001
            pass
        logger.warning(
            "No 'serialization' recorded in %s; assuming 'pipeline'. If this model was "
            "trained on the WDC/ER-Magellan corpora it needs 'colval' -- the wrong choice "
            "silently collapses match recall.", path)
        return "pipeline"

    @staticmethod
    def _canonical(text_a: str, text_b: str):
        """Put a pair into a fixed order so scoring is order-invariant.

        "Is A the same product as B" is a symmetric question, but a
        cross-encoder is not a symmetric function: [CLS] a [SEP] b [SEP] and
        [CLS] b [SEP] a [SEP] are different inputs with different token_type_ids.
        Nothing in training pushed back on that -- build_real_corpus.py dedups
        with tuple(sorted(...)), so every pair appears in exactly ONE order and
        the model never saw the mirror.

        Measured on the 190-pair Indian set: median difference between the two
        orders is 0.06 pp, but p95 is 35.9 pp, the worst case is 91.8 pp, and
        8 pairs (4.2%) cross the decision threshold -- the same answer flips
        depending on which product the caller happens to put first.

        Sorting the serialized texts costs nothing and makes the two calls
        literally the same forward pass. It does not make the model better --
        forward and reversed score within noise of each other (F1 80.85 vs
        81.75) -- it makes it CONSISTENT. Averaging both orders would also work
        but doubles inference, and throughput is already the bottleneck.
        """
        return (text_a, text_b) if text_a <= text_b else (text_b, text_a)

    def _text(self, title: str, brand: str = "", specs: str = "", description: str = "") -> str:
        if self.serialization == "colval":
            return _serialize_colval(title, brand, specs or description,
                                     description_words=self.description_words)
        return build_product_text(title, brand=brand, specs=specs, description=description)

    @torch.no_grad()
    def compare(
        self,
        title_a: str,
        title_b: str,
        brand_a: str = "",
        brand_b: str = "",
        specs_a: str = "",
        specs_b: str = "",
        description_a: str = "",
        description_b: str = "",
        threshold: float = None,
    ) -> ComparisonResult:
        # Resolved per call, not bound as a default argument. Default values are
        # evaluated once at import, which is exactly how config.NUM_LABELS
        # silently produced 2-class models when set after the import (trap 6.1).
        threshold = self.threshold if threshold is None else threshold
        text_a = self._text(title_a, brand_a, specs_a, description_a)
        text_b = self._text(title_b, brand_b, specs_b, description_b)
        text_a, text_b = self._canonical(text_a, text_b)

        encoding = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**encoding).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        num_labels = len(id2label)

        if num_labels == 2:
            same_prob = probs[1].item()
            label = int(same_prob >= threshold)
            prediction = "Same Product" if label == 1 else "Different Product"
            return ComparisonResult(similarity_score=same_prob * 100, prediction=prediction, label=label)

        predicted_id = int(torch.argmax(probs).item())
        relationship = id2label[predicted_id]
        confidence = probs[predicted_id].item()
        all_probabilities = {id2label[i]: float(probs[i].item()) for i in range(num_labels)}
        label = int(relationship == "EXACT_MATCH")  # binary-compat flag for old callers
        prediction = relationship.replace("_", " ").title()

        return ComparisonResult(
            similarity_score=confidence * 100,
            prediction=prediction,
            label=label,
            relationship=relationship,
            all_probabilities=all_probabilities,
        )

    @torch.no_grad()
    def score_pairs(self, pairs, batch_size: int = 32,
                    threshold: float = None):
        """Scores many pairs with REAL batching -- one forward pass per batch.

        `pairs`: list of dicts accepting the same keys as compare()
                 (title_a/brand_a/description_a/specs_a and the _b twins).

        This exists because the previous compare_batch() simply looped over
        compare(), doing one tokenisation and one forward pass per pair. The
        ranker calls this once per API request over a shortlist of up to 50
        candidates, so that loop was 50 sequential GPU round-trips per request
        -- the dominant source of latency in the service.

        Padding is `longest` per batch rather than `max_length`: compare()
        pads every input to 256 tokens regardless of content, which wastes
        most of the compute on short e-commerce titles.
        """
        if not pairs:
            return []

        threshold = self.threshold if threshold is None else threshold
        texts_a = [self._text(p.get("title_a", ""), p.get("brand_a", ""),
                              p.get("specs_a", ""), p.get("description_a", "")) for p in pairs]
        texts_b = [self._text(p.get("title_b", ""), p.get("brand_b", ""),
                              p.get("specs_b", ""), p.get("description_b", "")) for p in pairs]
        # Canonicalise each pair so a caller swapping the two products gets the
        # same score. Only the model INPUT is reordered -- results stay aligned
        # with `pairs`, so callers still get their own ordering back.
        canon = [self._canonical(a, b) for a, b in zip(texts_a, texts_b)]
        texts_a = [c[0] for c in canon]
        texts_b = [c[1] for c in canon]

        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        num_labels = len(id2label)
        results = []

        for start in range(0, len(pairs), batch_size):
            encoding = self.tokenizer(
                texts_a[start:start + batch_size],
                texts_b[start:start + batch_size],
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            probs = torch.softmax(self.model(**encoding).logits, dim=-1).cpu()

            for row in probs:
                if num_labels == 2:
                    same_prob = float(row[1])
                    label = int(same_prob >= threshold)
                    results.append(ComparisonResult(
                        similarity_score=same_prob * 100,
                        prediction="Same Product" if label else "Different Product",
                        label=label,
                    ))
                else:
                    top = int(torch.argmax(row).item())
                    relationship = id2label[top]
                    results.append(ComparisonResult(
                        similarity_score=float(row[top]) * 100,
                        prediction=relationship.replace("_", " ").title(),
                        label=int(relationship == "EXACT_MATCH"),
                        relationship=relationship,
                        all_probabilities={id2label[i]: float(row[i]) for i in range(num_labels)},
                    ))
        return results

    @torch.no_grad()
    def compare_batch(self, pairs):
        """Backwards-compatible alias; now genuinely batched via score_pairs()."""
        return self.score_pairs(pairs)


def _parse_args():
    parser = argparse.ArgumentParser(description="Compare two products with the fine-tuned model.")
    parser.add_argument("--title_a", type=str, default=None)
    parser.add_argument("--brand_a", type=str, default="")
    parser.add_argument("--specs_a", type=str, default="")
    parser.add_argument("--description_a", type=str, default="")
    parser.add_argument("--title_b", type=str, default=None)
    parser.add_argument("--brand_b", type=str, default="")
    parser.add_argument("--specs_b", type=str, default="")
    parser.add_argument("--description_b", type=str, default="")
    parser.add_argument("--model_dir", type=str, default=config.TRAINED_MODEL_DIR)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    comparer = ProductComparer(model_dir=args.model_dir)

    if args.interactive or not (args.title_a and args.title_b):
        print("=== Product Comparison (interactive mode) ===")
        title_a = input("Product A - Title: ")
        brand_a = input("Product A - Brand (optional): ")
        specs_a = input("Product A - Specs/Description (optional): ")
        title_b = input("Product B - Title: ")
        brand_b = input("Product B - Brand (optional): ")
        specs_b = input("Product B - Specs/Description (optional): ")
    else:
        title_a, brand_a, specs_a = args.title_a, args.brand_a, (args.specs_a or args.description_a)
        title_b, brand_b, specs_b = args.title_b, args.brand_b, (args.specs_b or args.description_b)

    result = comparer.compare(
        title_a=title_a, brand_a=brand_a, specs_a=specs_a,
        title_b=title_b, brand_b=brand_b, specs_b=specs_b,
    )

    print("\nProduct A:", title_a)
    if brand_a:
        print("Brand:", brand_a)
    if specs_a:
        print("Specs/Description:", specs_a)
    print("\nProduct B:", title_b)
    if brand_b:
        print("Brand:", brand_b)
    if specs_b:
        print("Specs/Description:", specs_b)
    print("\nOutput:")
    print(result)


if __name__ == "__main__":
    main()