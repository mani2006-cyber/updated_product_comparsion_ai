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

class ProductComparer:
    def __init__(self, model_dir: str = config.TRAINED_MODEL_DIR, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading trained model from {model_dir} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

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
        threshold: float = config.INFERENCE_THRESHOLD,
    ) -> ComparisonResult:
        text_a = build_product_text(title_a, brand=brand_a, specs=specs_a, description=description_a)
        text_b = build_product_text(title_b, brand=brand_b, specs=specs_b, description=description_b)

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
    def compare_batch(self, pairs):
        """pairs: list of dicts with title_a/specs_a/title_b/specs_b."""
        return [
            self.compare(
                title_a=p["title_a"],
                title_b=p["title_b"],
                specs_a=p.get("specs_a", ""),
                specs_b=p.get("specs_b", ""),
            )
            for p in pairs
        ]


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