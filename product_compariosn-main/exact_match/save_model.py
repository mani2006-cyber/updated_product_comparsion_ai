"""
save_model.py
=============
Steps 22-23: Save the best model + export the tokenizer in standard
Hugging Face format so `trained_model/` can be loaded anywhere with
`AutoModelForSequenceClassification.from_pretrained("trained_model")`.

Produces:
    trained_model/
        config.json
        model.safetensors           (or pytorch_model.bin)
        tokenizer.json
        tokenizer_config.json
        special_tokens_map.json
        (+ sentencepiece.bpe.model / spm.model, tokenizer-dependent)
"""

import os

from transformers import PreTrainedModel, PreTrainedTokenizerBase

import config
from utils import get_logger, save_json

logger = get_logger(__name__)


def export_trained_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str = config.TRAINED_MODEL_DIR,
    save_safetensors: bool = True,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir, safe_serialization=save_safetensors)
    tokenizer.save_pretrained(output_dir)

    save_json(
        {
            "base_model": config.MODEL_NAME,
            "task": "product_pair_classification",
            "num_labels": config.NUM_LABELS,
            "max_seq_length": config.MAX_SEQ_LENGTH,
            "label_map": {"0": "different_product", "1": "same_product"},
        },
        os.path.join(output_dir, "training_metadata.json"),
    )

    files = sorted(os.listdir(output_dir))
    logger.info(f"Exported model to {output_dir}: {files}")
    return output_dir


if __name__ == "__main__":
    # Standalone usage: re-export trained_model/ from the best checkpoint,
    # e.g. after re-running evaluate.py or if you retrained without
    # re-running the full train.py export step.
    import torch

    from model import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer()
    checkpoint = torch.load(config.BEST_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    export_trained_model(model, tokenizer)