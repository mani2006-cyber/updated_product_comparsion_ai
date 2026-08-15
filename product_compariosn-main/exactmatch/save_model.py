"""
exactmatch/save_model.py
=========================
Exports the best checkpoint to exactmatch/trained_model/ in plain
HF format, plus training_metadata.json recording exactly what this
checkpoint needs to be served correctly -- serialization layout,
description-word budget, and corpus composition. Same reasoning as the
root save_model.py: these belong to the CHECKPOINT, not to a global
config constant, because serving a model a different budget/layout than
it trained on does not raise, it silently costs accuracy (this project's
own trap 2.3).
"""
import json
import os

import exactmatch.config as config


def export_trained_model(model, tokenizer, corpus_info: dict) -> None:
    os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)
    model.save_pretrained(config.TRAINED_MODEL_DIR)
    tokenizer.save_pretrained(config.TRAINED_MODEL_DIR)

    metadata = {
        "base_model": config.MODEL_NAME,
        "task": "exact_match_binary",
        "num_labels": config.NUM_LABELS,
        "max_seq_length": config.MAX_SEQ_LENGTH,
        "label_map": {"0": "different_product", "1": "same_product"},
        "serialization": config.SERIALIZATION,
        "description_words": config.DESCRIPTION_WORDS,
        "trained_on": corpus_info,
        "inference_threshold": config.INFERENCE_THRESHOLD,
        "threshold_fitted_on": "not calibrated -- run calibrate_threshold.py --write "
                              f"--model {config.TRAINED_MODEL_DIR}",
    }
    with open(os.path.join(config.TRAINED_MODEL_DIR, "training_metadata.json"),
             "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Exported model to {config.TRAINED_MODEL_DIR}: {os.listdir(config.TRAINED_MODEL_DIR)}")
