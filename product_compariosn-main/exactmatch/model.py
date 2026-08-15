"""
exactmatch/model.py
====================
Loads the backbone + a plain 2-way classification head. No LoRA/PEFT
branch -- this package trains one small backbone fully, which is cheap
enough that adapters buy nothing at this scale (see root config.py's own
note: LoRA earns its place on larger backbones or multi-adapter setups,
neither of which applies here).
"""
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import exactmatch.config as config


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label={0: "different_product", 1: "same_product"},
        label2id={"different_product": 0, "same_product": 1},
        problem_type="single_label_classification",
    )
    return model, tokenizer
