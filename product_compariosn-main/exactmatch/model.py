"""
exactmatch/model.py
====================
Loads the backbone + a plain 2-way classification head. No LoRA/PEFT
branch -- this package trains one small backbone fully, which is cheap
enough that adapters buy nothing at this scale (see root config.py's own
note: LoRA earns its place on larger backbones or multi-adapter setups,
neither of which applies here).
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import exactmatch.config as config


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    # torch_dtype=torch.float32, pinned explicitly: some checkpoints advertise
    # a non-fp32 torch_dtype in their config, which recent `transformers`
    # versions honor by default. Accelerate's fp16 mixed precision needs fp32
    # master weights -- it autocasts to fp16 only inside the forward pass and
    # unscales fp32 gradients afterward via GradScaler. A model already loaded
    # in fp16 produces fp16 gradients, and GradScaler raises "Attempting to
    # unscale FP16 gradients." Measured live: training died at step 1 with
    # exactly that error before this pin was added -- the root exact_match/
    # model.py already carries this same fix; this package's first draft
    # dropped it while simplifying and reproduced the bug it already fixed.
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        id2label={0: "different_product", 1: "same_product"},
        label2id={"different_product": 0, "same_product": 1},
        problem_type="single_label_classification",
        dtype=torch.float32,
    )
    model = model.float()
    return model, tokenizer
