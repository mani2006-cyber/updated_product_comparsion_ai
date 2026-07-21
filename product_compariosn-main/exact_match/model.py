"""
model.py
========
Step 7: Model loading. Loads a pretrained Hugging Face sequence
classification head on top of the chosen backbone, and optionally
wraps it with a LoRA adapter (PEFT) per config.USE_LORA.
"""

from typing import Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

import config
from utils import get_logger

logger = get_logger(__name__)


def load_tokenizer(model_name: str = config.MODEL_NAME) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return tokenizer


def load_model(
    model_name: str = config.MODEL_NAME,
    num_labels: int = config.NUM_LABELS,
    use_lora: bool = config.USE_LORA,
) -> PreTrainedModel:
    if num_labels == 5:
        id2label = {
            0: "EXACT_MATCH",
            1: "SAME_PRODUCT_DIFFERENT_VARIANT",
            2: "SIMILAR_ALTERNATIVE",
            3: "WEAKLY_SIMILAR",
            4: "UNRELATED",
        }
    else:
        id2label = {0: "different_product", 1: "same_product"}
    label2id = {v: k for k, v in id2label.items()}

    model_config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type=config.PROBLEM_TYPE,
        id2label=id2label,
        label2id=label2id,
    )
    # Explicitly pin fp32: some checkpoints advertise a non-fp32 torch_dtype
    # in their config, which recent `transformers` versions will honor by
    # default. Accelerate's fp16 mixed-precision mode requires fp32 master
    # weights (it autocasts to fp16 only inside the forward pass and uses
    # GradScaler to unscale fp32 gradients afterward) -- loading the model
    # already in fp16 produces fp16 gradients and GradScaler raises
    # "Attempting to unscale FP16 gradients." Forcing fp32 here avoids that
    # regardless of what the checkpoint's config declares.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_config, torch_dtype=torch.float32
    )
    model = model.float()

    if use_lora:
        model = _apply_lora(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Loaded {model_name} | total params: {total_params:,} | "
        f"trainable params: {trainable_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )
    return model


def _apply_lora(model: PreTrainedModel) -> PreTrainedModel:
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias=config.LORA_BIAS,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def load_model_and_tokenizer(
    model_name: str = config.MODEL_NAME,
    num_labels: int = config.NUM_LABELS,
    use_lora: bool = config.USE_LORA,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    tokenizer = load_tokenizer(model_name)
    model = load_model(model_name, num_labels=num_labels, use_lora=use_lora)
    return model, tokenizer