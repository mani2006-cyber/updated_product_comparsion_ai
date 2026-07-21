"""
config.py
=========
Single source of truth for every path, hyperparameter, and switch used
across the pipeline (dataset.py, model.py, train.py, evaluate.py,
inference.py). Nothing else in the project should hardcode a path or a
hyperparameter -- import it from here instead.
"""

import os
import torch
from dataclasses import dataclass, field
from typing import List


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "products_structured.csv")  # CSV or JSON/JSONL
RESUME_FROM_TRAINED_MODEL = False
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

TRAINED_MODEL_DIR = os.path.join(ROOT_DIR, "trained_model")
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")

for _d in (DATA_DIR, OUTPUT_DIR, CHECKPOINT_DIR, PLOTS_DIR, LOGS_DIR,
           REPORTS_DIR, TRAINED_MODEL_DIR):
    os.makedirs(_d, exist_ok=True)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
# Why microsoft/deberta-v3-small:
#   - DeBERTa-v3 uses disentangled attention + ELECTRA-style pretraining,
#     which consistently beats BERT/RoBERTa of similar size on pairwise
#     sentence classification benchmarks (MNLI, STS-B, RTE) -- and product
#     matching is structurally the same task as NLI/STS: "do these two
#     spans of text refer to the same thing?".
#   - "small" variant (~140M params) trains fast enough on a single T4/
#     free-tier GPU (or even CPU for smoke tests) while still beating
#     distilbert/bert-base on accuracy per the DeBERTa-v3 paper.
#   - Ships a SentencePiece tokenizer, which handles messy e-commerce
#     tokens (model numbers, units like "128GB"/"128 GB", mixed case)
#     more gracefully than WordPiece.
# Swap-in alternatives (edit MODEL_NAME only, nothing else changes):
#   - "distilbert-base-uncased"      -> fastest, lowest accuracy ceiling
#   - "bert-base-uncased"            -> solid, well-understood baseline
#   - "microsoft/deberta-v3-base"    -> higher accuracy, ~3x slower/heavier
MODEL_NAME = "microsoft/deberta-v3-small"

NUM_LABELS = 2                # 0 = different product, 1 = same product
MAX_SEQ_LENGTH = 256          # title + specs for both products fits comfortably
PROBLEM_TYPE = "single_label_classification"

# Used when NUM_LABELS == 5 (the relationship_pairs.csv dataset). Order must
# match the id2label mapping in exact_match/model.py exactly.
RELATIONSHIP_LABEL_MAP = {
    "EXACT_MATCH": 0,
    "SAME_PRODUCT_DIFFERENT_VARIANT": 1,
    "SIMILAR_ALTERNATIVE": 2,
    "WEAKLY_SIMILAR": 3,
    "UNRELATED": 4,
}


# --------------------------------------------------------------------------
# LoRA / PEFT
# --------------------------------------------------------------------------
# LoRA is most useful when (a) the base model is large and/or (b) you plan
# to train many task-specific adapters off one frozen backbone. For a
# "small" backbone on a modest dataset, full fine-tuning is cheap and
# usually gives a small accuracy edge -- so LoRA defaults to OFF here.
# Flip USE_LORA = True to fine-tune with adapters instead (recommended if
# you later swap MODEL_NAME to a "base"/"large" model, or want to keep
# multiple category-specific adapters around a single frozen backbone).
USE_LORA = False

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES: List[str] = ["query_proj", "key_proj", "value_proj"]  # DeBERTa-v3 attention proj names
LORA_BIAS = "none"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
# The pipeline accepts either schema out of the box (see preprocessing.py):
#   A) CSV with columns: product1, product2, label
#   B) JSON/JSONL with: product1_title, product1_specs,
#                        product2_title, product2_specs, label
TEXT_COLUMNS_TITLE_ONLY = ["product1", "product2"]
TEXT_COLUMNS_WITH_SPECS = [
    "product1_title", "product1_specs", "product2_title", "product2_specs"
]
TEXT_COLUMNS_FULL = [
    "product1_title", "product1_brand", "product1_description",
    "product2_title", "product2_brand", "product2_description",
]
LABEL_COLUMN = "label"

VAL_SPLIT_RATIO = 0.15
TEST_SPLIT_RATIO = 0.15   # carved out of the data before train/val split
RANDOM_SEED = 42
STRATIFY_SPLITS = True


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
NUM_EPOCHS = 15
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2      # effective batch size = 16 * 2 = 32
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
LR_SCHEDULER_TYPE = "linear"         # linear | cosine | cosine_with_restarts
MAX_GRAD_NORM = 1.0

MIXED_PRECISION = "fp16" if torch.cuda.is_available() else "no"   # "no" | "fp16" | "bf16" (passed to Accelerate)

EARLY_STOPPING_PATIENCE = 3          # epochs with no val-loss improvement
EARLY_STOPPING_MIN_DELTA = 1e-4

LOG_EVERY_N_STEPS = 10
SAVE_EVERY_EPOCH = True              # keep a rolling "last epoch" checkpoint too

# The metric used to decide "best" checkpoint / early stopping.
PRIMARY_METRIC = "f1"                # accuracy | precision | recall | f1 | roc_auc


# --------------------------------------------------------------------------
# Inference
# --------------------------------------------------------------------------
INFERENCE_THRESHOLD = 0.5            # similarity score >= threshold -> "Same Product"


@dataclass
class RunConfig:
    """Convenience bundle if you want to pass a single object around
    instead of importing module-level constants everywhere."""
    model_name: str = MODEL_NAME
    num_labels: int = NUM_LABELS
    max_seq_length: int = MAX_SEQ_LENGTH
    use_lora: bool = USE_LORA
    num_epochs: int = NUM_EPOCHS
    train_batch_size: int = TRAIN_BATCH_SIZE
    eval_batch_size: int = EVAL_BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    seed: int = RANDOM_SEED