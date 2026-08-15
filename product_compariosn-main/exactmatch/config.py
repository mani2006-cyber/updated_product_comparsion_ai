"""
exactmatch/config.py
=====================
Every path / hyperparameter for the exactmatch pipeline, in one place --
same rule as the root config.py, deliberately NOT shared with it. This
package trains and serves exactly one thing: binary same-product /
different-product. No LoRA switch, no 5-class relationship map, no
title-only-vs-specs schema detection. If a knob doesn't affect that one
decision, it isn't here.

Deliberately reuses the SAME real, human-labelled corpus (WDC Products
2023 + ER-Magellan + hard-negative-mined LSPC) that trained v11 --
data/real_corpus_train.csv / real_corpus_valid.csv, built by
build_real_corpus.py + mine_hard_negatives.py at the repo root, unchanged.
This package does not re-derive or regenerate that data; it consumes it.
"""

import os

import torch

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Data -- consumes the corpus already built by build_real_corpus.py /
# mine_hard_negatives.py. This package does not build it.
# ---------------------------------------------------------------------------
TRAIN_CSV = os.path.join(ROOT_DIR, "data", "real_corpus_train.csv")
VALID_CSV = os.path.join(ROOT_DIR, "data", "real_corpus_valid.csv")
ER_DIR = os.path.join(ROOT_DIR, "data", "er_magellan")

OUTPUT_DIR = os.path.join(PACKAGE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
TRAINED_MODEL_DIR = os.path.join(PACKAGE_DIR, "trained_model")

for _d in (OUTPUT_DIR, CHECKPOINT_DIR, PLOTS_DIR, LOGS_DIR, TRAINED_MODEL_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Model -- same backbone reasoning as the root config.py: deberta-v3-small's
# disentangled attention + ELECTRA pretraining beats BERT/RoBERTa of similar
# size on pairwise sentence classification, and its SentencePiece tokenizer
# handles e-commerce tokens ("128GB" vs "128 GB") better than WordPiece.
# ---------------------------------------------------------------------------
MODEL_NAME = "microsoft/deberta-v3-small"
NUM_LABELS = 2  # 0 = different_product, 1 = same_product. Nothing else.
MAX_SEQ_LENGTH = 256

# COL/VAL serialization -- same layout the corpus was built with. A
# mismatch here does not raise, it silently collapses match recall (this
# project's own trap 2.3). Kept fixed, not configurable, on purpose.
SERIALIZATION = "colval"
DESCRIPTION_WORDS = 20  # see root config.py DESCRIPTION_WORDS for the measurement

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
NUM_EPOCHS = 15
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2  # effective batch = 32
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "linear"
MAX_GRAD_NORM = 1.0
MIXED_PRECISION = "fp16" if torch.cuda.is_available() else "no"

EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4

# RANDOM_SEED reads $SEED for the same reason the root config.py does:
# run-to-run variance on this task is ~2 benchmark F1 / ~3.5 Indian F1 on
# identical inputs (measured on v10 vs v11, same recipe, same seed value,
# different result -- cuDNN autotuning and dataloader ordering are
# nondeterministic on GPU even when the seed is fixed). Comparing recipes
# needs multiple seeds; this makes that cheap.
RANDOM_SEED = int(os.environ.get("SEED", 42))

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
INFERENCE_THRESHOLD = 0.50
