"""
dataset.py
==========
Step 3 (tokenization), Step 4 (preprocessing into tensors), and
Step 6 (DataLoader creation).

The two product texts are fed to the tokenizer as a *pair*
(text_a, text_b). For DeBERTa/BERT-style models this builds a single
sequence: [CLS] text_a [SEP] text_b [SEP], with token_type_ids marking
which side each token came from -- exactly the input format these
models were pretrained on for sentence-pair tasks (NLI, STS), which is
why it transfers so well to "are these two products the same?".
"""

import os
from typing import Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

import config


def _default_num_workers() -> int:
    return min(4, os.cpu_count() or 1)


class ProductPairDataset(Dataset):
    """Wraps a cleaned DataFrame (text_a, text_b, label) as a torch Dataset."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = config.MAX_SEQ_LENGTH,
    ):
        self.text_a: List[str] = df["text_a"].tolist()
        self.text_b: List[str] = df["text_b"].tolist()
        self.labels: List[int] = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        # Deliberately NOT padded here. Padding happens per batch in the
        # collator, so a batch is only as wide as its longest pair instead of a
        # flat MAX_SEQ_LENGTH. Product pairs average well under the cap, so most
        # of the fixed-width tensor was padding the model computed and discarded.
        # Truncation still applies -- max_length is the ceiling, not the target.
        encoding = self.tokenizer(
            self.text_a[idx],
            self.text_b[idx],
            truncation=True,
            max_length=self.max_length,
        )
        item = dict(encoding)
        item["labels"] = self.labels[idx]
        return item


def build_dataloader(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    shuffle: bool,
    max_length: int = config.MAX_SEQ_LENGTH,
    num_workers: Optional[int] = None,
) -> DataLoader:
    dataset = ProductPairDataset(df, tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=_default_num_workers() if num_workers is None else num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt"),
        drop_last=False,
    )


def build_all_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    tokenizer: PreTrainedTokenizerBase,
):
    """Convenience helper returning (train_loader, val_loader, test_loader)."""
    train_loader = build_dataloader(
        train_df, tokenizer, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True
    )
    val_loader = build_dataloader(
        val_df, tokenizer, batch_size=config.EVAL_BATCH_SIZE, shuffle=False
    )
    test_loader = None
    if test_df is not None and len(test_df) > 0:
        test_loader = build_dataloader(
            test_df, tokenizer, batch_size=config.EVAL_BATCH_SIZE, shuffle=False
        )
    return train_loader, val_loader, test_loader
