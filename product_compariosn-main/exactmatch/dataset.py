"""
exactmatch/dataset.py
======================
Loads data/real_corpus_train.csv / real_corpus_valid.csv (already
COL/VAL-serialized, already leakage-checked and deduplicated by
build_real_corpus.py) and turns them into tokenized DataLoaders.

No CSV-schema auto-detection: the input has exactly four columns
(text_a, text_b, label, source) and always will, because it's written by
one script. Padding is dynamic per batch (DataCollatorWithPadding), not
padded to MAX_SEQ_LENGTH up front -- most pairs are far shorter than the
256-token cap, and padding every batch to the cap wastes most of the
compute on short e-commerce titles.
"""
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase

import exactmatch.config as config


class PairDataset(Dataset):
    def __init__(self, text_a: List[str], text_b: List[str], labels: List[int],
                 tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.text_a = list(text_a)
        self.text_b = list(text_b)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.text_a[idx], self.text_b[idx],
            truncation=True, max_length=self.max_length,
        )
        enc["labels"] = int(self.labels[idx])
        return enc


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"text_a", "text_b", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing column(s) {missing}. This package only reads "
            "the fixed schema build_real_corpus.py writes -- run that first."
        )
    df = df.dropna(subset=["text_a", "text_b", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def build_dataloaders(tokenizer: PreTrainedTokenizerBase,
                       train_csv: str = None, valid_csv: str = None
                       ) -> Tuple[DataLoader, DataLoader, dict]:
    train_csv = train_csv or config.TRAIN_CSV
    valid_csv = valid_csv or config.VALID_CSV

    train_df = load_csv(train_csv)
    valid_df = load_csv(valid_csv)

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train_ds = PairDataset(train_df.text_a, train_df.text_b, train_df.label,
                           tokenizer, config.MAX_SEQ_LENGTH)
    valid_ds = PairDataset(valid_df.text_a, valid_df.text_b, valid_df.label,
                           tokenizer, config.MAX_SEQ_LENGTH)

    train_loader = DataLoader(train_ds, batch_size=config.TRAIN_BATCH_SIZE,
                              shuffle=True, collate_fn=collator)
    valid_loader = DataLoader(valid_ds, batch_size=config.EVAL_BATCH_SIZE,
                              shuffle=False, collate_fn=collator)

    info = {
        "train_n": len(train_df), "train_pos": int(train_df.label.sum()),
        "valid_n": len(valid_df), "valid_pos": int(valid_df.label.sum()),
    }
    return train_loader, valid_loader, info
