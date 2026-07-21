"""
metrics.py
==========
Steps 17-21: accuracy, precision, recall, F1, ROC-AUC, plus confusion
matrix / classification report / training curve plotting used by
train.py and evaluate.py.
"""

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for servers/notebooks
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config


def compute_metrics(labels: List[int], preds: List[int], probs: Optional[List] = None) -> Dict[str, float]:
    """
    labels: ground truth class indices
    preds:  predicted class indices
    probs:  predicted probability distribution per sample -- a list of
            per-class probabilities (needed for ROC-AUC). For binary this
            is [[p0, p1], ...]; for 5-class [[p0..p4], ...].
    """
    num_classes = config.NUM_LABELS
    average = "binary" if num_classes == 2 else "macro"

    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, average=average, zero_division=0)),
        "recall": float(recall_score(labels, preds, average=average, zero_division=0)),
        "f1": float(f1_score(labels, preds, average=average, zero_division=0)),
    }

    if probs is not None and len(set(labels)) > 1:
        try:
            probs_array = np.array(probs)
            if num_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(labels, probs_array[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(labels, probs_array, multi_class="ovr", average="macro")
                )
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


def get_classification_report(labels: List[int], preds: List[int]) -> str:
    if config.NUM_LABELS == 5:
        target_names = [name for name, _ in sorted(config.RELATIONSHIP_LABEL_MAP.items(), key=lambda kv: kv[1])]
    else:
        target_names = ["different_product", "same_product"]
    return classification_report(labels, preds, target_names=target_names, zero_division=0)


def save_confusion_matrix(
    labels: List[int],
    preds: List[int],
    save_path: str = os.path.join(config.PLOTS_DIR, "confusion_matrix.png"),
) -> str:
    cm = confusion_matrix(labels, preds)
    if config.NUM_LABELS == 5:
        tick_labels = [name for name, _ in sorted(config.RELATIONSHIP_LABEL_MAP.items(), key=lambda kv: kv[1])]
        figsize = (7, 6)
    else:
        tick_labels = ["different", "same"]
        figsize = (5, 4)
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def save_training_curves(
    history: Dict[str, List[float]],
    plots_dir: str = config.PLOTS_DIR,
) -> Dict[str, str]:
    """
    history expects keys: 'train_loss', 'val_loss', 'accuracy', 'f1'
    (each a list with one value per epoch). Produces loss, accuracy,
    and F1 graphs.
    """
    os.makedirs(plots_dir, exist_ok=True)
    saved = {}
    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss graph
    if "train_loss" in history and "val_loss" in history:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
        plt.plot(epochs, history["val_loss"], label="Val Loss", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(plots_dir, "loss_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        saved["loss"] = path

    # Accuracy graph
    if "accuracy" in history:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["accuracy"], label="Val Accuracy", marker="o", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Validation Accuracy over Epochs")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(plots_dir, "accuracy_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        saved["accuracy"] = path

    # F1 graph
    if "f1" in history:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history["f1"], label="Val F1", marker="o", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title("Validation F1 Score over Epochs")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(plots_dir, "f1_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        saved["f1"] = path

    return saved