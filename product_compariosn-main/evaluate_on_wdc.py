"""
evaluate_on_wdc.py
==================
Zero-shot evaluation of the trained model against WDC Products --
HUMAN-LABELLED product pairs from real e-shops.

WHY THIS EXISTS
---------------
Every number this project has produced was measured against labels generated
by `generate_relationship_pairs.py`. Run 8 showed how little those numbers
mean: eliminating 29.85% entity leakage moved accuracy by 0.07pp
(0.9508 -> 0.9501). That is impossible if the model were memorising products
-- and it is. The model is not memorising entities, it is recovering the
LABELLING RULE, and a rule generalises perfectly to unseen products.

So the 95% measures "can the model reverse-engineer our own code". No split
fix, extra data, or architecture change repairs that. The only way to learn
whether this model matches PRODUCTS is to grade it against labels a human
wrote.

WHAT THIS MEASURES -- AND WHAT IT DOES NOT
------------------------------------------
This is a ZERO-SHOT TRANSFER test: trained on our synthetic data, evaluated on
WDC having never seen WDC training data. The published Ditto / RoBERTa figures
are SUPERVISED, so they are NOT peers -- they are printed only as scale. The
question actually answered here is narrower and more useful:

    does anything we learned from synthetic data transfer to real listings?

THE UNSEEN DIMENSION
--------------------
WDC gold-standard filenames encode the benchmark dimensions:

    wdcproducts{CC}cc{RND}rnd{UN}un_gs.json.gz

`UN` is the share of test products never seen in training -- 000 / 050 / 100,
i.e. the Seen / Half-Seen / Unseen columns of Table 3 in Peeters et al.
Evaluating all three separates "recognises products it has met" from "matches
products it has never met", which is the real task.

TASK MAPPING
------------
WDC is binary (1 = same product). Our model emits 5 classes, so the mapping is
a judgement call and both defensible options are reported:

    strict : EXACT_MATCH                                   -> match
    loose  : EXACT_MATCH or SAME_PRODUCT_DIFFERENT_VARIANT -> match

`strict` is the closer reading (WDC treats a different variant as a different
entity); `loose` is reported in case our VARIANT class absorbs pairs WDC calls
matches. Argmax over five classes is also not necessarily the best binary
decision rule, so a threshold sweep over P(match) is reported alongside it --
a model can rank pairs correctly while its default cut-point sits in the wrong
place, and those two failures need entirely different fixes.

Usage:
    python evaluate_on_wdc.py --wdc /path/to/50pair.zip
    python evaluate_on_wdc.py --wdc /path/to/extracted_folder
    python evaluate_on_wdc.py                 # HuggingFace mirror fallback
"""

import argparse
import glob
import gzip
import json
import os
import zipfile
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from exact_match.preprocessing import build_product_text
from utils import get_logger

logger = get_logger(__name__)

CATEGORIES = ["computers", "cameras", "watches", "shoes"]
STRICT_MATCH_LABELS = {"EXACT_MATCH"}
LOOSE_MATCH_LABELS = {"EXACT_MATCH", "SAME_PRODUCT_DIFFERENT_VARIANT"}

UNSEEN_TAGS = {"000": "SEEN", "050": "HALF-SEEN", "100": "UNSEEN"}

# Table 3, pair-wise, medium training set, 50% corner cases. These systems were
# TRAINED on WDC. Printed as scale only, never as a peer of a zero-shot score.
PUBLISHED_F1 = {
    "RoBERTa": {"SEEN": 78.58, "HALF-SEEN": 75.91, "UNSEEN": 71.14},
    "Ditto": {"SEEN": 79.16, "HALF-SEEN": 75.22, "UNSEEN": 70.24},
    "HierGAT": {"SEEN": 75.17, "HALF-SEEN": 73.30, "UNSEEN": 68.74},
    "R-SupCon": {"SEEN": 81.88, "HALF-SEEN": 68.69, "UNSEEN": 57.23},
}


def load_wdc_local(path: str) -> Dict[str, List[dict]]:
    """Reads official WDC gold standards from a directory or a zip.

    Preferred over the HuggingFace mirror, which does not expose the
    seen/half-seen/unseen dimension this comparison depends on.
    """
    work = path
    if path.lower().endswith(".zip"):
        # Extract into the CURRENT WORKING DIRECTORY, not next to the zip.
        # Kaggle mounts /kaggle/input read-only, so extracting beside the
        # source archive raises PermissionError there. cwd is writable in
        # every environment this runs in.
        work = os.path.join(os.getcwd(), "_wdc_extracted")
        os.makedirs(work, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(work)
        logger.info(f"extracted {path} -> {work}")

    # Accept both '.json.gz' and plain '.json'. Kaggle decompresses archives
    # when a Dataset is uploaded, so the same benchmark arrives gzipped from
    # webdatacommons but un-gzipped once it has been through Kaggle.
    candidates = sorted(
        glob.glob(os.path.join(work, "**", "*_gs.json.gz"), recursive=True)
        + glob.glob(os.path.join(work, "**", "*_gs.json"), recursive=True)
    )

    found: Dict[str, List[dict]] = {}
    for src in candidates:
        base = os.path.basename(src)
        if "rnd" not in base or "un_gs" not in base:
            continue
        tag = UNSEEN_TAGS.get(base.split("rnd")[1].split("un")[0])
        if tag is None or tag in found:
            continue

        opener = gzip.open if src.endswith(".gz") else open
        with opener(src, "rt", encoding="utf-8") as fh:
            text = fh.read()
        # Normally JSON-lines; tolerate a plain JSON array too, so an
        # unreadable file is never mistaken for an empty benchmark.
        try:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError:
            rows = json.loads(text)
            if isinstance(rows, dict):
                rows = [rows]

        found[tag] = rows
        pos = sum(r["label"] for r in rows)
        logger.info(f"  {tag:<10} {base}  n={len(rows):,}  positives={pos:,} ({pos / len(rows):.1%})")

    if not found:
        raise SystemExit(
            f"No '*_gs.json[.gz]' gold-standard files under {work}. "
            "Point --wdc at the folder holding the wdcproducts*_gs.json files."
        )
    return found


def load_wdc_hf(categories: List[str], size: str, split: str) -> Dict[str, List[dict]]:
    """Fallback: HuggingFace products-2017 mirror (no unseen dimension)."""
    from datasets import load_dataset

    rows: List[dict] = []
    for cat in categories:
        name = f"{cat}_{size}"
        try:
            ds = load_dataset("wdc/products-2017", name, split=split)
            rows.extend(dict(r) for r in ds)
            logger.info(f"  loaded {name}/{split}: {len(ds):,} pairs")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"  could not load {name}/{split}: {exc}")
    if not rows:
        raise SystemExit("No WDC subsets loaded; check internet access and `datasets`.")
    return {"products-2017": rows}


def build_texts(rows) -> tuple:
    """Serialises each side with the SAME function used in training.

    Using build_product_text rather than a bespoke formatter matters: with a
    different format, a low score could mean 'does not transfer' OR 'the input
    looks unfamiliar', and those would be indistinguishable.
    """
    text_a, text_b = [], []
    for row in rows:
        text_a.append(build_product_text(
            row.get("title_left") or "",
            brand=row.get("brand_left") or "",
            description=row.get("description_left") or "",
        ))
        text_b.append(build_product_text(
            row.get("title_right") or "",
            brand=row.get("brand_right") or "",
            description=row.get("description_right") or "",
        ))
    return text_a, text_b


@torch.no_grad()
def predict_probabilities(model, tokenizer, text_a, text_b, device, batch_size=64) -> np.ndarray:
    """Real batching -- inference.py's compare_batch loops one pair at a time."""
    model.eval()
    out = []
    for i in tqdm(range(0, len(text_a), batch_size), desc="Scoring pairs"):
        enc = tokenizer(
            text_a[i:i + batch_size], text_b[i:i + batch_size],
            truncation=True, max_length=config.MAX_SEQ_LENGTH,
            padding=True, return_tensors="pt",
        ).to(device)
        out.append(torch.softmax(model(**enc).logits, dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def summarise(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> Dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = float((y_true == y_pred).mean())
    print(f"  {title}")
    print(f"    precision {p:.4f} | recall {r:.4f} | F1 {f:.4f} | accuracy {acc:.4f}")
    return {"precision": p, "recall": r, "f1": f, "accuracy": acc}


def sweep_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple:
    """Best achievable F1 over P(match), and the cut-point producing it."""
    best_f1, best_t = -1.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        f = f1_score(y_true, (scores >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    return best_f1, best_t


def evaluate_split(tag, rows, model, tokenizer, id2label, name_to_id, device, batch_size):
    y_true = np.array([r["label"] for r in rows], dtype=int)
    text_a, text_b = build_texts(rows)
    probs = predict_probabilities(model, tokenizer, text_a, text_b, device, batch_size)
    pred_names = np.array([id2label[i] for i in probs.argmax(axis=1)])

    print()
    print("=" * 72)
    print(f"{tag}  --  n={len(y_true):,}  positives={y_true.sum():,} ({y_true.mean():.1%})")
    print("=" * 72)

    out = {}
    for label, match_set in (("strict", STRICT_MATCH_LABELS), ("loose", LOOSE_MATCH_LABELS)):
        present = [n for n in match_set if n in name_to_id]
        if not present:
            continue
        y_pred = np.isin(pred_names, list(match_set)).astype(int)
        m = summarise(y_true, y_pred, f"argmax -- {label} ({', '.join(sorted(match_set))})")
        score = probs[:, [name_to_id[n] for n in present]].sum(axis=1)
        best_f1, best_t = sweep_threshold(y_true, score)
        print(f"    best F1 over P(match) sweep: {best_f1:.4f} at threshold {best_t:.2f}")
        m["best_f1_swept"] = best_f1
        out[label] = m

    # Hard negatives carry the benchmark's difficulty. Scoring well overall
    # while failing here would mean the model only rejects obvious mismatches.
    hard = np.array([bool(r.get("is_hard_negative")) for r in rows])
    hard_neg = hard & (y_true == 0)
    if hard_neg.any():
        y_pred = np.isin(pred_names, list(STRICT_MATCH_LABELS)).astype(int)
        fp = float(y_pred[hard_neg].mean())
        print(f"  false-positive rate on {int(hard_neg.sum()):,} HARD negatives: {fp:.1%}")

    print("  predicted class distribution on these real pairs:")
    for name, n in zip(*np.unique(pred_names, return_counts=True)):
        print(f"    {name:<32} {n:>6,}  ({n / len(pred_names):6.2%})")

    y_pred_strict = np.isin(pred_names, list(STRICT_MATCH_LABELS)).astype(int)
    print(classification_report(y_true, y_pred_strict,
                                target_names=["non-match", "match"], zero_division=0))
    return out


def main():
    ap = argparse.ArgumentParser(description="Zero-shot evaluation on WDC Products.")
    ap.add_argument("--model_dir", default=config.TRAINED_MODEL_DIR)
    ap.add_argument("--wdc", default=None,
                    help="Folder of extracted WDC files, or a zip such as 50pair.zip. "
                         "Omit to fall back to the HuggingFace products-2017 mirror.")
    ap.add_argument("--categories", nargs="+", default=CATEGORIES, choices=CATEGORIES)
    ap.add_argument("--size", default="medium", choices=["small", "medium", "large", "xlarge"])
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading model from {args.model_dir} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    name_to_id = {v: k for k, v in id2label.items()}
    logger.info(f"Model classes: {id2label}")
    if len(id2label) != 5:
        logger.warning("Not the 5-class relationship model; mappings below may not apply.")

    if args.wdc:
        splits = load_wdc_local(args.wdc)
    else:
        logger.info("No --wdc given; falling back to the HuggingFace mirror.")
        splits = load_wdc_hf(args.categories, args.size, args.split)

    order = [t for t in ("SEEN", "HALF-SEEN", "UNSEEN") if t in splits] or list(splits)
    results = {t: evaluate_split(t, splits[t], model, tokenizer, id2label,
                                 name_to_id, device, args.batch_size) for t in order}

    print()
    print("=" * 72)
    print("SUMMARY -- F1 x 100, best over the P(match) threshold sweep")
    print("=" * 72)
    print(f"{'system':<28}" + "".join(f"{c:>12}" for c in order))
    for label in ("strict", "loose"):
        vals = [results[t].get(label, {}).get("best_f1_swept") for t in order]
        if any(v is not None for v in vals):
            row = "".join(f"{v * 100:>12.2f}" if v is not None else f"{'-':>12}" for v in vals)
            print(f"{'THIS MODEL (' + label + ', 0-shot)':<28}{row}")
    if set(order) & set(UNSEEN_TAGS.values()):
        for sysname, byt in PUBLISHED_F1.items():
            row = "".join(f"{byt[c]:>12.2f}" if c in byt else f"{'-':>12}" for c in order)
            print(f"{sysname + ' (SUPERVISED)':<28}{row}")

    print()
    print("=" * 72)
    print("HOW TO READ THIS")
    print("=" * 72)
    print("The published rows were TRAINED on WDC's own training split. This")
    print("model never saw WDC at all, so a lower score is expected and is NOT")
    print("evidence the architecture is worse. The gap measures how much of what")
    print("our synthetic data taught actually transfers to real listings.")
    print("Source: Peeters et al., Table 3 -- https://arxiv.org/html/2301.09521")


if __name__ == "__main__":
    main()
