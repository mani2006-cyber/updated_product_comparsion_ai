# Product Comparison AI

Fine-tunes a pretrained Hugging Face transformer to decide whether two
e-commerce product listings (title + specs) refer to **the same product**
or **different products** — the entity-resolution problem behind any
price-comparison platform.

## Why `microsoft/deberta-v3-small`

Product matching is structurally the same task as NLI/STS ("do these two
spans of text mean the same thing?"). DeBERTa-v3's disentangled attention
and ELECTRA-style pretraining consistently beat BERT/RoBERTa on exactly
this kind of pairwise sentence classification, and the "small" variant
(~140M params) trains fast on a single T4 while its SentencePiece
tokenizer handles messy e-commerce tokens ("128GB" vs "128 GB", mixed
case) more gracefully than WordPiece. Swap `MODEL_NAME` in `config.py` to
`distilbert-base-uncased` (fastest) or `microsoft/deberta-v3-base`
(higher ceiling) with zero other code changes.

## Folder structure

```
product_comparison_ai/
├── requirements.txt
├── config.py            # every path / hyperparameter / switch
├── preprocessing.py      # load -> clean -> split
├── dataset.py            # tokenization -> DataLoaders
├── model.py               # backbone + optional LoRA
├── metrics.py             # accuracy/precision/recall/F1/ROC-AUC + plots
├── utils.py                # seed, early stopping, ETA, checkpoints
├── train.py                 # full training loop
├── evaluate.py               # standalone evaluation + reports
├── save_model.py              # export trained_model/ in HF format
├── inference.py                 # compare two products
├── data/
│   └── products.csv              # your dataset (product1,product2,label)
├── outputs/
│   ├── checkpoints/                # best_model.pt, last_epoch.pt
│   ├── plots/                       # loss/accuracy/F1 curves, confusion matrix
│   ├── logs/                         # run.log
│   └── reports/                       # classification_report_*.txt
└── trained_model/                       # final exported HF model
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

## Setup

```bash
pip install -r requirements.txt
```

## Dataset schema

Two schemas are auto-detected:

**A) CSV, title-only** (matches your uploaded `products.csv`):
```
product1,product2,label
Apple iPhone 16 Pro 128GB Desert Titanium,Apple iPhone 16 Pro (128 GB) Desert Titanium,1
```

**B) JSON/JSONL, title + specs:**
```json
{"product1_title": "iPhone 15 128GB Black", "product1_specs": "Apple A16, 128GB, 6.1 OLED",
 "product2_title": "Apple iPhone 15 Black 128 GB", "product2_specs": "A16 Bionic, 128 GB, 6.1-inch OLED",
 "label": 1}
```
`label`: `1` = same product, `0` = different product.

Drop your file at `data/products.csv` (or point `config.RAW_DATA_PATH`
at a `.json`/`.jsonl` file) and everything downstream adapts automatically.

## Training pipeline — step by step

| # | Step | Where |
|---|------|-------|
| 1 | Load dataset | `preprocessing.load_raw_data` |
| 2 | Clean data (lowercase, unit normalization, dedup, drop invalid labels) | `preprocessing.clean_dataframe` |
| 3 | Tokenization (pair encoding: `[CLS] A [SEP] B [SEP]`) | `dataset.ProductPairDataset` |
| 4 | Preprocessing into tensors (padding/truncation to `MAX_SEQ_LENGTH`) | `dataset.ProductPairDataset.__getitem__` |
| 5 | Train/Val/Test split (stratified) | `preprocessing.split_data` |
| 6 | DataLoader creation | `dataset.build_all_dataloaders` |
| 7 | Model loading (pretrained backbone + classification head, optional LoRA) | `model.load_model_and_tokenizer` |
| 8 | Fine-tuning loop | `train.train` |
| 9 | Loss function (class-weighted CrossEntropyLoss) | `train.train` |
| 10 | Optimizer (AdamW) | `train.train` |
| 11 | LR scheduler (linear warmup + decay) | `train.train` via `get_scheduler` |
| 12 | Mixed precision (fp16/bf16 via Accelerate) | `Accelerator(mixed_precision=...)` |
| 13 | Gradient accumulation | `accelerator.accumulate(model)` |
| 14 | Early stopping (on val loss) | `utils.EarlyStopping` |
| 15 | Checkpoint saving (best + last epoch) | `utils.save_checkpoint` |
| 16 | Evaluation loop | `train.run_evaluation` / `evaluate.evaluate_model` |
| 17-21 | Accuracy, Precision, Recall, F1, ROC-AUC | `metrics.compute_metrics` |
| 22 | Save best model | `save_model.export_trained_model` |
| 23 | Export tokenizer | `save_model.export_trained_model` |
| 24 | Inference | `inference.ProductComparer` |

### During training you'll see, per step and per epoch:

- live progress bar with running loss, current learning rate, and ETA
- per-epoch: `train_loss`, `val_loss`, `lr`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc`
- a log line whenever a new best checkpoint is saved

## Usage

**Train:**
```bash
python train.py
# or, for proper multi-GPU / AMP setup:
accelerate launch train.py
```
Produces `outputs/checkpoints/best_model.pt`, `outputs/plots/{loss,accuracy,f1}_curve.png`,
and exports the final model to `trained_model/`.

**Evaluate** (confusion matrix + classification report on the held-out test split):
```bash
python evaluate.py --split test
```

**Compare two products:**
```bash
python inference.py \
  --title_a "iPhone 15 128GB Black" \
  --specs_a "Apple A16, 128GB Storage, 6.1 OLED, 48MP Camera" \
  --title_b "Apple iPhone 15 Black 128 GB" \
  --specs_b "A16 Bionic, 128 GB, 6.1-inch OLED, 48 MP"
```
```
Output:
Similarity Score: 98.7%
Prediction:
Same Product
```

Or from Python:
```python
from inference import ProductComparer
comparer = ProductComparer()
result = comparer.compare(title_a="...", specs_a="...", title_b="...", specs_b="...")
print(result.similarity_score, result.prediction)
```

## Notes on your uploaded dataset

`data/products.csv` has 219 rows in the title-only schema
(`product1, product2, label`) — no separate specs column, which the
pipeline handles natively. For better accuracy, consider expanding it:
219 rows is enough to validate the pipeline end-to-end, but a few
thousand labeled pairs (with more hard negatives — similar-but-different
products) will meaningfully improve generalization before you rely on
this model in production.

## Switching to LoRA

Set `USE_LORA = True` in `config.py`. Most useful if you move to a
larger backbone (`deberta-v3-base`/`large`) or want multiple
category-specific adapters around one frozen backbone — recommended
once you outgrow the small backbone rather than at the current dataset
size.
