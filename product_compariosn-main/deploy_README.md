# Product Comparison API — deployment folder

Self-contained. Everything here is needed to serve the model; nothing here is
needed to train it. Regenerate with `python build_deploy.py --model` from the
repo root.

## Which model is in here, and why

**v11** — `deberta-v3-small` cross-encoder, binary, decision threshold **0.50**,
description budget **20 words**, trained on 58,112 pairs including 19,967
similarity-mined hard pairs.

Five checkpoints were trained and measured. v11 is best on every axis that
matters and is 24% cheaper to serve:

| | v7 | desc20 v9 | hardneg v10 | **v11** |
|---|---|---|---|---|
| Benchmark mean | 81.70 | 80.07 | 80.29 | **82.26** |
| WDC-UNSEEN | 74.78 | — | 72.23 | **75.52** |
| Indian F1 (SE 3.6) | 79.45 | 79.72 | 77.33 | **80.85** |
| Indian precision | 69.88 | 71.25 | 66.67 | **73.08** |
| Indian errors | 30 | 29 | 34 | **27** |
| hard-slice precision | 15.38 | 16.00 | 12.50 | **17.39** |
| cost of not knowing | 1.04 | 2.29 | 1.91 | **0.63** |
| inference cost | 1.00x | 0.76x | 0.76x | **0.76x** |

WDC-UNSEEN 75.52 beats the published RoBERTa baseline (71.14) by 4.38, the
widest margin this project has recorded.

**A caution about how much of this to believe.** v10 and v11 are the SAME
recipe on the SAME corpus, differing only by random seed, and they span
80.29-82.26 on benchmarks, 77.33-80.85 on Indian F1 and 12.50-17.39 on
hard-slice precision. Seed variance is therefore larger than most differences
in the table above. v11 is genuinely the best checkpoint measured, but a 1-2
point gap between recipes here is not evidence about the recipe. Future
comparisons need 2-3 seeds each.

## Known limits of this model — read before trusting an answer

Measured on 190 hand-labelled real Indian e-commerce pairs:

| slice | F1 | note |
|---|---|---|
| easy | **97.96** | different categories, obvious duplicates |
| medium | 90.62 | ordinary cross-merchant matching |
| **hard** | **28.57** | precision **17.39%** |

The model is excellent where the distinction is lexically obvious and **fails
on near-misses**, over-predicting "same product" (21 false positives against 6
false negatives). Hard-slice precision has never exceeded 17.39 across five
independently trained checkpoints spanning three corpora, so this is a property
of the available training data, not a bad seed.

Specific failures on v11, each scored above 98% confidence:

```
JBL Tune 215 (neckband) vs  JBL Tune 215TWS             99.95%
Noise Buds VS104        vs  Noise Buds VS104 Max        99.86%
Mamaearth 150ml 2-pack  vs  Mamaearth 100ml 2-pack      99.19%
genuine boAt Airdopes   vs  "Thirty First For Boat"     99.05%
genuine OnePlus Z2      vs  "Compatible with OnePlus"   98.12%
Nike Revolution 6 std   vs  Revolution 6 4E (wide)      97.41%
```

**Practical consequence:** do not auto-accept a match on score alone for
same-brand candidates whose titles differ only by a suffix (`Max`, `Pro`,
`Gen 2`, `Lite`), by a form-factor word, or where one side contains `for` /
`compatible with`. Those need human review or a rule-based guard in front.
A high score in those cases carries no information.

## ⚠️ Read before exposing this to a network

This service has **no authentication and no rate limiting**. Anyone who can
reach the port gets unlimited inference on your hardware. `/compare` caps
candidates at 200 and `/search` caps `retrieve_k` at 200, so a single request
is bounded, but the number of requests is not.

Run it on localhost, inside a private network, or behind a gateway that
provides auth and rate limiting. Do not put it on a public IP as-is.

## What's here

```
config.py, utils.py               shared settings and logging
api/                              FastAPI app, routes, schemas
exact_match/inference.py          ProductComparer (the cross-encoder)
exact_match/preprocessing.py      text building
ranking/                          two-stage orchestration + FAISS retrieval
generate_relationship_pairs.py    categorize(), used by the ranker
product_taxonomy.py               category rules
data_quality/                     contradiction rules used by the above
trained_model_real/               the v7 checkpoint (541 MB)
requirements.txt, run.sh, run.ps1
smoke_test.py                     proves it works — see below
```

The last four Python modules look unrelated to serving but are not optional:
`api → ranking.ranker → ranking.candidate_retrieval → generate_relationship_pairs
→ {product_taxonomy, data_quality.contradiction_rules}`. Removing any of them
fails at the first `/compare`, not at startup.

## Install

```bash
python -m venv venv
```

```bash
venv/bin/pip install -r requirements.txt
```

On Windows use `venv\Scripts\pip` instead. Torch is ~2 GB; CPU-only is fine.

`/search` additionally needs `faiss-cpu` and `sentence-transformers` (commented
out in `requirements.txt`) plus a built index. `/compare` needs neither.

## Run

```bash
./run.sh
```

Windows: `.\run.ps1`. Override the port with `PORT=9000 ./run.sh`, and the
checkpoint with `MODEL_DIR=/path/to/model`.

Startup takes 10–30 seconds — a 541 MB checkpoint is loaded **once**, not per
request. Wait for `Application startup complete.` before testing.

## Test it yourself

### 1. Automated

```bash
python smoke_test.py
```

Exit code 0 means every check passed, so this can gate a deploy. It checks the
checkpoint metadata on disk, `/health`, a true match, an unrelated product,
ranking order, and input validation.

**Why both a match and a non-match test:** the failure this service is most
exposed to is a serialization mismatch — the model is trained on
`COL brand VAL ... COL title VAL ...` and, fed anything else, silently stops
recognising matches instead of erroring. Measured on this project: three
obvious matches scored 49.8% / 20.6% / 4.4% instead of ~100%, recall on real
matches was zero, and `/health` reported `ok` throughout. A match test catches
that but passes trivially on a model that says "same" to everything; a
non-match test catches the always-yes model but a broken model passes it too.
Either check alone can be green while the service is useless.

### 2. By hand

Health — `model_loaded` must be `true` and `num_labels` must be `2`:

```bash
curl http://localhost:8000/health
```

A true match. Expect `similarity_score` above 99:

```bash
curl -s -X POST http://localhost:8000/compare -H "Content-Type: application/json" -d "{\"product\":{\"id\":\"P1\",\"title\":\"boAt Airdopes 141 Elite ANC | 42H Earbuds with 35dB ANC Black\",\"brand\":\"boAt\",\"description\":\"42 hours playtime, 35dB ANC\"},\"candidates\":[{\"id\":\"P2\",\"title\":\"boAt Airdopes 141 Elite ANC | Black\",\"brand\":\"boAt\",\"description\":\"ANC true wireless earbuds\"}]}"
```

An unrelated product. Expect a score near 0:

```bash
curl -s -X POST http://localhost:8000/compare -H "Content-Type: application/json" -d "{\"product\":{\"id\":\"P1\",\"title\":\"boAt Airdopes 141 Elite ANC\",\"brand\":\"boAt\",\"description\":\"ANC earbuds\"},\"candidates\":[{\"id\":\"P3\",\"title\":\"Nike Air Max 270 Running Shoes\",\"brand\":\"Nike\",\"description\":\"Mesh upper\"}]}"
```

Interactive docs, which let you build requests in a browser:

```bash
python -c "import webbrowser; webbrowser.open('http://localhost:8000/docs')"
```

### 3. What "working" looks like

All values below are measured on this checkpoint, not estimated.

| Check | Expected |
|---|---|
| `/health` | `model_loaded: true`, `num_labels: 2` |
| True match (same product, extra words in title) | **99.99%**, returned as `exact_match` |
| Several merchants selling one product | best in `exact_match`, **rest in `other_matches`** |
| Same product, different colour | **~0.00%**, returned under `similar_products` as `SIMILAR_ALTERNATIVE` |
| Unrelated product | **omitted from the response entirely** |
| Empty `candidates` | HTTP 422 |
| No model found | HTTP 503, not a crash |

**`other_matches` is why candidate counts must balance.** `exact_match` is a
single slot, but several merchants listing one product is the normal case for
price comparison — the whole point of the service. Every additional candidate
the model judges to be the same product goes in `other_matches`, best first.
An earlier version kept only the top one and silently discarded the rest;
`smoke_test.py` now counts candidates in versus out, which is the only check
that catches that class of bug.

Two of those surprise people:

**A colour variant scores near zero, not near 100.** The model answers one
question — "is this the same purchasable product?" — and Black is not White.
The *ranker* still surfaces it as a `SIMILAR_ALTERNATIVE` with the reason
`Same brand: boAt`. Low score plus alternative relationship is correct output
here, not a failure.

**An unrelated product is dropped, not scored low.** It will not appear in
`similar_products` at all, so don't test for a small number — test for absence.

**If matches score 20–50% instead of >99%, stop.** That is the serialization
mismatch, not a weak model. Check `trained_model_real/training_metadata.json`
contains `"serialization": "colval"`. A checkpoint missing that field falls
back to the wrong format and takes match recall to zero while `/health` still
reports `ok`.

## Known limits

- No auth, no rate limiting (above)
- `/search` returns 503 unless a FAISS index is present at `data/product_index`
  or `INDEX_DIR`
- The index cannot be updated incrementally; rebuilding 112k products takes
  ~16 minutes on CPU
- Latency was measured on CPU only; concurrent load was never tested
- Decision threshold is 0.50, recorded in the checkpoint. It was fitted on
  validation and deliberately left at the default because no fitted value
  cleared the noise floor
