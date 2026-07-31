# Product Comparison API — deployment folder

Self-contained. Everything here is needed to serve the model; nothing here is
needed to train it. Regenerate with `python build_deploy.py --model` from the
repo root.

## Which model is in here, and why

**v7** — `deberta-v3-small` cross-encoder, binary, decision threshold **0.50**,
description budget **100 words**.

Four checkpoints were trained and measured against each other. v7 wins because
it is the only one with a statistically detectable advantage anywhere
(benchmark mean 81.70 vs 80.07–81.91), ties every rival on Indian data within
noise, and is the only one verified end to end from its archive.

| | v7 | desc20 v9 | hardneg v10 |
|---|---|---|---|
| Benchmark mean | **81.70** | 80.07 | 80.29 |
| Indian F1 (SE 3.6) | 79.45 | 79.72 | 77.33 |
| Indian errors | 30 | **29** | 34 |
| hard-slice precision | 15.38 | **16.00** | 12.50 |
| inference cost | 1.00x | **0.76x** | **0.76x** |

`desc20 v9` is a legitimate swap if inference cost matters — statistically
identical on Indian data and 24% cheaper — but it early-stopped an epoch sooner
than v7, so its benchmark deficit is unresolved. Settle that before switching.

## Known limits of this model — read before trusting an answer

Measured on 190 hand-labelled real Indian e-commerce pairs:

| slice | F1 | note |
|---|---|---|
| easy | **100.00** | different categories, obvious duplicates |
| medium | 89.55 | ordinary cross-merchant matching |
| **hard** | **25.81** | precision **15.38%** |

The model is excellent where the distinction is lexically obvious and **fails
on near-misses**, over-predicting "same product" (25 false positives against 5
false negatives). Confirmed reproducible across four independently trained
checkpoints, so it is a property of the training data, not a bad seed.

Specific failures, each scored above 99% confidence:

```
Noise Buds VS104        vs  Noise Buds VS104 Max      99.99%
boAt Airdopes 141       vs  boAt Airdopes 141 Pro     99.60%
JBL Tune 215 (neckband) vs  JBL Tune 215TWS           99.96%
genuine boAt Airdopes   vs  "Thirty First For Boat"   99.92%
Airdopes 141 Bold Black vs  Airdopes 141 Active Black 99.97%
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
