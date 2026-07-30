# Product Comparison API — deployment folder

Self-contained. Everything here is needed to serve the model; nothing here is
needed to train it. Regenerate with `python build_deploy.py --model` from the
repo root.

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
| Same product, different colour | **~0.00%**, returned under `similar_products` as `SIMILAR_ALTERNATIVE` |
| Unrelated product | **omitted from the response entirely** |
| Empty `candidates` | HTTP 422 |
| No model found | HTTP 503, not a crash |

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
