# Start here

Read `PROJECT_HANDOVER.pdf` first — it is the real document, written to be read
cold. This file is only a pointer and a list of what is in flight.

## State as of 14 August 2026

**Shipped model: v11** — `trained_model_v11/`, archived as
`trained_model_desc20_v11.zip`. `deberta-v3-small`, binary, threshold **0.50**,
description budget **20 words** (recorded inside the checkpoint — see below).

- Benchmark mean **82.26**, WDC-UNSEEN **75.52** (best recorded, +4.38 over RoBERTa)
- Indian real-market F1 **80.85** on 190 hand-labelled pairs
- `deploy/` is built on v11: **12/12 smoke checks**, load-tested
- Tests: **53 passed, 1 xfail**

## Three things that will bite you

1. **The description budget and the decision threshold live in the checkpoint,
   not in config.** `config.DESCRIPTION_WORDS` is 20, but v7 trained at 100 and
   still serves at 100 because its metadata says so. Serving the wrong budget
   does not raise, it quietly costs accuracy. Same for `inference_threshold`.

2. **Run-to-run variance is ~2 benchmark F1 / ~3.5 Indian F1 on identical
   inputs.** v10 and v11 are the same recipe at the same seed and differ by
   that much. Never conclude from one run. `config.RANDOM_SEED` reads `$SEED`.

3. **The hard slice is the real limitation.** 17.39% precision on near-misses
   (`VS104` vs `VS104 Max`, `141` vs `141 Pro`, third-party knockoffs). Stable
   across five checkpoints. Everything easy scores 97–100.

## In flight — not finished

| Item | State |
|---|---|
| Swap augmentation (`--swap-augment`) | Written and verified locally, **not trained yet** |
| `kaggle_job/` headless kernel | Written; needs `~/.kaggle/kaggle.json` and the WDC dataset slug in `kernel-metadata.json` |
| Auth + rate limiting | **Not started.** The real deployment blocker |
| `--workers N` | Not applied. Throughput is 2.3 req/s regardless of concurrency |
| ~2,000 Indian labelled pairs | 190 done (eval only). The last accuracy lever |

## Commands that matter

```bash
python build_real_corpus.py --wdc 50pair --wdc-size large --swap-augment
```

```bash
python train_on_real_corpus.py --wdc 50pair
```

```bash
python evaluate_indian.py --model trained_model_v11
```

```bash
python calibrate_threshold.py --model trained_model_v11 --wdc 50pair --write
```

```bash
python build_deploy.py --model --from trained_model_v11
```

Order matters on Kaggle: **calibrate before zipping**, or the archive ships
without `inference_threshold`. Every run before v11 got this wrong.

Do **not** run `add_lspc_corpus.py` — measured null (S2.7). Use
`mine_hard_negatives.py` instead, which is what v11 was built with.

## Files kept vs deleted

Deleted 3.5 GB of scratch on 14 Aug: `text_part_*.{txt,zip}` (dead pipeline),
`50pair.zip` (extracted to `50pair/`), the empty `trained_model/`,
`trained_model_v8/`, and the defective `BROKEN_DO_NOT_DEPLOY_*v7.zip`.

Kept: `trained_model_v7_final.zip` (verified v7), `trained_model_desc20_v11.zip`
(shipped), `trained_model_desc20_v9.zip` and `trained_model_v8.zip` (records),
`deploy_2.0.zip`. `trained_model_real/` is v7; `trained_model_v11/` is v11.
