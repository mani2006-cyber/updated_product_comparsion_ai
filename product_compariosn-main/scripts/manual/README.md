# Manual scripts

Ad-hoc scripts that are run by hand and inspected by eye. **None of them contain
assertions** — they print output or write files, and nothing fails automatically.

They used to live in `tests/` (and, for `diagnostic_pairs.py`, the repo root),
where `pytest` collected them as if they were tests. That was actively
dangerous: collecting `merge_general_pairs_v2.py` *executes* it, and it writes to
`data/`. A bare `pytest` run would silently rewrite a dataset.

Nothing here is part of the test suite. Do not move these back into `tests/`.

| Script | What it does |
|---|---|
| `merge_general_pairs_v2.py` | Merges the catalog + audio pair CSVs, dropping `WEAKLY_SIMILAR`, and **writes** `data/relationship_pairs_general_v2.csv`. |
| `stress_test_audio.py` | Prints `ProductComparer` output for 5 hand-written audio-accessory pairs. Needs a trained checkpoint. |
| `diagnostic_pairs.py` | Prints `ProductComparer` output for ~35 pairs across exact-match / variant / alternative / unrelated / typo / brand-confusion buckets. Needs a trained checkpoint. |

All three assume the repo root is the working directory:

```bash
python scripts/manual/diagnostic_pairs.py
```
