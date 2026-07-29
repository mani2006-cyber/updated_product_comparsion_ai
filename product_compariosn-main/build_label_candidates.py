"""
build_label_candidates.py
=========================
Turns the collected real listings into CANDIDATE PAIRS for human/AI labelling.

It assigns NO labels. That is the point.

WHY NO LABELS
-------------
Every previous dataset in this project was labelled by a rule, and the model
learned the rule rather than the task: 95% on synthetic labels, 45.9 F1 on
human-labelled WDC pairs. Google Shopping's own clustering is not a safe
substitute either -- in the collected data it grouped "Thirty First Earbuds
For Boat 141" (a third-party item at ~2x the price) with the genuine boAt
product, and placed one identical Amazon listing into two different colour
clusters at once. Any script that turned those groupings into labels would
rebuild the exact failure this project just escaped.

So this file only proposes pairs worth a human deciding, and records the
evidence needed to decide them.

CANDIDATE SOURCES
-----------------
  cross_merchant   two offers Google grouped as one product, from different
                   stores. Likely matches, but includes Google's errors.
  same_cluster_alt two offers Google grouped, same store or same title -- for
                   catching duplicate/reseller listings.
  near_miss        two products from the SAME search but DIFFERENT Google
                   clusters (e.g. "Airdopes 141" vs "141 Gen 2" vs "141 Pro").
                   Adjacent models: the hard negatives models actually fail on.
  amazon_cluster   two Amazon listings surfaced by the same query. Dense in
                   adjacent models, colour variants, and accessories.
  cross_category   listings from unrelated queries. Easy negatives, included
                   only in small numbers -- a set made mostly of these would
                   score well and mean nothing.

EVIDENCE RECORDED
-----------------
price_ratio is included because it is genuinely diagnostic: the boAt knockoff
was flagged by ~2x price on a "same" product, and the ₹199-vs-₹819 listing by
0.24x. It is evidence for a human, never an automatic rule.

Usage:
    python build_label_candidates.py
    python build_label_candidates.py --max-per-source 80
"""

import argparse
import itertools
import json
import os
import random
import re
from typing import Dict, List, Optional

RANDOM_SEED = 20260730


def _price(value) -> Optional[float]:
    if not value:
        return None
    digits = re.sub(r"[^\d.]", "", str(value).replace(",", ""))
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


def _ratio(a, b) -> Optional[float]:
    pa, pb = _price(a), _price(b)
    if not pa or not pb:
        return None
    lo, hi = sorted((pa, pb))
    return round(hi / lo, 2) if lo > 0 else None


def load_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _row(pair_id, source, a, b, a_title, b_title, a_store, b_store, a_price, b_price, note) -> Dict:
    return {
        "pair_id": pair_id,
        "candidate_source": source,
        "title_a": a_title,
        "title_b": b_title,
        "store_a": a_store,
        "store_b": b_store,
        "price_a": a_price,
        "price_b": b_price,
        "price_ratio": _ratio(a_price, b_price),
        "google_cluster_a": a,
        "google_cluster_b": b,
        "context": note,
        # Deliberately blank -- filled in by the labeller, not by this script.
        "label": "",
        "justification": "",
        "difficulty": "",
    }


def build(shopping: List[dict], amazon: List[dict], max_per_source: int) -> List[Dict]:
    rng = random.Random(RANDOM_SEED)
    out: List[Dict] = []

    by_cluster: Dict[str, List[dict]] = {}
    for r in shopping:
        by_cluster.setdefault(r["cluster_id"], []).append(r)

    # 1. cross-merchant: Google says same product, different stores
    cross = []
    for cid, offers in by_cluster.items():
        for a, b in itertools.combinations(offers, 2):
            src = "cross_merchant" if a["store_name"] != b["store_name"] else "same_cluster_alt"
            cross.append(_row(f"X{len(cross):04d}", src, cid, cid,
                              a["offer_title"], b["offer_title"],
                              a["store_name"], b["store_name"], a["price"], b["price"],
                              f"Google grouped both under: {a.get('google_product_title')}"))
    rng.shuffle(cross)
    out += cross[:max_per_source]

    # 2. near-miss: same search query, DIFFERENT Google cluster -> adjacent models
    by_query: Dict[str, List[str]] = {}
    for cid, offers in by_cluster.items():
        by_query.setdefault(offers[0]["query"], []).append(cid)
    near = []
    for query, cids in by_query.items():
        for ca, cb in itertools.combinations(cids, 2):
            a, b = by_cluster[ca][0], by_cluster[cb][0]
            near.append(_row(f"N{len(near):04d}", "near_miss", ca, cb,
                             a["offer_title"], b["offer_title"],
                             a["store_name"], b["store_name"], a["price"], b["price"],
                             f"Different Google clusters from the same search {query!r}: "
                             f"{a.get('google_product_title')}  VS  {b.get('google_product_title')}"))
    rng.shuffle(near)
    out += near[:max_per_source]

    # 3. amazon same-query clusters
    by_amz: Dict[str, List[dict]] = {}
    for r in amazon:
        by_amz.setdefault(r["query_cluster"], []).append(r)
    amz = []
    for query, items in by_amz.items():
        picks = items[:8]
        for a, b in itertools.combinations(picks, 2):
            amz.append(_row(f"A{len(amz):04d}", "amazon_cluster", a["asin"], b["asin"],
                            a["title"], b["title"], "Amazon.in", "Amazon.in",
                            a.get("price"), b.get("price"),
                            f"Both returned by Amazon search {query!r}"))
    rng.shuffle(amz)
    out += amz[:max_per_source]

    # 4. cross-category easy negatives -- capped low on purpose
    queries = sorted(by_amz)
    easy = []
    if len(queries) >= 2:
        for _ in range(max_per_source):
            qa, qb = rng.sample(queries, 2)
            a, b = rng.choice(by_amz[qa]), rng.choice(by_amz[qb])
            easy.append(_row(f"E{len(easy):04d}", "cross_category", a["asin"], b["asin"],
                             a["title"], b["title"], "Amazon.in", "Amazon.in",
                             a.get("price"), b.get("price"),
                             f"Unrelated searches: {qa!r} vs {qb!r}"))
    out += easy[:max(6, max_per_source // 5)]

    for i, r in enumerate(out, 1):
        r["pair_id"] = f"P{i:04d}"
    return out


def main():
    ap = argparse.ArgumentParser(description="Build unlabelled candidate pairs for adjudication.")
    ap.add_argument("--shopping", nargs="*", default=["data/shopping_offer_clusters.jsonl",
                                                     "data/shopping_offer_clusters_b.jsonl"])
    ap.add_argument("--amazon", default="data/amazon_in_listings.jsonl")
    ap.add_argument("--out", default="data/label_candidates.csv")
    ap.add_argument("--max-per-source", type=int, default=70)
    args = ap.parse_args()

    shopping: List[dict] = []
    for path in args.shopping:
        rows = load_jsonl(path)
        # cluster ids repeat across files; namespace them by filename
        tag = os.path.splitext(os.path.basename(path))[0][-1]
        for r in rows:
            r["cluster_id"] = f"{r['cluster_id']}{tag}"
        shopping += rows
        print(f"  {path}: {len(rows)} offers")
    amazon = load_jsonl(args.amazon)
    print(f"  {args.amazon}: {len(amazon)} listings")

    rows = build(shopping, amazon, args.max_per_source)

    import csv
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    import collections
    print(f"\n{len(rows)} candidate pairs -> {args.out}")
    for src, n in collections.Counter(r["candidate_source"] for r in rows).most_common():
        print(f"  {src:<20} {n:>4}")
    print("\nlabel / justification / difficulty are BLANK by design -- to be adjudicated.")


if __name__ == "__main__":
    main()
