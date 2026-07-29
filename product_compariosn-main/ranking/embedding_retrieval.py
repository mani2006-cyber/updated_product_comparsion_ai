"""
ranking/embedding_retrieval.py
==============================
Stage 1 of retrieve-then-rerank: embedding + FAISS search over a product
catalog, so the cross-encoder only ever scores a shortlist.

WHY THIS EXISTS
---------------
`candidate_retrieval.py` filters a pool the CALLER already supplies, using a
keyword category heuristic and a linear scan. That answers "which of these 15
products matches?" but not "find the match in my catalog of 500,000" -- which
is the actual product question. The cross-encoder cannot answer it either: at
~5-10ms per pair on GPU, scoring a 500k catalog is over an hour per query.

So: a bi-encoder embeds every product ONCE, offline. At query time FAISS finds
the ~50 nearest neighbours in milliseconds, and the cross-encoder re-ranks just
those. This is the standard architecture behind every production matcher, and
it is what the existing cross-encoder is actually for.

TWO MODELS, TWO TEXT FORMATS -- BOTH CORRECT
--------------------------------------------
The cross-encoder is fed `COL brand VAL ... COL title VAL ...` because that is
what it was fine-tuned on; feeding it anything else silently collapsed match
recall to zero once already. The bi-encoder here is an off-the-shelf sentence
transformer that never saw that convention, so it gets plain natural text.
These formats MUST differ. Do not "unify" them.

EXACT SEARCH BY DEFAULT
-----------------------
IndexFlatIP is exact, needs no training, and has no recall loss. On 500k x 384
floats it answers in tens of milliseconds -- fast enough that approximation
would trade correctness for a speedup nobody has yet measured a need for.
`--approx` switches to IVF for catalogs where that stops being true.

Vectors are L2-normalised, so inner product IS cosine similarity.

Usage:
    # build an index from a catalog
    python -m ranking.embedding_retrieval build \\
        --catalog data/amazon_in_listings.jsonl --out data/product_index

    # measure recall@k before trusting it
    python -m ranking.embedding_retrieval evaluate \\
        --index data/product_index --clusters data/shopping_offer_clusters.jsonl
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np

# Small, fast, widely used. 384-dim, ~80MB. Stronger retrievers exist
# (BAAI/bge-small-en-v1.5, gte-small); this is deliberately the conservative
# default so retrieval quality can be MEASURED before anything fancier is
# justified. Swap with --model.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_embedding_text(product: Dict) -> str:
    """Plain natural text for the bi-encoder -- NOT the cross-encoder's
    COL/VAL layout. See the module docstring: the two models are trained on
    different conventions and must each get their own."""
    parts = [str(product.get("title") or "").strip()]
    brand = str(product.get("brand") or "").strip()
    if brand and brand.lower() not in parts[0].lower():
        parts.append(brand)
    description = str(product.get("description") or "").strip()
    if description:
        parts.append(description[:300])
    return " ".join(p for p in parts if p)


class ProductIndex:
    """Embeds a catalog and answers nearest-neighbour queries."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: Optional[str] = None):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.index = None
        self.products: List[Dict] = []

    # ---- build / persist -------------------------------------------------
    def build(self, products: List[Dict], batch_size: int = 128) -> "ProductIndex":
        import faiss

        self.products = list(products)
        texts = [build_embedding_text(p) for p in self.products]
        vectors = self.model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=True,
        ).astype("float32")

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        return self

    def save(self, out_dir: str) -> str:
        import faiss

        os.makedirs(out_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(out_dir, "index.faiss"))
        with open(os.path.join(out_dir, "products.jsonl"), "w", encoding="utf-8") as fh:
            for p in self.products:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
            # The model name is stored because querying with a DIFFERENT
            # encoder than the index was built with returns plausible-looking
            # nonsense rather than an error -- the same class of silent failure
            # as the cross-encoder serialization mismatch.
            json.dump({"model_name": self.model_name, "size": len(self.products)}, fh, indent=2)
        return out_dir

    @classmethod
    def load(cls, index_dir: str, device: Optional[str] = None) -> "ProductIndex":
        import faiss

        with open(os.path.join(index_dir, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        obj = cls(model_name=meta["model_name"], device=device)
        obj.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "products.jsonl"), encoding="utf-8") as fh:
            obj.products = [json.loads(line) for line in fh if line.strip()]
        return obj

    # ---- query -----------------------------------------------------------
    def search(self, query: Dict, k: int = 50, exclude_self: bool = True) -> List[Dict]:
        """Returns the k nearest catalog products, each with `_retrieval_score`."""
        vector = self.model.encode([build_embedding_text(query)], convert_to_numpy=True,
                                   normalize_embeddings=True).astype("float32")
        # Over-fetch so that dropping the query itself cannot shrink the
        # shortlist below k.
        scores, ids = self.index.search(vector, min(k + 1, len(self.products)))

        out = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            product = self.products[idx]
            if exclude_self and query.get("id") is not None and product.get("id") == query.get("id"):
                continue
            enriched = dict(product)
            enriched["_retrieval_score"] = float(score)
            out.append(enriched)
            if len(out) >= k:
                break
        return out


def load_catalog(path: str) -> List[Dict]:
    """Reads a catalog from JSONL or CSV, normalising to id/title/brand/description."""
    products: List[Dict] = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as fh:
            raw = [json.loads(line) for line in fh if line.strip()]
    else:
        import pandas as pd
        raw = pd.read_csv(path).to_dict("records")

    for i, r in enumerate(raw):
        title = r.get("title") or r.get("product_title") or r.get("offer_title") or ""
        if not str(title).strip():
            continue
        products.append({
            "id": r.get("id") or r.get("asin") or r.get("product_id") or f"P{i:07d}",
            "title": str(title),
            "brand": str(r.get("brand") or r.get("store_name") or ""),
            "description": str(r.get("description") or ""),
        })
    return products


def evaluate_recall(index: ProductIndex, clusters_path: List[str], ks=(1, 5, 10, 20, 50)) -> None:
    """Recall@k using Google Shopping clusters as ground truth.

    Retrieval recall is the ceiling on the whole system: a true match the
    retriever never surfaces is one the cross-encoder can never score, no
    matter how accurate it is. Measuring this is not optional.

    Google's clustering is imperfect (it grouped a third-party 'For Boat 141'
    accessory with the real product), so treat these numbers as approximate --
    they understate true recall wherever Google was wrong.
    """
    import collections

    by_cluster = collections.defaultdict(list)
    for path in clusters_path:
        if not os.path.exists(path):
            continue
        tag = os.path.basename(path)[-7]
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                by_cluster[f"{r['cluster_id']}{tag}"].append(r)

    probes = [(offers[0], offers[1:]) for offers in by_cluster.values() if len(offers) > 1]
    if not probes:
        print("No multi-offer clusters found; cannot evaluate recall.")
        return

    titles_in_index = {p["title"] for p in index.products}
    hits = {k: 0 for k in ks}
    evaluated = 0

    for query, siblings in probes:
        gold = {s["offer_title"] for s in siblings if s["offer_title"] in titles_in_index}
        if not gold:
            continue                       # siblings absent from the catalog
        evaluated += 1
        found = index.search({"id": None, "title": query["offer_title"],
                              "brand": query.get("store_name", "")}, k=max(ks))
        ranked = [p["title"] for p in found]
        for k in ks:
            if gold & set(ranked[:k]):
                hits[k] += 1

    print(f"\nrecall@k over {evaluated} cluster probes "
          f"(catalog size {len(index.products):,}):")
    for k in ks:
        print(f"  recall@{k:<3} {hits[k] / max(evaluated, 1):6.1%}")
    print("\nRecall@k caps the whole pipeline: anything not retrieved is never re-ranked.")


def evaluate_pairs(index: ProductIndex, ground_truth_csv: str, ks=(1, 5, 10, 20, 50, 100),
                   sample: int = 2000, seed: int = 20260730) -> None:
    """Recall@k against known matching id pairs, at catalog scale.

    Why this matters more than the cluster evaluation: recall@50 was 100% on a
    406-product catalog, but retrieving 50 items there means returning 12% of
    everything. That number says nothing about production. Here the catalog is
    ~113k, so top-50 is 0.04% of it -- a genuine test of whether the embedding
    stage can find a needle.

    Recall@k is the ceiling on the whole pipeline: a true match the retriever
    never surfaces is one the cross-encoder never gets to score.
    """
    import csv
    import random

    by_id = {p["id"]: i for i, p in enumerate(index.products)}
    pairs = []
    with open(ground_truth_csv, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            a, b = row["product1_id"], row["product2_id"]
            if a in by_id and b in by_id and a != b:
                pairs.append((a, b))

    if not pairs:
        print("No usable ground-truth pairs found.")
        return

    rng = random.Random(seed)
    rng.shuffle(pairs)
    probes = pairs[:sample]

    hits = {k: 0 for k in ks}
    max_k = max(ks)
    for a, b in probes:
        query = index.products[by_id[a]]
        found = index.search(query, k=max_k)
        ranked = [p.get("id") for p in found]
        for k in ks:
            if b in ranked[:k]:
                hits[k] += 1

    print(f"\nrecall@k over {len(probes):,} probes | catalog {len(index.products):,} products")
    print(f"(top-{max_k} is {max_k / len(index.products):.3%} of the catalog)")
    for k in ks:
        print(f"  recall@{k:<4} {hits[k] / len(probes):6.1%}")
    print("\nThis caps the pipeline: whatever is not retrieved is never re-ranked.")


def main():
    ap = argparse.ArgumentParser(description="Embedding + FAISS retrieval over a product catalog.")
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="Embed a catalog and write a FAISS index.")
    b.add_argument("--catalog", required=True, help="JSONL or CSV of products")
    b.add_argument("--out", default="data/product_index")
    b.add_argument("--model", default=DEFAULT_MODEL)

    e = sub.add_parser("evaluate", help="Measure recall@k against known clusters.")
    e.add_argument("--index", default="data/product_index")
    e.add_argument("--clusters", nargs="+",
                   default=["data/shopping_offer_clusters.jsonl",
                            "data/shopping_offer_clusters_b.jsonl"])

    ep = sub.add_parser("evaluate-pairs", help="Recall@k against known matching id pairs.")
    ep.add_argument("--index", default="data/product_index")
    ep.add_argument("--ground-truth", default="data/scale_ground_truth.csv")
    ep.add_argument("--sample", type=int, default=2000)

    q = sub.add_parser("query", help="Ad-hoc nearest-neighbour lookup.")
    q.add_argument("--index", default="data/product_index")
    q.add_argument("--title", required=True)
    q.add_argument("-k", type=int, default=10)

    args = ap.parse_args()

    if args.command == "build":
        products = load_catalog(args.catalog)
        print(f"catalog: {len(products):,} products from {args.catalog}")
        index = ProductIndex(model_name=args.model).build(products)
        index.save(args.out)
        print(f"index written to {args.out} (model {args.model})")

    elif args.command == "evaluate":
        index = ProductIndex.load(args.index)
        evaluate_recall(index, args.clusters)

    elif args.command == "evaluate-pairs":
        index = ProductIndex.load(args.index)
        evaluate_pairs(index, args.ground_truth, sample=args.sample)

    else:
        index = ProductIndex.load(args.index)
        for p in index.search({"id": None, "title": args.title}, k=args.k):
            print(f"  {p['_retrieval_score']:.3f}  {p['title'][:95]}")


if __name__ == "__main__":
    main()
