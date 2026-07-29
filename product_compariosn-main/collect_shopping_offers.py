"""
collect_shopping_offers.py
==========================
Collects CROSS-MERCHANT product offers from Google Shopping (via the RapidAPI
"real-time-product-search" endpoint) to build candidate match pairs from real
Indian listings.

WHY THIS SOURCE
---------------
Product matching needs the same product described differently by different
sellers. Amazon search alone cannot supply that: one ASIN has one title. Google
Shopping groups offers for a product across merchants, so a single request
returns several independent descriptions of the same item:

    bigbasket.com  Rs.819    boAt Airdopes 141 TWS Earphones - Bold Black 1 pc
    Zepto          Rs.999    boAt Airdopes 141/8 TWS Earbuds with mic, 42H Battery

That pair is exactly the signal a rule-based generator can never invent, and
it is why the model trained on synthetic data scored 45.9 F1 on real pairs
while the same model trained on 6,000 real ones scored 75.6.

WHY THE OUTPUT IS "CANDIDATES", NOT LABELS
------------------------------------------
Google's clustering is wrong often enough to matter. The same probe returned:

    Flipkart       Rs.1,505  Thirty First Earbuds For Boat 141 Airdopes Wireless

"Thirty First ... For Boat 141" is a third-party item COMPATIBLE WITH the boAt,
at nearly double the price -- the same `for <device>` accessory pattern that
contaminated the Step 4 generation. Google grouped it anyway.

So `same_cluster` below records Google's opinion and nothing more. Treating it
as truth would rebuild the failure this project just escaped: labels produced
by a machine rule, learned faithfully, and meaningless. Every pair still needs
adjudication.

BUDGET
------
Costs 1 request per search plus 1 per product expanded, so budget is explicit
and enforced. The free tier rate-limits hard: a 1.2s delay produced 22
consecutive 429s on a previous run, and each failure still consumes quota.
Default delay is therefore deliberately conservative.

CREDENTIALS
-----------
Read from RAPIDAPI_KEY; never stored in this file.

Usage:
    python collect_shopping_offers.py --budget 30
    python collect_shopping_offers.py --budget 12 --queries "Noise ColorFit Pulse 2"
"""

import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from typing import Dict, List

API_HOST = "real-time-product-search.p.rapidapi.com"
OUT_PATH = "data/shopping_offer_clusters.jsonl"

# Specific products, so each search returns a tight cluster of lookalikes.
DEFAULT_QUERIES: List[str] = [
    "boAt Airdopes 141", "Noise Buds VS104", "realme Buds T100",
    "OnePlus Bullets Z2", "JBL Tune 215BT", "pTron Bassbuds",
    "Noise ColorFit Pulse 2", "boAt Wave Call smartwatch",
    "Fire-Boltt Ninja Call Pro", "Redmi Note 13 5G", "Samsung Galaxy M14 5G",
    "vivo T2 5G", "Prestige mixer grinder 750W", "Philips air fryer",
    "Bajaj electric kettle", "SanDisk Ultra 128GB pendrive",
    "Nike Revolution 6 men", "Adidas Runfalcon 3", "Milton water bottle",
    "Mamaearth face wash",
]


def _get(path: str, params: Dict[str, str], key: str) -> Dict:
    url = f"https://{API_HOST}/{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url, headers={"x-rapidapi-key": key, "x-rapidapi-host": API_HOST})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser(description="Collect cross-merchant offer clusters.")
    ap.add_argument("--budget", type=int, required=True,
                    help="Hard ceiling on API requests (searches + expansions).")
    ap.add_argument("--queries", nargs="*", default=None)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--country", default="in")
    ap.add_argument("--per-query", type=int, default=4,
                    help="Products expanded per search. Each costs one request.")
    ap.add_argument("--sleep", type=float, default=6.0,
                    help="Seconds between calls. 1.2s previously triggered sustained 429s.")
    args = ap.parse_args()

    key = os.environ.get("RAPIDAPI_KEY")
    if not key:
        raise SystemExit("RAPIDAPI_KEY is not set; it is deliberately not stored in this file.")

    queries = args.queries or DEFAULT_QUERIES
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    used, clusters, offers_total = 0, 0, 0

    with open(args.out, "w", encoding="utf-8") as fh:
        for query in queries:
            if used >= args.budget:
                break
            try:
                found = _get("search", {"q": query, "country": args.country,
                                        "language": "en", "limit": "10"}, key)
                used += 1
            except Exception as exc:  # noqa: BLE001
                used += 1             # failed calls still consume quota
                print(f"  search {query!r} FAILED ({exc}) -- used {used}")
                time.sleep(args.sleep)
                continue

            products = found.get("data", {}).get("products", []) or []
            # Only products Google says have several sellers can yield a
            # cross-merchant pair; the rest would waste a request.
            multi = [p for p in products if p.get("has_multiple_offers")][:args.per_query]
            print(f"search {query!r}: {len(products)} products, {len(multi)} with multiple offers "
                  f"-- used {used}")
            time.sleep(args.sleep)

            for product in multi:
                if used >= args.budget:
                    break
                try:
                    detail = _get("product-offers",
                                  {"product_id": product["product_id"],
                                   "country": args.country, "language": "en",
                                   "limit": "20"}, key)
                    used += 1
                except Exception as exc:  # noqa: BLE001
                    used += 1
                    print(f"    offers FAILED ({exc}) -- used {used}")
                    time.sleep(args.sleep)
                    continue

                data = detail.get("data")
                offer_list = data if isinstance(data, list) else (data or {}).get("offers", [])
                offer_list = [o for o in (offer_list or []) if o.get("offer_title")]
                if len(offer_list) < 2:
                    time.sleep(args.sleep)
                    continue

                clusters += 1
                for o in offer_list:
                    fh.write(json.dumps({
                        "cluster_id": f"g{clusters:04d}",
                        # Google's grouping. An OPINION to be adjudicated, not a label.
                        "same_cluster_per_google": True,
                        "google_product_title": product.get("product_title"),
                        "query": query,
                        "offer_title": o.get("offer_title"),
                        "store_name": o.get("store_name"),
                        "price": o.get("price"),
                        "original_price": o.get("original_price"),
                        "product_condition": o.get("product_condition"),
                        "offer_page_url": o.get("offer_page_url"),
                    }, ensure_ascii=False) + "\n")
                    offers_total += 1
                print(f"    + cluster {clusters} ({len(offer_list)} offers) "
                      f"{str(product.get('product_title'))[:52]} -- used {used}")
                time.sleep(args.sleep)

    print(f"\n{clusters} clusters / {offers_total} offers -> {args.out}  ({used} requests used)")
    print("NOTE: same_cluster_per_google is Google's opinion. Adjudicate before training.")


if __name__ == "__main__":
    main()
