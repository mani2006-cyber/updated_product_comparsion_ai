"""
smoke_test.py
=============
Proves a running deployment is actually working, not merely up.

    python smoke_test.py                          # against http://localhost:8000
    python smoke_test.py --url http://host:8000

WHY EACH CHECK EXISTS
---------------------
The failure this deployment is most exposed to is a SERIALIZATION MISMATCH. The
model is trained on `COL brand VAL ... COL title VAL ...`; fed anything else it
does not error, it just stops recognising matches. Measured on this project:
three unmistakable matches scored 49.8% / 20.6% / 4.4% instead of ~100%, recall
on real matches was zero, and /health reported "ok" the whole time.

So a health check is not enough, and neither is a single match test:

  * a MATCH test catches the serialization fault (scores collapse), but passes
    trivially on a model that says "same" to everything
  * a NON-MATCH test catches an always-yes model, but a broken model passes it
    too -- broken means everything looks different

Both directions are required. Either alone can be green while the service is
useless.

Exit code is 0 only if every check passes, so this can gate a deploy.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

# (title, brand, description)
MATCH_A = ("boAt Airdopes 141 Elite ANC | 42H Earbuds with 35dB ANC Black",
           "boAt", "42 hours playtime, 35dB ANC, Bluetooth 5.3")
MATCH_B = ("boAt Airdopes 141 Elite ANC | Black",
           "boAt", "ANC true wireless earbuds")
DIFFERENT = ("Nike Air Max 270 Running Shoes Size 9",
             "Nike", "Mesh upper, air cushioning")
VARIANT = ("boAt Airdopes 141 Elite ANC | White",
           "boAt", "ANC true wireless earbuds")


def post(url, payload, timeout=120):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.load(r)


def get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, json.load(r)


def product(pid, t):
    return {"id": pid, "title": t[0], "brand": t[1], "description": t[2]}


def score_of(body, title):
    if body.get("exact_match") and body["exact_match"]["title"] == title:
        return body["exact_match"]["similarity_score"]
    for s in body.get("similar_products", []):
        if s["title"] == title:
            return s["similarity_score"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model-dir", default="trained_model_real",
                    help="Checked directly for the serialization field.")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    checks = []

    def check(name, passed, detail=""):
        checks.append((name, bool(passed), detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    # ---- 0. the checkpoint on disk ---------------------------------------
    print("\ncheckpoint metadata")
    meta_path = os.path.join(args.model_dir, "training_metadata.json")
    try:
        meta = json.load(open(meta_path, encoding="utf-8"))
        check("serialization == colval", meta.get("serialization") == "colval",
              f"got {meta.get('serialization')!r}")
        check("num_labels == 2", meta.get("num_labels") == 2,
              f"got {meta.get('num_labels')}")
        check("inference_threshold present", meta.get("inference_threshold") is not None,
              f"got {meta.get('inference_threshold')}")
        # The description budget must match what this checkpoint TRAINED on, not
        # whatever config currently says. v7 predates the field and must resolve
        # to 100; a checkpoint trained after the 20-word change records 20.
        # Serving the wrong budget costs accuracy without raising.
        budget = meta.get("description_words", "absent -> 100 (legacy)")
        print(f"  [INFO] description budget: {budget}")
    except FileNotFoundError:
        check("training_metadata.json readable", False, f"missing at {meta_path}")

    # ---- 1. health --------------------------------------------------------
    print("\nGET /health")
    try:
        _, h = get(f"{base}/health")
        print("  " + json.dumps(h))
        check("model_loaded is true", h.get("model_loaded") is True)
        check("num_labels == 2", h.get("num_labels") == 2, f"got {h.get('num_labels')}")
    except urllib.error.URLError as e:
        check("service reachable", False, str(e))
        print("\nService is not reachable. Start it with ./run.sh (or run.ps1) first.")
        sys.exit(1)

    # ---- 2. the match direction ------------------------------------------
    # A true match must score high. This is the check that catches a
    # serialization mismatch: the score collapses instead of erroring.
    print("\nPOST /compare  -- true match (must score HIGH)")
    _, body = post(f"{base}/compare", {
        "product": product("P1", MATCH_A),
        "candidates": [product("P2", MATCH_B)],
    })
    s = score_of(body, MATCH_B[0])
    print("  " + json.dumps(body)[:300])
    check("true match scores > 90%", s is not None and s > 90.0, f"got {s}")

    # ---- 3. the non-match direction --------------------------------------
    # Guards against a model that says "same" to everything, which would sail
    # through check 2.
    print("\nPOST /compare  -- unrelated product (must score LOW)")
    _, body = post(f"{base}/compare", {
        "product": product("P1", MATCH_A),
        "candidates": [product("P3", DIFFERENT)],
    })
    s_diff = score_of(body, DIFFERENT[0])
    print("  " + json.dumps(body)[:300])
    # Absent is a PASS, and a stronger one than a low score: the ranker drops
    # candidates it judges unrelated instead of returning them with a small
    # number, so an unrelated product legitimately never appears in the
    # response at all.
    check("unrelated absent or scores < 50%",
          s_diff is None or s_diff < 50.0,
          "filtered out of the response" if s_diff is None else f"got {s_diff}")

    # ---- 4. ranking order -------------------------------------------------
    # With all three candidates at once, the true match must outrank both the
    # colour variant and the unrelated product. Tests the ranker, not just the
    # scorer.
    print("\nPOST /compare  -- three candidates (ordering)")
    _, body = post(f"{base}/compare", {
        "product": product("P1", MATCH_A),
        "candidates": [product("P3", DIFFERENT), product("P4", VARIANT),
                       product("P2", MATCH_B)],
        "top_n": 3,
    })
    print("  " + json.dumps(body)[:400])
    s_match = score_of(body, MATCH_B[0])
    s_var = score_of(body, VARIANT[0])
    s_un = score_of(body, DIFFERENT[0])
    check("true match outranks unrelated",
          s_match is not None and (s_un is None or s_match > s_un),
          f"{s_match} vs " + ("filtered out" if s_un is None else str(s_un)))
    if s_var is not None:
        check("true match outranks colour variant", s_match > s_var,
              f"{s_match} vs {s_var}")

    # ---- 4b. nothing may be silently dropped -----------------------------
    # The ranker used to keep ONE match and discard every other candidate it
    # also judged the same product. Measured live on three Garnier listings at
    # 99.9878 / 99.9853 / 99.9848: one returned, two deleted, similar_products
    # empty. That destroys the core use case, since several merchants listing
    # one product is normal for price comparison. Counting candidates in vs out
    # is the only check that catches it -- every other assertion passed.
    print("\nPOST /compare  -- candidate conservation (3 near-identical matches)")
    _, body = post(f"{base}/compare", {
        "product": product("P1", MATCH_A),
        "candidates": [product("P2", MATCH_B), product("P4", VARIANT),
                       product("P5", ("boAt Airdopes 141 Elite ANC | 42H, 35dB ANC",
                                      "boAt", "ANC earbuds"))],
        "top_n": 10,
    })
    returned = (1 if body.get("exact_match") else 0) \
        + len(body.get("other_matches", [])) + len(body.get("similar_products", []))
    print(f"  in=3  out={returned}  "
          f"(exact_match + {len(body.get('other_matches', []))} other_matches "
          f"+ {len(body.get('similar_products', []))} similar)")
    check("no candidate is silently dropped", returned == 3, f"3 in, {returned} out")
    check("other_matches field present", "other_matches" in body)

    # ---- 5. input validation ---------------------------------------------
    print("\nPOST /compare  -- empty candidates (must be rejected)")
    try:
        st, _ = post(f"{base}/compare",
                     {"product": product("P1", MATCH_A), "candidates": []})
        check("empty candidate list rejected", False, f"accepted with {st}")
    except urllib.error.HTTPError as e:
        check("empty candidate list rejected", e.code == 422, f"HTTP {e.code}")

    # ---- summary ----------------------------------------------------------
    failed = [n for n, ok, _ in checks if not ok]
    print("\n" + "=" * 62)
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
        if any("colval" in f or "match scores" in f for f in failed):
            print("\nA collapsed match score with a healthy /health is the "
                  "signature of a serialization mismatch. Check that "
                  "training_metadata.json contains \"serialization\": \"colval\".")
    print("=" * 62)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
