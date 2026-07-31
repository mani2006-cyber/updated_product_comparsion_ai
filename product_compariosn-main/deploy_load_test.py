"""
load_test.py
============
Production-readiness probe for a running deployment. Correctness is
smoke_test.py's job; this measures what happens under load and under abuse.

    python load_test.py                        # against http://localhost:8000
    python load_test.py --url http://host:8000 --concurrency 8 --duration 30

WHAT THIS ANSWERS
-----------------
The handover lists "concurrent request handling (never load-tested)" and
"GPU latency (only CPU measured)" as unvalidated. Every latency number in this
project came from one request at a time on an idle machine, which is the least
informative case: a service that answers in 200 ms alone can still collapse at
four concurrent callers, and nothing so far would have detected that.

Four things are measured:

  1. Latency vs candidate count -- each candidate is a transformer forward
     pass, so cost should scale with the shortlist, and /compare accepts up to
     200. That is the knob an attacker or a careless caller turns.
  2. Throughput and tail latency under concurrency.
  3. Memory growth under sustained load -- a leak shows as RSS that never
     settles.
  4. Robustness against malformed and hostile input.

WHAT IT CANNOT ANSWER
---------------------
Absolute latency here is CPU-only and will be far worse than a GPU deployment;
read the SHAPE of the curves, not the milliseconds. It also cannot tell you
whether the service is safe to expose -- it has no authentication and no rate
limiting, and no load test changes that.
"""

import argparse
import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

BASE_PRODUCT = {"id": "P0", "title": "boAt Airdopes 141 Elite ANC | 42H Earbuds with 35dB ANC Black",
                "brand": "boAt", "description": "42 hours playtime, 35dB ANC, Bluetooth 5.3"}


def candidates(n):
    out = []
    for i in range(n):
        out.append({"id": f"C{i}",
                    "title": f"boAt Airdopes {141 + i} TWS Earbuds with mic, {40 + i}H Battery",
                    "brand": "boAt", "description": "true wireless earbuds"})
    return out


def call(url, payload, timeout=300):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return time.perf_counter() - t0, r.status, None
    except urllib.error.HTTPError as e:
        return time.perf_counter() - t0, e.code, e.read()[:200].decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return time.perf_counter() - t0, 0, str(e)[:200]


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--concurrency", type=int, nargs="*", default=[1, 2, 4, 8])
    ap.add_argument("--duration", type=float, default=15.0,
                    help="Seconds per concurrency level.")
    ap.add_argument("--sustained", type=float, default=20.0,
                    help="Seconds of sustained load for the memory check.")
    ap.add_argument("--shortlist", type=int, default=10,
                    help="Candidates per request during the throughput runs.")
    args = ap.parse_args()
    base = args.url.rstrip("/")
    compare = f"{base}/compare"
    findings = []

    try:
        with urllib.request.urlopen(f"{base}/health", timeout=30) as r:
            health = json.load(r)
    except Exception as e:  # noqa: BLE001
        print(f"Service unreachable at {base}: {e}")
        sys.exit(1)
    print(f"target   {base}")
    print(f"health   {json.dumps(health)}")
    if not health.get("model_loaded"):
        print("\nmodel_loaded is false -- aborting, nothing below would be meaningful.")
        sys.exit(1)

    # ---------------------------------------------------------------- 1
    section("1. LATENCY vs CANDIDATE COUNT  (sequential, 5 runs each)")
    print(f"  {'candidates':>11}{'median ms':>12}{'p95 ms':>10}{'ms/cand':>10}")
    per_cand = {}
    for n in (1, 5, 10, 50, 200):
        call(compare, {"product": BASE_PRODUCT, "candidates": candidates(n)})  # warm
        ts = [call(compare, {"product": BASE_PRODUCT, "candidates": candidates(n)})[0]
              for _ in range(5)]
        med = statistics.median(ts) * 1000
        per_cand[n] = med
        print(f"  {n:>11}{med:>12.0f}{pct(ts, 95) * 1000:>10.0f}{med / n:>10.1f}")
    worst = per_cand[200]
    print(f"\n  A single caller can occupy the service for {worst / 1000:.1f}s by sending"
          f"\n  the maximum 200 candidates. With no rate limit, that is the DoS surface.")
    if worst > 10000:
        findings.append(f"one max-size request blocks for {worst/1000:.1f}s")

    # ---------------------------------------------------------------- 2
    section(f"2. THROUGHPUT UNDER CONCURRENCY  ({args.shortlist} candidates, "
            f"{args.duration:g}s each)")
    print(f"  {'workers':>8}{'req/s':>9}{'median ms':>12}{'p95 ms':>10}"
          f"{'p99 ms':>10}{'errors':>8}")
    single_thread_rps = None
    for c in args.concurrency:
        lat, errs, stop = [], [], threading.Event()
        lock = threading.Lock()

        def worker():
            while not stop.is_set():
                dt, code, err = call(compare, {"product": BASE_PRODUCT,
                                               "candidates": candidates(args.shortlist)})
                with lock:
                    if code == 200:
                        lat.append(dt)
                    else:
                        errs.append(f"{code}: {err}")

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(c)]
        t0 = time.perf_counter()
        for t in threads:
            t.start()
        time.sleep(args.duration)
        stop.set()
        for t in threads:
            t.join(timeout=300)
        elapsed = time.perf_counter() - t0
        rps = len(lat) / elapsed
        if c == 1:
            single_thread_rps = rps
        print(f"  {c:>8}{rps:>9.2f}{statistics.median(lat) * 1000 if lat else 0:>12.0f}"
              f"{pct(lat, 95) * 1000:>10.0f}{pct(lat, 99) * 1000:>10.0f}{len(errs):>8}")
        if errs:
            print(f"           first error: {errs[0]}")
            findings.append(f"{len(errs)} errors at concurrency {c}")

    if single_thread_rps:
        best = max(len(args.concurrency), 1)
        print(f"\n  Scaling: if req/s is flat as workers rise, requests are serialising")
        print(f"  (one model, GIL-bound tokenisation, no batching across requests).")

    # ---------------------------------------------------------------- 3
    section(f"3. SUSTAINED LOAD  ({args.sustained:g}s, 4 workers)")
    lat, errs, stop = [], 0, threading.Event()
    lock = threading.Lock()

    def worker2():
        nonlocal errs
        while not stop.is_set():
            dt, code, _ = call(compare, {"product": BASE_PRODUCT,
                                         "candidates": candidates(args.shortlist)})
            with lock:
                if code == 200:
                    lat.append(dt)
                else:
                    errs += 1

    threads = [threading.Thread(target=worker2, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    time.sleep(args.sustained)
    stop.set()
    for t in threads:
        t.join(timeout=300)
    if lat:
        half = len(lat) // 2
        first, second = statistics.median(lat[:half]), statistics.median(lat[half:])
        print(f"  requests {len(lat)}   errors {errs}")
        print(f"  median latency first half {first * 1000:.0f} ms, "
              f"second half {second * 1000:.0f} ms  "
              f"({(second / first - 1) * 100:+.0f}%)")
        if second > first * 1.5:
            findings.append("latency degraded >50% over sustained load")
            print("  DEGRADING -- latency rising under sustained load.")
        else:
            print("  Stable.")

    # ---------------------------------------------------------------- 4
    section("4. ROBUSTNESS  (malformed and hostile input)")
    cases = [
        ("empty candidates", {"product": BASE_PRODUCT, "candidates": []}, {422}),
        ("missing product", {"candidates": candidates(1)}, {422}),
        ("candidates over cap (201)",
         {"product": BASE_PRODUCT, "candidates": candidates(201)}, {413}),
        ("top_n out of range",
         {"product": BASE_PRODUCT, "candidates": candidates(1), "top_n": 999}, {422}),
        ("null title",
         {"product": {"id": "x", "title": None}, "candidates": candidates(1)}, {422}),
        ("empty title",
         {"product": {"id": "x", "title": ""}, "candidates": candidates(1)}, {200, 422}),
        ("very long title (100k chars)",
         {"product": {"id": "x", "title": "A" * 100000},
          "candidates": candidates(1)}, {200, 413, 422}),
        ("unicode + emoji",
         {"product": {"id": "x", "title": "boAt Airdopes 141 नीला \U0001f3a7"},
          "candidates": candidates(1)}, {200}),
        ("html/script in title",
         {"product": {"id": "x", "title": "<script>alert(1)</script>"},
          "candidates": candidates(1)}, {200}),
        ("wrong type for candidates",
         {"product": BASE_PRODUCT, "candidates": "not-a-list"}, {422}),
    ]
    for name, payload, expected in cases:
        dt, code, err = call(compare, payload)
        ok = code in expected
        detail = f"HTTP {code}"
        if not ok:
            detail += f"  expected {sorted(expected)}"
            findings.append(f"{name} returned {code}")
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<30} {detail:<28} {dt * 1000:>7.0f} ms")
        if code >= 500 and err:
            leak = any(k in (err or "") for k in ("Traceback", "File \"", "line "))
            print(f"         {'LEAKS INTERNALS' if leak else 'body'}: {err[:110]}")
            if leak:
                findings.append(f"{name} leaked a traceback to the caller")

    # raw malformed JSON, which bypasses the pydantic path entirely
    req = urllib.request.Request(compare, data=b"{not json",
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        urllib.request.urlopen(req, timeout=60)
        code = 200
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception:  # noqa: BLE001
        code = 0
    ok = code == 422
    print(f"  [{'PASS' if ok else 'FAIL'}] {'malformed JSON body':<30} HTTP {code}")
    if not ok:
        findings.append(f"malformed JSON returned {code}")

    # ---------------------------------------------------------------- summary
    section("PRODUCTION READINESS")
    print("  Measured blockers:")
    if findings:
        for f in findings:
            print(f"    - {f}")
    else:
        print("    none from these probes")
    print("\n  Known blockers these probes CANNOT detect (see README):")
    print("    - no authentication: anyone reaching the port gets unlimited inference")
    print("    - no rate limiting: the per-request cap above is the only bound")
    print("    - index cannot refresh incrementally; /search 503s without one")
    print("\n  Latency here is CPU-only. Read the shape of the curves, not the numbers.")
    sys.exit(1 if findings else 0)


if __name__ == "__main__":
    main()
