"""
build_deploy.py
===============
Assembles deploy/ -- a self-contained folder holding ONLY what is needed to
serve the model, plus the checkpoint and a smoke test.

    python build_deploy.py            # code only, reuses an existing model/
    python build_deploy.py --model    # also copy the 541 MB checkpoint

WHY A BUILDER AND NOT A HAND-COPIED FOLDER
------------------------------------------
The serving import graph is not obvious. api/main.py reaches
ranking.ranker -> ranking.candidate_retrieval -> generate_relationship_pairs ->
data_quality.contradiction_rules AND product_taxonomy, so a folder assembled by
eye silently misses files and fails at the first /compare rather than at start.
This script copies the closure and then IMPORTS the result in a subprocess to
prove it, so a broken deploy folder fails here instead of in production.

The layout deliberately mirrors the repo root. config.ROOT_DIR is the directory
config.py sits in, and api/main.py resolves the checkpoint at
ROOT_DIR/trained_model_real -- flattening or nesting the tree would break that
path in a way that only shows up at runtime.
"""

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(ROOT, "deploy")

# Exact closure of the serving path. Verified by import, not by inspection.
MODULES = [
    "config.py",
    "utils.py",
    "generate_relationship_pairs.py",   # ranking.candidate_retrieval needs categorize()
    "product_taxonomy.py",              # generate_relationship_pairs needs it
]
PACKAGES = {
    "api": None,                        # None = whole package
    "exact_match": ["__init__.py", "inference.py", "preprocessing.py"],
    "ranking": ["__init__.py", "ranker.py", "candidate_retrieval.py",
                "embedding_retrieval.py"],
    "data_quality": ["__init__.py", "contradiction_rules.py"],
}
DEFAULT_MODEL_SRC = os.path.join(ROOT, "trained_model_real")
# The destination name is fixed because api/main.py resolves the checkpoint at
# ROOT_DIR/trained_model_real. Which checkpoint goes in there is chosen by
# --from, so shipping a new model never means overwriting the previous one --
# v7 stays intact on disk while v11 deploys.
MODEL_DST = os.path.join(DEPLOY, "trained_model_real")

REQUIREMENTS = """\
# Serving only. Training/research dependencies are deliberately absent.
torch>=2.1.0
transformers==5.14.1
safetensors>=0.4.0
sentencepiece>=0.2.0
protobuf>=4.25.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.4.0
fastapi>=0.110.0
uvicorn>=0.29.0

# Only needed for POST /search (catalog retrieval). /compare works without them.
# faiss-cpu>=1.8.0
# sentence-transformers>=3.0.0
"""

RUN_SH = """\
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
"""

RUN_PS1 = """\
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$port = if ($env:PORT) { $env:PORT } else { "8000" }
uvicorn api.main:app --host 0.0.0.0 --port $port
"""


def copy_tree(src, dst, only=None):
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        s = os.path.join(src, name)
        if os.path.isdir(s):
            if name == "__pycache__":
                continue
            if only is None:
                copy_tree(s, os.path.join(dst, name))
            continue
        if not name.endswith(".py"):
            continue
        if name.startswith("test_") or name == "test_ranking.py":
            continue                      # tests are not part of a deployment
        if only is not None and name not in only:
            continue
        shutil.copy2(s, os.path.join(dst, name))


def verify(with_model: bool) -> bool:
    """Import the assembled folder in a clean subprocess.

    Importing here in-process would prove nothing: this interpreter already has
    the repo on sys.path, so a missing file would resolve against the original.
    """
    probe = (
        "import api.main, ranking.ranker, exact_match.inference; "
        "print('imports OK')"
    )
    env = dict(os.environ, PYTHONPATH=DEPLOY, PYTHONIOENCODING="utf-8")
    r = subprocess.run([sys.executable, "-c", probe], cwd=DEPLOY, env=env,
                       capture_output=True, text=True)
    print("  " + (r.stdout.strip() or r.stderr.strip().splitlines()[-1:] or [""])[0]
          if not r.stdout.strip() else "  " + r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr)
        return False

    if with_model:
        import json
        meta = json.load(open(os.path.join(MODEL_DST, "training_metadata.json"),
                              encoding="utf-8"))
        ok = meta.get("serialization") == "colval" and meta.get("num_labels") == 2
        print(f"  checkpoint serialization={meta.get('serialization')} "
              f"threshold={meta.get('inference_threshold')} "
              f"num_labels={meta.get('num_labels')} -> {'OK' if ok else 'BAD'}")
        return ok
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true",
                    help="Also copy the checkpoint (~541 MB).")
    ap.add_argument("--from", dest="model_src", default=DEFAULT_MODEL_SRC,
                    help="Checkpoint directory to deploy. Defaults to "
                         "trained_model_real/.")
    ap.add_argument("--clean", action="store_true", help="Remove deploy/ first.")
    args = ap.parse_args()

    if args.clean and os.path.isdir(DEPLOY):
        shutil.rmtree(DEPLOY)
    os.makedirs(DEPLOY, exist_ok=True)

    print("copying modules:")
    for m in MODULES:
        shutil.copy2(os.path.join(ROOT, m), os.path.join(DEPLOY, m))
        print(f"  {m}")
    for pkg, only in PACKAGES.items():
        copy_tree(os.path.join(ROOT, pkg), os.path.join(DEPLOY, pkg), only)
        print(f"  {pkg}/")

    for name, body in (("requirements.txt", REQUIREMENTS),
                       ("run.sh", RUN_SH), ("run.ps1", RUN_PS1)):
        with open(os.path.join(DEPLOY, name), "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
    shutil.copy2(os.path.join(ROOT, "deploy_smoke_test.py"),
                 os.path.join(DEPLOY, "smoke_test.py"))
    shutil.copy2(os.path.join(ROOT, "deploy_README.md"),
                 os.path.join(DEPLOY, "README.md"))
    print("  requirements.txt, run.sh, run.ps1, smoke_test.py, README.md")

    if args.model:
        src = os.path.abspath(args.model_src)
        if not os.path.isfile(os.path.join(src, "config.json")):
            raise SystemExit(f"{src} is not a checkpoint directory (no config.json).")
        print(f"copying checkpoint ({src} -> trained_model_real/) ...")
        if os.path.isdir(MODEL_DST):
            shutil.rmtree(MODEL_DST)
        shutil.copytree(src, MODEL_DST)

    has_model = os.path.isdir(MODEL_DST)
    print("\nverifying:")
    ok = verify(has_model)
    if not has_model:
        print("  NOTE: no checkpoint in deploy/. Re-run with --model, or set MODEL_DIR.")

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(DEPLOY) for f in fs)
    print(f"\ndeploy/ is {total / 1e6:.1f} MB")
    print("OK" if ok else "*** VERIFICATION FAILED ***")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
