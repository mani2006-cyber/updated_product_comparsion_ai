"""
Kaggle job: does a per-category decision threshold beat the single global
one, for the shipped v11 checkpoint?

Pushed with `kaggle kernels push -p kaggle_job_category`. Runs headless on a
T4/P100, so everything it needs must be derivable from the repo plus the two
attached datasets -- there is no interactive step.

Moved here from a local CPU run (see calibrate_category_threshold.py commit
message) purely for speed: scoring ~30k pairs (validation + all six test
benchmarks) took long enough on CPU that a GPU pass finishes before the CPU
run would have gotten through half the splits.
"""
import os
import subprocess
import sys

REPO = "https://github.com/mani2006-cyber/updated_product_comparsion_ai.git"
WORK = "/kaggle/working"
SRC = f"{WORK}/pc/product_compariosn-main"


def sh(cmd, cwd=None):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"FAILED ({r.returncode}): {cmd}")


sh(f"git clone --depth 1 {REPO} {WORK}/pc")

# Environment fixes learned the hard way in kaggle_job/run.py (see its git
# history for the full story): P100s need a cu118 torch build; transformers
# 5.14.1 needs that build to be >=2.4; peft and torchvision both poison
# transformers' lazy module loader when present against this combination,
# and neither is used by anything in this repo's calibration path.
sh("pip install --no-cache-dir -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118",
   cwd=SRC)
sh("pip uninstall -y -q torchvision torchaudio", cwd=SRC)
sh("python -c \"import torch; "
   "print('torch', torch.__version__, "
   "'| cuda available:', torch.cuda.is_available(), "
   "'| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); "
   "x = torch.randn(4, 4, device='cuda'); y = x @ x; torch.cuda.synchronize(); "
   "print('GPU matmul OK:', y.shape)\"",
   cwd=SRC)
sh("pip install --no-cache-dir -q transformers==5.14.1", cwd=SRC)
sh("pip install --no-cache-dir -q -r requirements.txt --no-warn-conflicts", cwd=SRC)
sh("pip uninstall -y -q peft", cwd=SRC)

sh(f"mkdir -p {WORK}/wdc && find /kaggle/input -iname 'wdcproducts*' "
   f"-exec cp {{}} {WORK}/wdc/ \\;")

# The v11 checkpoint dataset ships as model.zip (it's the same
# trained_model_desc20_v11.zip already in the repo, just uploaded separately
# since checkpoints are gitignored). Unzip once into a plain directory.
sh(f"mkdir -p {WORK}/v11 && find /kaggle/input -iname 'model.zip' "
   f"-exec unzip -oq {{}} -d {WORK}/v11 \\;")
sh(f"ls -la {WORK}/v11")

# The exported checkpoint carries only tokenizer.json + tokenizer_config.json
# (no spm.model -- save_model.py's export never included it). Loading that
# tokenizer.json failed here with 'Couldn't instantiate the backend
# tokenizer ... You need to have sentencepiece ... installed', even with
# sentencepiece installed, because there is no spm.model locally to convert
# from either. The vocabulary is identical across every deberta-v3-small
# checkpoint this project has trained -- fine-tuning changes weights, not
# tokenization -- so fetching a fresh tokenizer for the base model and
# dropping it into the checkpoint dir is exact, not approximate. Only
# tokenizer files are written; config.json / model.safetensors (the actual
# weights) are untouched.
sh(f"python -c \"from transformers import AutoTokenizer; "
   f"AutoTokenizer.from_pretrained('microsoft/deberta-v3-small').save_pretrained('{WORK}/v11')\"",
   cwd=SRC)

# Regenerate data/real_corpus_valid.csv exactly as v11 was calibrated on.
# mine_hard_negatives.py (v11's actual training recipe) only augments the
# TRAIN split -- "Validation is unchanged" is printed by that script itself
# -- so the plain build below reproduces the same 10,609-pair validation set
# v11 was calibrated against, without needing to redo hard-negative mining.
sh(f"python build_real_corpus.py --wdc {WORK}/wdc --wdc-size large", cwd=SRC)

sh(f"python calibrate_category_threshold.py --model {WORK}/v11 --wdc {WORK}/wdc "
   f"--min-positives 20", cwd=SRC)

print("\nDONE. Copy the POOLED TEST F1 block above -- that delta is the whole answer.")
