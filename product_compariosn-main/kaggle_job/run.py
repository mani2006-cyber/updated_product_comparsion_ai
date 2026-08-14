"""
Kaggle job: does swap augmentation make the model order-invariant?

Pushed with `kaggle kernels push -p kaggle_job`. Runs headless on a T4, so
everything it needs must be derivable from the repo plus the attached WDC
dataset -- there is no interactive step.

TWO SEEDS, NOT ONE. v10 and v11 were the same recipe on the same corpus and
differed by 2.0 benchmark F1 and 3.5 Indian F1 on seed alone. A single run
cannot separate a recipe effect from a seed effect at that scale, so this
trains twice and reports both.
"""
import json
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

# Kaggle can hand out a Tesla P100 (compute capability sm_60, Pascal) instead
# of a T4/L4. The Kaggle base image's preinstalled torch is built for CUDA
# 12.x, whose wheels dropped kernel support for anything older than sm_70
# (Volta) -- training fails deep in the first forward pass with
# "CUDA error: no kernel image is available for execution on the device",
# which says nothing about the GPU being the cause. Pinning a CUDA 11.8 build
# keeps sm_60 kernels and runs on P100, T4 and L4 alike; CUDA wheels bundle
# their own runtime, so this only needs the P100's driver to be new enough
# for 11.8, which it is on every Kaggle host observed so far.
sh("pip install --no-cache-dir -q torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118",
   cwd=SRC)
# The base image's torchvision/torchaudio were built against whatever torch
# the image shipped with, not the cu118 build just installed above. Nothing
# here does vision or audio work, but transformers' lazy loader imports
# torchvision at PreTrainedModel resolution time regardless, and an ABI
# mismatch there ("operator torchvision::nms does not exist") poisons the
# loader for every other name too -- the same class of cascade peft caused.
# Removing them is simpler and safer than chasing a matching build.
sh("pip uninstall -y -q torchvision torchaudio", cwd=SRC)
sh("python -c \"import torch; "
   "print('torch', torch.__version__, "
   "'| cuda available:', torch.cuda.is_available(), "
   "'| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
   "'| capability:', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None); "
   "x = torch.randn(4, 4, device='cuda'); y = x @ x; torch.cuda.synchronize(); "
   "print('GPU matmul OK:', y.shape)\"",
   cwd=SRC)
sh("pip install --no-cache-dir -q transformers==5.14.1", cwd=SRC)
sh("pip install --no-cache-dir -q -r requirements.txt --no-warn-conflicts", cwd=SRC)
# This recipe never sets USE_LORA=True and nothing in exact_match/ imports
# peft directly, but requirements.txt installs it anyway (for anyone who
# does flip that flag). peft's constants module unconditionally imports
# BloomPreTrainedModel from transformers; against 5.14.1 that import fails,
# and the failure poisons transformers' lazy module loader so that even
# unrelated names -- `from transformers import get_scheduler`, needed by
# every run regardless of LoRA -- fail too. Removing peft here removes the
# broken import path; accelerate and transformers both treat peft as
# optional and degrade gracefully when it's absent.
sh("pip uninstall -y -q peft", cwd=SRC)
sh(f"mkdir -p {WORK}/wdc && find /kaggle/input -name 'wdcproducts*' "
   f"-exec cp {{}} {WORK}/wdc/ \\;")

for seed in (42, 1337):
    print(f"\n{'=' * 78}\nSEED {seed}\n{'=' * 78}", flush=True)
    sh(f"python build_real_corpus.py --wdc {WORK}/wdc --wdc-size large --swap-augment",
       cwd=SRC)
    sh(f"SEED={seed} python train_on_real_corpus.py --wdc {WORK}/wdc", cwd=SRC)
    sh("python evaluate_indian.py --model trained_model_real", cwd=SRC)
    sh(f"python calibrate_threshold.py --model trained_model_real --wdc {WORK}/wdc --write",
       cwd=SRC)
    # zip AFTER calibrate so inference_threshold is inside the archive -- every
    # previous run zipped first and shipped a checkpoint missing that field
    import shutil
    shutil.make_archive(f"{WORK}/swapaug_seed{seed}", "zip", f"{SRC}/trained_model_real")
    print(f"saved swapaug_seed{seed}.zip", flush=True)

print("\nDONE. Compare each seed's hard-slice precision and the symmetry check.")
