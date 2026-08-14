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
sh("pip install --no-cache-dir -q transformers==5.14.1", cwd=SRC)
sh("pip install --no-cache-dir -q -r requirements.txt --no-warn-conflicts", cwd=SRC)
sh(f"mkdir -p {WORK}/wdc && find /kaggle/input -name 'wdcproducts*' "
   f"-exec cp {{}} {WORK}/wdc/ \;")

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
