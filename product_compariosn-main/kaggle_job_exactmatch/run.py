"""
Kaggle job: trains the new standalone exactmatch/ package.

Pushed with `kaggle kernels push -p kaggle_job_exactmatch`. Same real
sources as v7-v11 -- WDC Products 2023 + ER-Magellan + hard-negative-mined
LSPC, all human-labelled -- built fresh via the existing, unmodified
build_real_corpus.py + mine_hard_negatives.py at the repo root. exactmatch/
consumes that corpus; it does not regenerate or invent data.

Reuses every environment fix already proven across kaggle_job/ and
kaggle_job_category/ (see their git history): cu118 torch for P100
compatibility, transformers' >=2.4 floor, peft/torchvision removed to
avoid poisoning transformers' lazy loader.
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

# Same corpus v11 trained on: plain build, then hard-negative mining on top.
# mine_hard_negatives.py leaves validation untouched ("Validation is
# unchanged" -- printed by the script itself), so this reproduces the exact
# 10,609-pair validation set v11 was calibrated against too.
sh(f"python build_real_corpus.py --wdc {WORK}/wdc --wdc-size large", cwd=SRC)
sh(f"python mine_hard_negatives.py --wdc {WORK}/wdc --max-pairs 20000", cwd=SRC)

sh("python -m exactmatch.train", cwd=SRC)
sh(f"python -m exactmatch.evaluate --model exactmatch/trained_model --wdc {WORK}/wdc", cwd=SRC)

import shutil
shutil.make_archive(f"{WORK}/exactmatch_model", "zip", f"{SRC}/exactmatch/trained_model")
print("saved exactmatch_model.zip", flush=True)

print("\nDONE. Compare the benchmark mean above to v11's 82.26.")
