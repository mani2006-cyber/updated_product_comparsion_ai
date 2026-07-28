"""
kaggle_train_driver.py
=======================
Runs the existing exact_match/ training pipeline (train.py) unmodified,
on Kaggle, against the real 5-class relationship dataset
(data/relationship_pairs_final.csv), then evaluates on the held-out
test split and produces a classification report + confusion matrix.

This file does NOT change config.py, save_model.py, or anything under
exact_match/ -- it only overrides config values in memory, at runtime,
before those modules get imported. See the chat explanation for why
that ordering matters (Python default-argument binding).

Usage (inside a Kaggle Notebook, GPU accelerator enabled):
    !python kaggle_train_driver.py
"""

import glob
import json
import os
import shutil
import sys
import zipfile

KAGGLE_INPUT = "/kaggle/input"
WORK_DIR = "/kaggle/working"
REPO_COPY_DIR = os.path.join(WORK_DIR, "repo")

# Directories we never want to copy into the writable working copy --
# they're either huge, regenerable, or Kaggle-input-only. "data" is
# excluded deliberately: the training CSV is read directly from the
# read-only /kaggle/input mount via data_csv_path, so none of data/'s
# other multi-hundred-MB-to-multi-GB files need to be duplicated.
_EXCLUDE_DIR_NAMES = {"venv", "__pycache__", ".git", "trained_model_v1",
                      ".pytest_cache", "outputs", "trained_model", "data"}


def _find_one(filename: str, search_root: str = None) -> str:
    """Recursively finds `filename` under a Kaggle input mount.

    `search_root` deliberately defaults to None and resolves KAGGLE_INPUT
    at *call* time, not as a default argument value. A default of
    `search_root=KAGGLE_INPUT` would bind once at import and silently
    ignore any later reassignment of the module constant -- the same
    default-argument-binding trap that forces _override_config() to run
    before exact_match is imported (see its docstring). Resolving at call
    time is also what makes this script testable against a simulated
    /kaggle/input tree.
    """
    root = search_root if search_root is not None else KAGGLE_INPUT
    matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"Could not find '{filename}' anywhere under {root}. "
            "Did you attach your code/data Kaggle Dataset to this notebook "
            "(Add Input, top-right panel)?"
        )
    # glob returns filesystem order, not sorted -- if a stray copy exists in
    # a nested folder (an old backup, a venv), prefer the shallowest path so
    # we pick the real top-level repo file rather than whichever came first.
    return min(matches, key=lambda p: (p.count(os.sep), len(p)))


def _copy_repo_to_working_dir(config_py_path: str) -> str:
    """config.py currently lives under the read-only /kaggle/input mount --
    config.py itself calls os.makedirs() at import time, which would raise
    PermissionError there. So we copy the code (not the multi-GB data
    files) into /kaggle/working, which is writable, and import from there."""
    src_root = os.path.dirname(config_py_path)
    if os.path.exists(REPO_COPY_DIR):
        shutil.rmtree(REPO_COPY_DIR)

    shutil.copytree(
        src_root, REPO_COPY_DIR,
        ignore=shutil.ignore_patterns(*_EXCLUDE_DIR_NAMES, "*.zip", "text_part_*"),
    )
    return REPO_COPY_DIR


def _setup_paths_for_kaggle():
    config_py_path = _find_one("config.py")
    repo_dir = _copy_repo_to_working_dir(config_py_path)

    sys.path.insert(0, repo_dir)
    os.chdir(repo_dir)

    data_csv_path = _find_one("relationship_pairs_final.csv")
    return data_csv_path


def _override_config(data_csv_path: str):
    """Every path/hyperparameter override MUST happen here, before this
    function returns -- and this function must be called before `import
    exact_match...` anywhere. exact_match/model.py and save_model.py read
    config.NUM_LABELS / config.TRAINED_MODEL_DIR as *default argument
    values*, which Python binds once, at the moment those modules are
    first imported -- not each time the function runs. Setting
    config.NUM_LABELS = 5 after exact_match.model has already been
    imported would silently have no effect."""
    import config

    config.RAW_DATA_PATH = data_csv_path
    config.NUM_LABELS = 5  # relationship_pairs_final.csv is the 5-class schema

    config.OUTPUT_DIR = os.path.join(WORK_DIR, "outputs")
    config.CHECKPOINT_DIR = os.path.join(config.OUTPUT_DIR, "checkpoints")
    config.PLOTS_DIR = os.path.join(config.OUTPUT_DIR, "plots")
    config.LOGS_DIR = os.path.join(config.OUTPUT_DIR, "logs")
    config.REPORTS_DIR = os.path.join(config.OUTPUT_DIR, "reports")
    config.TRAINED_MODEL_DIR = os.path.join(WORK_DIR, "trained_model")
    config.BEST_CHECKPOINT_PATH = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")

    for d in (config.OUTPUT_DIR, config.CHECKPOINT_DIR, config.PLOTS_DIR,
              config.LOGS_DIR, config.REPORTS_DIR, config.TRAINED_MODEL_DIR):
        os.makedirs(d, exist_ok=True)

    return config


def _fix_label_map_bug(config):
    """save_model.py always writes the binary label_map into
    training_metadata.json, even for the 5-class model (a known bug --
    see training_metadata.json in the existing trained_model/ right now).
    Patched here after export instead of editing save_model.py, since
    this script's job is to drive training, not modify the pipeline."""
    metadata_path = os.path.join(config.TRAINED_MODEL_DIR, "training_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    metadata["label_map"] = {str(v): k for k, v in config.RELATIONSHIP_LABEL_MAP.items()}
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Fixed label_map in {metadata_path}")


def _zip_results(config) -> str:
    zip_path = os.path.join(WORK_DIR, "trained_model_and_reports.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for folder in (config.TRAINED_MODEL_DIR, config.PLOTS_DIR,
                       config.REPORTS_DIR, config.LOGS_DIR):
            for root, _, files in os.walk(folder):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    zf.write(fpath, os.path.relpath(fpath, WORK_DIR))
    return zip_path


def main():
    print("Step 1/5: locating code + dataset under /kaggle/input ...")
    data_csv_path = _setup_paths_for_kaggle()
    print(f"  data: {data_csv_path}")
    print(f"  code copied to: {REPO_COPY_DIR}")

    print("Step 2/5: overriding config for Kaggle paths + 5-class schema ...")
    config = _override_config(data_csv_path)

    import torch
    if not torch.cuda.is_available():
        # Fail loudly rather than silently starting a CPU run: ~57k training
        # rows x up to 15 epochs on CPU would not finish in a Kaggle session,
        # and the failure would only become obvious hours later. The usual
        # causes are (a) the notebook Accelerator is still set to "None", or
        # (b) pip reinstalled a CPU-only torch build over Kaggle's CUDA one.
        raise SystemExit(
            "ERROR: no GPU visible to torch -- refusing to start a CPU run.\n"
            "  1. Notebook -> Settings -> Accelerator -> GPU T4 x2 (or P100), then restart.\n"
            f"  2. Check you did not reinstall torch over Kaggle's CUDA build "
            f"(current torch: {torch.__version__}; a '+cpu' suffix means it was replaced).\n"
            "     Install the other requirements WITHOUT torch -- see the run instructions."
        )
    print(f"  CUDA available: True ({torch.cuda.get_device_name(0)})")

    print("Step 3/5: training (this calls the existing exact_match/train.py train() unmodified) ...")
    from exact_match.train import train
    train()

    # train()'s locals are freed on return, but CUDA's caching allocator holds
    # the freed blocks. evaluate.main() loads a second copy of the model, so
    # release the cache first to avoid an avoidable OOM on a 16GB T4.
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("Step 4/5: evaluating best model on the held-out test split ...")
    from exact_match import evaluate as exact_match_evaluate
    sys.argv = [sys.argv[0]]  # evaluate.main() parses argv; strip Kaggle's own args
    exact_match_evaluate.main()

    _fix_label_map_bug(config)

    print("Step 5/5: zipping trained_model/ + outputs/ for download ...")
    zip_path = _zip_results(config)
    print(f"\nDone. Download this file from the Kaggle 'Output' tab: {zip_path}")
    print("Paste back the full console output above (training log + test-split "
          "classification report + confusion matrix line) so it can be reviewed.")


if __name__ == "__main__":
    main()
