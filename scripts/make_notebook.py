"""Regenerate notebooks/colab_end_to_end.ipynb (kept as code so the notebook diffs cleanly).

    python scripts/make_notebook.py

The notebook runs scripts/run_all.sh, so the list of experiments lives in one place.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/FireCloud24/shortvideo-coldstart.git"

CELLS = [
    ("md", "# Cold-start short-video recommendation: end to end\n\n"
           "Runs the whole pipeline on a free Colab T4: download, tests, embeddings, baselines, all models "
           "x 3 seeds, LoRA on both towers, report.\n\n"
           "**Runtime > Change runtime type > T4 GPU** before running. The full run takes several hours; "
           "results are written as each step finishes, so a session that is cut off still leaves the "
           "finished runs in `results/`."),
    ("code", f"REPO_URL = \"{REPO_URL}\"  # change if your fork lives elsewhere\n"
             "SEEDS = \"0 1 2\"\n"
             "IMAGE_LORA_STEPS = 3000  # step budget for LoRA on the image tower (the slow stage)\n\n"
             "import os\n"
             "# Absolute paths, so re-running this cell does not clone a second copy inside the first.\n"
             "if not os.path.exists('/content/shortvideo-coldstart'):\n"
             "    !git clone -q $REPO_URL /content/shortvideo-coldstart\n"
             "%cd /content/shortvideo-coldstart\n"
             "!git log -1 --format='code version: %h %s'\n"
             "!pip install -q -r requirements.txt\n"
             "# Colab ships an old torchao that makes peft refuse to import; this project does not use torchao.\n"
             "!pip uninstall -y -q torchao\n"
             "import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"),
    ("md", "## 0. Keep the results safe (recommended)\n"
           "A free Colab session can end without warning, and everything on its disk is then lost. With "
           "`USE_DRIVE = True` the run copies results, feature files and checkpoints to a folder on your "
           "Google Drive after every step, and a later session picks up where the last one stopped. "
           "Colab will ask you to allow access to Drive. Delete the Drive folder to start from scratch "
           "(do this after any change to the code)."),
    ("code", "USE_DRIVE = True\n"
             "PERSIST_DIR = ''\n"
             "if USE_DRIVE:\n"
             "    from google.colab import drive\n"
             "    drive.mount('/content/drive')\n"
             "    PERSIST_DIR = '/content/drive/MyDrive/shortvideo-coldstart-run'\n"
             "print('saving to:', PERSIST_DIR or 'this session only')"),
    ("md", "## 1. Tests"),
    ("code", "!python -m pytest -q"),
    ("md", "## 2. Download MicroLens-100K\nAbout 710 MB. Re-running skips files already present."),
    ("code", "!python scripts/download.py\n!cat data/raw/manifest.json | head -20"),
    ("md", "## 3. Smoke test\nA few items through every encoder and one training epoch, so a problem shows "
           "up in minutes rather than hours."),
    ("code", "!python scripts/embed.py --model cnclip --limit 64\n"
             "!python scripts/embed.py --model enclip --limit 64\n"
             "!python scripts/embed.py --model qwen --limit 32\n"
             "!python scripts/train.py --config configs/sasrec_id.yaml --seeds 0 "
             "--set train.epochs=1 out_dir=results_smoke"),
    ("md", "## 4. Every experiment\n"
           "`scripts/run_all.sh` runs the baselines, the content models, LoRA and the report, cheapest "
           "first. The full log goes to `run_all.log`; the next cell prints its summary lines."),
    ("code", "import os\n"
             "# The repository's results/ folder holds an earlier run; remove it so no old number survives.\n"
             "!rm -rf results/runs results/finetune results/features\n"
             "if PERSIST_DIR and os.path.isdir(f'{PERSIST_DIR}/results'):\n"
             "    print('resuming from', PERSIST_DIR)\n"
             "    !mkdir -p results data/features checkpoints\n"
             "    !cp -r \"{PERSIST_DIR}/results/.\" results/\n"
             "    !cp -r \"{PERSIST_DIR}/features/.\" data/features/\n"
             "    !cp -r \"{PERSIST_DIR}/checkpoints/.\" checkpoints/\n"
             "    !cp \"{PERSIST_DIR}/run_all.log\" run_all.log\n"
             "    !echo \"=== resumed in a new session ===\" >> run_all.log\n"
             "!SEEDS=\"{SEEDS}\" IMAGE_LORA_STEPS={IMAGE_LORA_STEPS} PERSIST_DIR=\"{PERSIST_DIR}\" "
             "bash scripts/run_all.sh >> run_all.log 2>&1\n"
             "!cp run_all.log results/run_all.log\n"
             "!grep -E '^=== |seed [0-9]+: |^!!! |^wrote |cost: ' run_all.log"),
    ("md", "## 5. Report"),
    ("code", "from IPython.display import Markdown, Image, display\n"
             "display(Markdown(open('results/results.md').read()))\n"
             "display(Image('results/warm_vs_cold.png'))\n"
             "display(Markdown(open('results/error_slices.md').read()))\n"
             "display(Image('results/error_slices.png'))"),
    ("md", "## 6. Save the results\n"
           "`results/` holds only metrics, per-example ranks, tables, charts and the run log (no dataset "
           "files), so it can be committed. Download the zip and commit it from your machine, together "
           "with this notebook (File > Download > .ipynb) so the outputs above are kept."),
    ("code", "!zip -q -r results.zip results\nfrom google.colab import files\nfiles.download('results.zip')"),
]


def build() -> dict:
    cells = []
    for kind, source in CELLS:
        cell = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update({"execution_count": None, "outputs": []})
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                     "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "notebooks" / "colab_end_to_end.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=1), encoding="utf-8")
    print(f"wrote {out}")
