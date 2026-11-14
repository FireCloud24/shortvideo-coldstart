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
           "x 3 seeds, the tables and figures, then the vector-length test.\n\n"
           "**Runtime > Change runtime type > T4 GPU** before running. The full run takes several hours; "
           "results are written as each step finishes, so a session that is cut off still leaves the "
           "finished runs in `results/`."),
    ("code", f"REPO_URL = \"{REPO_URL}\"  # change if your fork lives elsewhere\n"
             "SEEDS = \"0 1 2\"\n"
             "EXTRAS = 0  # 1 also runs English CLIP titles, LoRA and Qwen2-VL (slow; not needed for the finding)\n"
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
             "    PERSIST_DIR = '/content/drive/MyDrive/shortvideo-coldstart-run2'\n"
             "print('saving to:', PERSIST_DIR or 'this session only')"),
    ("md", "## 1. Tests"),
    ("code", "!python -m pytest -q"),
    ("md", "## 2. Download MicroLens-100K\nAbout 710 MB. Re-running skips files already present."),
    ("code", "!python scripts/download.py\n!cat data/raw/manifest.json | head -20"),
    ("md", "## 3. Smoke test\nA few items through every encoder and one training epoch, so a problem shows "
           "up in minutes rather than hours."),
    ("code", "!python scripts/embed.py --model cnclip --limit 64\n"
             "!python scripts/embed.py --model enclip --limit 64\n"
             "!python scripts/train.py --config configs/sasrec_id.yaml --seeds 0 "
             "--set train.epochs=1 out_dir=results_smoke"),
    ("md", "## 4. Every experiment\n"
           "`scripts/run_all.sh` runs the baselines, the content models, LoRA and the report, cheapest "
           "first. The full log goes to `run_all.log`; the next cell prints its summary lines."),
    ("code", "import os\n"
             "# The repository's results/ folder holds an earlier run; remove it so no old number survives.\n"
             "!rm -rf results/runs results/finetune results/features results/lr_search results/*.md results/*.png\n"
             "if PERSIST_DIR and os.path.isdir(f'{PERSIST_DIR}/results'):\n"
             "    print('resuming from', PERSIST_DIR)\n"
             "    !mkdir -p results data/features checkpoints\n"
             "    !cp -r \"{PERSIST_DIR}/results/.\" results/\n"
             "    !cp -r \"{PERSIST_DIR}/features/.\" data/features/\n"
             "    !cp -r \"{PERSIST_DIR}/checkpoints/.\" checkpoints/\n"
             "    !cp \"{PERSIST_DIR}/run_all.log\" run_all.log\n"
             "    !echo \"=== resumed in a new session ===\" >> run_all.log\n"
             "!SEEDS=\"{SEEDS}\" EXTRAS={EXTRAS} IMAGE_LORA_STEPS={IMAGE_LORA_STEPS} PERSIST_DIR=\"{PERSIST_DIR}\" "
             "bash scripts/run_all.sh >> run_all.log 2>&1\n"
             "!cp run_all.log results/run_all.log\n"
             "!grep -E '^=== |seed [0-9]+: |^!!! |^wrote |cost: ' run_all.log"),
    ("md", "## 5. Report"),
    ("code", "from IPython.display import Markdown, Image, display\n"
             "for name in ('tradeoff', 'exposure', 'categories'):\n"
             "    display(Image(f'results/{name}.png'))\n"
             "    display(Markdown(open(f'results/{name}.md').read()))\n"
             "display(Markdown(open('results/results.md').read()))\n"
             "display(Image('results/warm_vs_cold.png'))\n"
             "display(Markdown(open('results/error_slices.md').read()))\n"
             "display(Image('results/error_slices.png'))"),
    ("md", "## 5b. Vector-length test\n"
           "Retrains two ID + content models with their weights saved and multiplies the new videos' vectors "
           "by a factor, with nothing retrained. About half an hour."),
    ("code", "!bash scripts/run_scale_test.sh > scale_test.log 2>&1\n"
             "!grep -E '^=== |as trained|^!!! |^wrote ' scale_test.log\n"
             "from IPython.display import Markdown, Image, display\n"
             "display(Image('results/scale_test/scale_test.png'))\n"
             "display(Markdown(open('results/scale_test/scale_test.md').read()))"),
    ("md", "## 6. Save the results\n"
           "`results/` holds only metrics, per-example ranks, tables, charts and the run log (no dataset "
           "files), so it can be committed. Download the zip and commit it from your machine, together "
           "with this notebook (File > Download > .ipynb) so the outputs above are kept."),
    ("code", "!zip -q -r results.zip results\nfrom google.colab import files\nfiles.download('results.zip')"),
]

# Kaggle: a session can run for 12 hours and nothing is asked of the browser when the notebook
# is run with "Save Version > Save & Run All". Everything under /kaggle/working is kept as the
# version's output; the code and the dataset go to /kaggle/temp, which is not kept.
KAGGLE_CELLS = [
    ("md", "# Cold-start short-video recommendation: end to end (Kaggle)\n\n"
           "Settings (right-hand panel): **Accelerator: GPU T4 x2** (or P100) and **Internet: on**. "
           "Then **Save Version > Save & Run All (Commit)**, which runs in the background for up to 12 "
           "hours.\n\n"
           "The run stops starting new steps after `MAX_HOURS` and saves what it has. To continue, add the "
           "previous version's output as an input (**Add Input > Your Work > this notebook**) and run "
           "again: finished runs are skipped.\n\n"
           "Keep this notebook private. Its output holds features computed from MicroLens, and the "
           "dataset's licence does not allow passing on modified copies."),
    ("code", f"REPO_URL = \"{REPO_URL}\"\n"
             "SEEDS = \"0 1 2\"\n"
             "EXTRAS = 0  # 1 also runs English CLIP titles, LoRA and Qwen2-VL (slow; not needed for the finding)\n"
             "IMAGE_LORA_STEPS = 3000\n"
             "MAX_HOURS = 10  # start no new step after this; leaves time to save before the 12-hour limit\n\n"
             "import glob, os\n"
             "WORK = '/kaggle/temp/shortvideo-coldstart'\n"
             "PERSIST_DIR = '/kaggle/working/persist'\n"
             "os.makedirs('/kaggle/temp', exist_ok=True)\n"
             "if not os.path.exists(WORK):\n"
             "    !git clone -q $REPO_URL $WORK\n"
             "%cd $WORK\n"
             "!git log -1 --format='code version: %h %s'\n"
             "!pip install -q -r requirements.txt\n"
             "!pip uninstall -y -q torchao\n"
             "import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')\n"
             "assert torch.cuda.is_available(), 'No GPU: set Accelerator to GPU in the notebook settings'"),
    ("md", "## 1. Tests"),
    ("code", "!python -m pytest -q"),
    ("md", "## 2. Download MicroLens-100K\nAbout 710 MB, into `/kaggle/temp` (not kept in the output)."),
    ("code", "!python scripts/download.py\n!cat data/raw/manifest.json | head -20"),
    ("md", "## 3. Smoke test"),
    ("code", "!python scripts/embed.py --model cnclip --limit 64\n"
             "!python scripts/embed.py --model enclip --limit 64\n"
             "!python scripts/train.py --config configs/sasrec_id.yaml --seeds 0 "
             "--set train.epochs=1 out_dir=results_smoke"),
    ("md", "## 4. Every experiment\n"
           "Resumes from an earlier version's output when one is attached as an input."),
    ("code", "# The repository's results/ folder holds an earlier run; remove it so no old number survives.\n"
             "!rm -rf results/runs results/finetune results/features results/lr_search results/*.md results/*.png\n"
             "earlier = sorted(glob.glob('/kaggle/input/*/persist') + glob.glob('/kaggle/input/*/*/persist'))\n"
             "if earlier and not os.path.isdir(f'{PERSIST_DIR}/results'):\n"
             "    print('resuming from', earlier[-1])\n"
             "    !mkdir -p $PERSIST_DIR\n"
             "    !cp -r \"{earlier[-1]}/.\" $PERSIST_DIR/\n"
             "if os.path.isdir(f'{PERSIST_DIR}/results'):\n"
             "    !mkdir -p results data/features checkpoints\n"
             "    !cp -r $PERSIST_DIR/results/. results/\n"
             "    !cp -r $PERSIST_DIR/features/. data/features/\n"
             "    !cp -r $PERSIST_DIR/checkpoints/. checkpoints/\n"
             "    !cp $PERSIST_DIR/run_all.log run_all.log\n"
             "    !echo \"=== resumed in a new session ===\" >> run_all.log\n"
             "!SEEDS=\"{SEEDS}\" EXTRAS={EXTRAS} IMAGE_LORA_STEPS={IMAGE_LORA_STEPS} MAX_HOURS={MAX_HOURS} "
             "PERSIST_DIR=\"{PERSIST_DIR}\" bash scripts/run_all.sh >> run_all.log 2>&1\n"
             "!cp run_all.log results/run_all.log\n"
             "!grep -E '^=== |seed [0-9]+: |^!!! |^wrote |cost: ' run_all.log"),
    ("md", "## 5. Report"),
    ("code", "from IPython.display import Markdown, Image, display\n"
             "for name in ('tradeoff', 'exposure', 'categories', 'warm_vs_cold', 'error_slices'):\n"
             "    if os.path.exists(f'results/{name}.png'):\n"
             "        display(Image(f'results/{name}.png'))\n"
             "for name in ('lr_search/chosen', 'tradeoff', 'exposure', 'categories', 'results'):\n"
             "    if os.path.exists(f'results/{name}.md'):\n"
             "        display(Markdown(open(f'results/{name}.md').read()))"),
    ("md", "## 5b. Vector-length test\n"
           "Retrains two ID + content models with their weights saved and multiplies the new videos' vectors "
           "by a factor, with nothing retrained. About half an hour."),
    ("code", "!bash scripts/run_scale_test.sh > scale_test.log 2>&1\n"
             "!grep -E '^=== |as trained|^!!! |^wrote ' scale_test.log\n"
             "from IPython.display import Markdown, Image, display\n"
             "display(Image('results/scale_test/scale_test.png'))\n"
             "display(Markdown(open('results/scale_test/scale_test.md').read()))"),
    ("md", "## 6. Save the results\n"
           "`results.zip` holds only metrics, per-example ranks, tables, charts and the run log (no dataset "
           "files), so it can be committed. It appears under this version's **Output**."),
    ("code", "!cp -r results/. $PERSIST_DIR/results/\n"
             "!cd $WORK && zip -q -r /kaggle/working/results.zip results\n"
             "!ls -la /kaggle/working"),
]


def build(spec=None) -> dict:
    cells = []
    for kind, source in (CELLS if spec is None else spec):
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
    folder = Path(__file__).resolve().parents[1] / "notebooks"
    folder.mkdir(exist_ok=True)
    for name, spec in (("colab_end_to_end.ipynb", CELLS), ("kaggle_end_to_end.ipynb", KAGGLE_CELLS)):
        (folder / name).write_text(json.dumps(build(spec), indent=1), encoding="utf-8")
        print(f"wrote {folder / name}")
