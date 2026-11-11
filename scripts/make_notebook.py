"""Regenerate notebooks/colab_end_to_end.ipynb (kept as code so the notebook diffs cleanly).

    python scripts/make_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_URL = "https://github.com/FireCloud24/shortvideo-coldstart.git"

CELLS = [
    ("md", "# Cold-start short-video recommendation: end to end\n\n"
           "Runs the whole pipeline on a free Colab T4: download, tests, embeddings, all models x 3 seeds, "
           "LoRA, report.\n\n**Runtime > Change runtime type > T4 GPU** before running."),
    ("code", f"REPO_URL = \"{REPO_URL}\"  # change if your fork lives elsewhere\n"
             "SEEDS = \"0 1 2\"\n\n"
             "import os\n"
             "# Absolute paths, so re-running this cell does not clone a second copy inside the first.\n"
             "if not os.path.exists('/content/shortvideo-coldstart'):\n"
             "    !git clone -q $REPO_URL /content/shortvideo-coldstart\n"
             "%cd /content/shortvideo-coldstart\n"
             "!pip install -q -r requirements.txt\n"
             "# Colab ships an old torchao that makes peft refuse to import; this project does not use torchao.\n"
             "!pip uninstall -y -q torchao\n"
             "import torch; print(torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"),
    ("md", "## 1. Tests"),
    ("code", "!python -m pytest -q"),
    ("md", "## 2. Download MicroLens-100K (T2)\nAbout 710 MB. Re-running skips files already present."),
    ("code", "!python scripts/download.py\n!cat data/raw/manifest.json | head -20"),
    ("md", "## 3. Embeddings (T6)\nA 64-item check first, so a problem shows up in a minute rather than an hour."),
    ("code", "!python scripts/embed.py --model cnclip --limit 64\n!python scripts/embed.py --model qwen --limit 32"),
    ("code", "!python scripts/embed.py --model cnclip\n!python scripts/embed.py --model qwen\n"
             "!cat data/features/*.meta.json"),
    ("md", "## 4. Baselines and content models (T5, T7)"),
    ("code", "CONFIGS = ['popularity', 'sasrec_id', 'content_cnclip_image', 'content_cnclip_text',\n"
             "           'content_cnclip_image_text', 'content_qwen_vl',\n"
             "           'id_content_cnclip_image_text', 'id_content_qwen_vl']\n"
             "for name in CONFIGS:\n"
             "    seeds = '0' if name == 'popularity' else SEEDS  # popularity has no randomness\n"
             "    !python scripts/train.py --config configs/{name}.yaml --seeds {seeds}"),
    ("md", "## 5. LoRA fine-tuning (T8)\n"
           "Text tower by default. If it runs out of memory, lower `--batch-size`, or use the adapter "
           "fallback in the next cell instead."),
    ("code", "!python scripts/finetune.py --mode lora --modality text\n"
             "!python scripts/embed.py --model cnclip --what text --lora-dir checkpoints/lora_text --tag lora\n"
             "!python scripts/train.py --config configs/content_cnclip_text_lora.yaml --seeds {SEEDS}"),
    ("code", "# Fallback / extra comparison: 2-layer adapter on the cached image features.\n"
             "!python scripts/finetune.py --mode adapter --feature cnclip_image\n"
             "!python scripts/train.py --config configs/content_cnclip_image.yaml --seeds {SEEDS} "
             "--set name=content_cnclip_image_adapter \"model.features=[cnclip_image_adapter]\""),
    ("md", "## 6. Report (T9)"),
    ("code", "!python scripts/report.py\n"
             "from IPython.display import Markdown, Image, display\n"
             "display(Markdown(open('results/results.md').read()))\n"
             "display(Image('results/warm_vs_cold.png'))\n"
             "display(Markdown(open('results/error_slices.md').read()))\n"
             "display(Image('results/error_slices.png'))"),
    ("md", "## 7. Save the results\n"
           "`results/` holds only metrics, tables and charts (no dataset files), so it can be committed. "
           "Download the zip and commit it from your machine."),
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
