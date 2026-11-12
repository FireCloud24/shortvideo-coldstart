"""Content features: reading titles and covers, the two encoders, and the cache.

Feature files live in data/features/<name>.npy with shape [max_item_id + 1, dim],
indexed by the **original MicroLens item id** (row 0 is unused and zero). A
<name>.meta.json next to each records the model, runtime and sanity checks.

Note on language: MicroLens-100K ships English titles only
(MicroLens-100k_title_en.csv). Chinese-CLIP's text tower is a Chinese RoBERTa,
so its title embeddings here come from out-of-language input. Treat the text
results as a lower bound and see the README limitations.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

CNCLIP_DEFAULT = "OFA-Sys/chinese-clip-vit-base-patch16"
QWEN_DEFAULT = "Qwen/Qwen2-VL-2B-Instruct"
ENCLIP_DEFAULT = "openai/clip-vit-base-patch16"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


# ---------------------------------------------------------------- raw inputs

def read_titles(path: str | Path) -> dict[int, str]:
    """``item, title`` per line, no header. Titles may contain commas."""
    titles = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item, _, title = line.partition(",")
        titles[int(item)] = title.strip()
    return titles


def read_categories(path: str | Path) -> dict[int, str]:
    """``item, category`` per line (MicroLens tags_to_summary.csv), no header."""
    categories = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item, _, category = line.partition(",")
        if item.strip().isdigit():
            categories[int(item)] = category.strip()
    return categories


def find_covers(covers_dir: str | Path) -> dict[int, Path]:
    """Map item id -> cover path for files named ``{item}.jpg``."""
    covers = {}
    for p in Path(covers_dir).rglob("*"):
        if p.suffix.lower() in IMAGE_SUFFIXES and p.stem.isdigit():
            covers[int(p.stem)] = p
    return covers


# ---------------------------------------------------------------- the cache

def check_features(arr: np.ndarray, n_rows: int) -> dict:
    """Shape and NaN checks; raises on anything that would silently corrupt training."""
    if arr.ndim != 2 or arr.shape[0] != n_rows:
        raise ValueError(f"expected shape [{n_rows}, dim], got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{int((~np.isfinite(arr)).sum())} NaN/inf values in features")
    zero_rows = int((np.abs(arr[1:]).sum(axis=1) == 0).sum())
    return {"shape": list(arr.shape), "zero_rows": zero_rows, "dtype": str(arr.dtype)}


def save_features(features_dir: str | Path, name: str, arr: np.ndarray, meta: dict) -> Path:
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    meta = {**meta, **check_features(arr, arr.shape[0])}
    np.save(features_dir / f"{name}.npy", arr.astype(np.float32))
    (features_dir / f"{name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return features_dir / f"{name}.npy"


def load_feature_table(features_dir: str | Path, names: list[str], item_ids: np.ndarray) -> torch.Tensor:
    """Features for the model, row i = item index i (row 0 = padding = zeros).

    Each named feature is L2-normalised per item and the features are
    concatenated. Every real item must have a row in every file.
    """
    if not names:
        raise ValueError("no feature names given (set model.features in the config)")
    parts = []
    for name in names:
        arr = np.load(Path(features_dir) / f"{name}.npy")
        if item_ids[1:].max() >= arr.shape[0]:
            raise ValueError(f"{name}.npy has {arr.shape[0]} rows but item id {item_ids[1:].max()} is needed")
        check_features(arr, arr.shape[0])
        part = torch.zeros(len(item_ids), arr.shape[1])
        part[1:] = torch.from_numpy(arr[item_ids[1:]]).float()
        parts.append(torch.nn.functional.normalize(part, dim=1))  # zero rows stay zero
    return torch.cat(parts, dim=1)


# ---------------------------------------------------------------- encoders

def cnclip_text_features(model, input_ids, attention_mask) -> torch.Tensor:
    """[CLS] state through the text projection (what ChineseCLIPModel.get_text_features does)."""
    out = model.text_model(input_ids=input_ids, attention_mask=attention_mask)
    return model.text_projection(out.last_hidden_state[:, 0])


def cnclip_image_features(model, pixel_values) -> torch.Tensor:
    out = model.vision_model(pixel_values=pixel_values)
    return model.visual_projection(out.pooler_output)


def open_image(path: Path):
    from PIL import Image

    with Image.open(path) as im:
        return im.convert("RGB")


class CnClipEncoder:
    """Chinese-CLIP image and text towers. Pass model/tokenizer/image_processor to
    inject a small offline model in tests; otherwise they are downloaded."""

    def __init__(self, model_name: str = CNCLIP_DEFAULT, device: str | None = None, lora_dir: str | None = None,
                 model=None, tokenizer=None, image_processor=None, max_text_len: int = 52):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name
        if model is None:
            from transformers import AutoImageProcessor, AutoTokenizer, ChineseCLIPModel

            model = ChineseCLIPModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            image_processor = AutoImageProcessor.from_pretrained(model_name)
        if lora_dir is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, lora_dir)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_text_len = max_text_len

    def tokenize(self, texts: list[str]) -> dict:
        return self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_text_len,
                              return_tensors="pt")

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        tok = self.tokenize(texts).to(self.device)
        return cnclip_text_features(self.model, tok["input_ids"], tok["attention_mask"]).float().cpu().numpy()

    @torch.no_grad()
    def encode_images(self, paths: list[Path]) -> np.ndarray:
        pixels = self.image_processor(images=[open_image(p) for p in paths], return_tensors="pt")["pixel_values"]
        return cnclip_image_features(self.model, pixels.to(self.device)).float().cpu().numpy()


class EnClipTextEncoder:
    """The text tower of the original (English) CLIP.

    MicroLens-100K ships its titles in English. Chinese-CLIP's text tower was trained on
    Chinese captions, so feeding it English titles is a mismatch; this encoder reads the
    same titles with a tower trained on English. Pass model/tokenizer to inject a small
    offline model in tests.
    """

    def __init__(self, model_name: str = ENCLIP_DEFAULT, device: str | None = None, model=None, tokenizer=None,
                 max_text_len: int = 77):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name
        if model is None:
            from transformers import AutoTokenizer, CLIPModel

            model = CLIPModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_text_len,
                             return_tensors="pt")
        out = self.model.text_model(input_ids=tok["input_ids"].to(self.device),
                                    attention_mask=tok["attention_mask"].to(self.device))
        # pooler_output is the end-of-text token's state; this is what get_text_features projects.
        return self.model.text_projection(out.pooler_output).float().cpu().numpy()


class QwenVLEncoder:
    """One fused vector per video from Qwen2-VL: the cover and the title go in as
    one chat turn and the final-layer hidden states are mean-pooled.

    NOT exercised by the unit tests (the model is ~4.5 GB); first run it on Colab
    with ``--limit 32``.
    """

    PROMPT = "Video title: {title}"

    def __init__(self, model_name: str = QWEN_DEFAULT, device: str | None = None, max_pixels: int = 256 * 28 * 28):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_name = model_name
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
        self.model = self.model.to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, min_pixels=64 * 28 * 28, max_pixels=max_pixels)
        self.processor.tokenizer.padding_side = "right"

    def _chat(self, title: str, with_image: bool) -> str:
        content = ([{"type": "image"}] if with_image else []) + [
            {"type": "text", "text": self.PROMPT.format(title=title)}
        ]
        return self.processor.apply_chat_template([{"role": "user", "content": content}],
                                                  tokenize=False, add_generation_prompt=False)

    @torch.no_grad()
    def _encode_group(self, titles: list[str], paths: list[Path] | None) -> np.ndarray:
        texts = [self._chat(t, paths is not None) for t in titles]
        images = [open_image(p) for p in paths] if paths is not None else None
        inputs = self.processor(text=texts, images=images, padding=True, return_tensors="pt").to(self.device)
        out = self.model(**inputs, output_hidden_states=True)
        hidden = out.hidden_states[-1].float()
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        return ((hidden * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()

    def encode_pairs(self, titles: list[str], paths: list[Path | None]) -> np.ndarray:
        """Items with a cover use image + title; items without use the title alone."""
        out = [None] * len(titles)
        for has_image in (True, False):
            idx = [i for i, p in enumerate(paths) if (p is not None) == has_image]
            if not idx:
                continue
            vecs = self._encode_group([titles[i] for i in idx], [paths[i] for i in idx] if has_image else None)
            for i, v in zip(idx, vecs):
                out[i] = v
        return np.stack(out)


# ---------------------------------------------------------------- batching

def embed_in_batches(encode, inputs: list, batch_size: int, log_every: int = 50, log=print) -> np.ndarray:
    """Run ``encode(batch) -> [len(batch), dim]`` over ``inputs`` and stack the results."""
    chunks, start_time = [], time.time()
    for b, start in enumerate(range(0, len(inputs), batch_size)):
        chunks.append(np.asarray(encode(inputs[start:start + batch_size]), dtype=np.float32))
        if log_every and (b + 1) % log_every == 0:
            done = start + batch_size
            log(f"    {min(done, len(inputs))}/{len(inputs)}  ({time.time() - start_time:.0f}s)")
    return np.concatenate(chunks) if chunks else np.zeros((0, 0), dtype=np.float32)


def scatter_rows(item_ids: list[int], vectors: np.ndarray, n_rows: int) -> np.ndarray:
    """Place vectors[i] at row item_ids[i] of a zero [n_rows, dim] array."""
    out = np.zeros((n_rows, vectors.shape[1]), dtype=np.float32)
    out[np.asarray(item_ids)] = vectors
    return out
