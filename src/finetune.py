"""Adapting the encoder to the recommendation task (RQ3).

Frozen features were trained to match images with captions, not to predict what
a user watches next. Here the item encoder itself receives the next-item
training signal, in one of two ways:

* ``lora``    LoRA matrices inside a Chinese-CLIP tower (text or image). The
              tower runs on every training step, so this is the expensive one.
* ``adapter`` a 2-layer residual MLP on top of the cached frozen features. The
              fallback when LoRA does not fit in memory; costs seconds.

Both are trained with the same loop: a SASRec trunk reads a user's history as
item vectors produced by the encoder and must pick the next item out of all the
items in the batch (in-batch softmax; a full softmax would need every item
encoded on every step).

Afterwards the adapted encoder embeds all items once more and the result is
cached like any other feature (``<name>_lora`` / ``<name>_adapter``). The
comparison with frozen features then uses the *same* downstream config and the
same number of epochs; the only difference is the feature file. The extra cost
of this stage is logged to results/finetune/<name>.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import PAD, Split, training_arrays
from .features import cnclip_image_features, cnclip_text_features, open_image
from .models import SASRec
from .train import set_seed

LORA_TARGETS = {
    # Regexes matched against full module names: the attention query/value projections of one tower.
    "text": r".*text_model.*\.(query|value)",
    "image": r".*vision_model.*\.(q_proj|v_proj)",
}


def add_lora(model, modality: str, r: int = 8, alpha: int = 16, dropout: float = 0.05):
    """Freeze ``model`` and add trainable LoRA matrices to one tower."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS[modality])
    return get_peft_model(model, config)


class ResidualAdapter(nn.Module):
    """feature + MLP(feature). Starts as the identity (last layer initialised to zero)."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TextItemEncoder(nn.Module):
    """Item indices -> Chinese-CLIP text features, through the (LoRA) text tower."""

    def __init__(self, clip, tokenizer, titles_by_index: list[str], max_text_len: int = 52):
        super().__init__()
        self.clip = clip
        tok = tokenizer(titles_by_index, padding="max_length", truncation=True, max_length=max_text_len,
                        return_tensors="pt")
        self.register_buffer("input_ids", tok["input_ids"], persistent=False)
        self.register_buffer("attention_mask", tok["attention_mask"], persistent=False)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        mask = self.attention_mask[items]
        width = max(int(mask.sum(1).max()), 1)  # trim the batch to its longest title
        return cnclip_text_features(self.clip, self.input_ids[items, :width], mask[:, :width])


class ImageItemEncoder(nn.Module):
    """Item indices -> Chinese-CLIP image features, through the (LoRA) vision tower."""

    def __init__(self, clip, image_processor, cover_by_index: list):
        super().__init__()
        self.clip = clip
        self.image_processor = image_processor
        self.covers = cover_by_index  # path or None per item index
        size = image_processor.crop_size if getattr(image_processor, "do_center_crop", False) else image_processor.size
        self.blank = torch.zeros(3, size["height"], size["width"])

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        device = next(self.clip.parameters()).device
        pixels = []
        for i in items.tolist():
            path = self.covers[i]
            if path is None:
                pixels.append(self.blank)
            else:
                pixels.append(self.image_processor(images=open_image(path), return_tensors="pt")["pixel_values"][0])
        return cnclip_image_features(self.clip, torch.stack(pixels).to(device))


class CachedItemEncoder(nn.Module):
    """Item indices -> adapter(frozen cached feature)."""

    def __init__(self, features: torch.Tensor, hidden: int = 256):
        super().__init__()
        self.register_buffer("features", features.float(), persistent=False)
        self.adapter = ResidualAdapter(features.shape[1], hidden)

    def forward(self, items: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.features[items])


class InBatchRecommender(nn.Module):
    """item encoder -> projection -> SASRec trunk, scored against the batch's own items."""

    def __init__(self, item_encoder: nn.Module, feat_dim: int, dim: int, max_len: int, n_layers: int = 2,
                 n_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.item_encoder = item_encoder
        self.proj = nn.Sequential(nn.Linear(feat_dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(4 * dim, dim))
        self.trunk = SASRec(nn.Identity(), dim, max_len, n_layers, n_heads, dropout)

    def loss(self, seqs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """seqs, targets: [B, L] item indices (PAD = 0), as produced by training_arrays."""
        both = torch.cat([seqs, targets], dim=1)
        unique, inverse = torch.unique(both, return_inverse=True)
        if unique[0] != PAD:  # make local index 0 the padding slot in every batch
            unique = torch.cat([unique.new_zeros(1), unique])
            inverse = inverse + 1
        feats = F.normalize(self.item_encoder(unique[1:]).float(), dim=1)
        table = torch.cat([feats.new_zeros(1, self.proj[-1].out_features), self.proj(feats)])

        local_seqs, local_targets = inverse[:, : seqs.shape[1]], inverse[:, seqs.shape[1]:]
        hidden = self.trunk.encode(table[local_seqs], local_seqs == PAD)
        keep = local_targets != PAD
        logits = hidden[keep] @ table.T
        logits[:, PAD] = float("-inf")
        return F.cross_entropy(logits, local_targets[keep])


def finetune(model: InBatchRecommender, split: Split, max_len: int, epochs: int, batch_size: int, lr: float,
             device, seed: int, max_steps: int | None = None, log=print, log_every: int = 50) -> dict:
    """Train ``model`` on the warm training sequences. Returns a cost/loss log."""
    set_seed(seed)
    model.to(device).train()
    inputs, targets = (torch.as_tensor(a) for a in training_arrays(split.train_seqs, max_len))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = torch.Generator().manual_seed(seed)
    if use_amp:
        torch.cuda.reset_peak_memory_stats(device)

    losses, step, start = [], 0, time.time()
    for _ in range(epochs):
        order = torch.randperm(len(inputs), generator=generator)
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = model.loss(inputs[idx].to(device), targets[idx].to(device))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            step += 1
            if log_every and step % log_every == 0:
                log(f"  step {step}  loss {np.mean(losses[-log_every:]):.4f}  ({time.time() - start:.0f}s)")
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break
    model.eval()
    return {
        "steps": step,
        "first_loss": float(np.mean(losses[:10])) if losses else None,
        "last_loss": float(np.mean(losses[-10:])) if losses else None,
        "seconds": round(time.time() - start, 1),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated(device) / 2**20) if use_amp else None,
        "n_trainable_params": sum(p.numel() for p in params),
        "n_trainable_encoder_params": sum(p.numel() for p in model.item_encoder.parameters() if p.requires_grad),
    }


@torch.no_grad()
def export_adapted_features(encoder: CachedItemEncoder, item_ids: np.ndarray, n_rows: int) -> np.ndarray:
    """Run every item through the trained adapter; rows are indexed by original item id."""
    encoder.eval()
    vectors = encoder(torch.arange(1, len(item_ids), device=encoder.features.device)).cpu().numpy()
    out = np.zeros((n_rows, vectors.shape[1]), dtype=np.float32)
    out[item_ids[1:]] = vectors
    return out


def write_cost_log(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
