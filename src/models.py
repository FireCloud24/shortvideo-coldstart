"""Popularity baseline and SASRec (Kang & McAuley, 2018).

SASRec is written so that *where item vectors come from* is a separate module
(an "item table"). The sequence model only ever sees a table of shape
[n_items + 1, dim] (row 0 is padding): it looks up the user's history in it and
scores every item by a dot product with it. Swapping the table swaps the
experiment: learned IDs, frozen content features, or a gated sum of both.
"""
from __future__ import annotations

import torch
from torch import nn

PAD = 0


class Popularity(nn.Module):
    """Scores every item by its number of training interactions, for every user."""

    def __init__(self, n_items: int):
        super().__init__()
        self.register_buffer("counts", torch.zeros(n_items + 1))

    def fit(self, train_seqs) -> "Popularity":
        counts = torch.zeros_like(self.counts)
        for seq in train_seqs:
            counts.index_add_(0, torch.as_tensor(seq, dtype=torch.long), torch.ones(len(seq)))
        counts[PAD] = 0
        self.counts.copy_(counts)
        return self

    @torch.no_grad()
    def score_last(self, seqs: torch.Tensor) -> torch.Tensor:
        return self.counts.unsqueeze(0).expand(seqs.shape[0], -1)


class IdItems(nn.Module):
    """One learned vector per item ID. A cold item's row is never trained."""

    def __init__(self, n_items: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_items + 1, dim, padding_idx=PAD)
        nn.init.normal_(self.emb.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()

    def forward(self) -> torch.Tensor:
        return self.emb.weight


def _zero_padding_row(table: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros(table.shape[0], 1, dtype=torch.bool, device=table.device)
    mask[PAD] = True
    return table.masked_fill(mask, 0.0)


class ContentItems(nn.Module):
    """Item vector = small trained projection of a frozen content feature.

    The features are a fixed buffer (never updated). Only the projection is
    trained, on warm items; a cold item gets its vector the same way, from its
    own cover/title features, with no ID and no history needed.
    """

    def __init__(self, features: torch.Tensor, dim: int, dropout: float = 0.2):
        super().__init__()
        self.register_buffer("features", features.float(), persistent=False)
        self.proj = nn.Sequential(
            nn.Linear(features.shape[1], 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )

    def forward(self) -> torch.Tensor:
        return _zero_padding_row(self.proj(self.features))


class IdContentItems(nn.Module):
    """Item vector = content vector + gate * ID vector (a gated sum).

    The gate is a per-dimension sigmoid computed from both vectors. Two things
    make this usable for cold items:

    * ``is_cold`` items have no trained ID, so their ID part is always dropped
      and they fall back to pure content. Knowing an item is new is legitimate
      at serving time: it is exactly the items with no interactions yet.
    * In training, each warm item's ID is hidden with probability ``id_dropout``
      so the sequence model also learns to work from content alone.
    """

    def __init__(self, features: torch.Tensor, n_items: int, dim: int, is_cold: torch.Tensor,
                 id_dropout: float = 0.3, dropout: float = 0.2):
        super().__init__()
        self.content = ContentItems(features, dim, dropout)
        self.ids = IdItems(n_items, dim)
        self.gate = nn.Linear(2 * dim, dim)
        self.id_dropout = id_dropout
        self.register_buffer("is_cold", is_cold.bool(), persistent=False)

    def forward(self) -> torch.Tensor:
        content = self.content()
        ids = self.ids()
        keep = ~self.is_cold
        if self.training and self.id_dropout > 0:
            keep = keep & (torch.rand(keep.shape, device=keep.device) >= self.id_dropout)
        gate = torch.sigmoid(self.gate(torch.cat([ids, content], dim=1)))
        table = content + keep.unsqueeze(1).to(ids.dtype) * gate * ids
        return _zero_padding_row(table)


class SASRec(nn.Module):
    def __init__(self, items: nn.Module, dim: int, max_len: int, n_layers: int = 2, n_heads: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.items = items
        self.max_len = max_len
        self.n_heads = n_heads
        self.pos = nn.Embedding(max_len, dim)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=4 * dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def encode(self, seq_emb: torch.Tensor, is_pad: torch.Tensor) -> torch.Tensor:
        """seq_emb [B, L, dim], is_pad [B, L] bool -> hidden states [B, L, dim].

        Position t may attend to positions <= t that are not padding. Sequences
        are left-padded, so a padding position has nothing valid to attend to; it
        is allowed to see itself only to keep the softmax finite, and its output
        is never used.
        """
        B, L, _ = seq_emb.shape
        positions = torch.arange(self.max_len - L, self.max_len, device=seq_emb.device)
        x = self.drop(seq_emb + self.pos(positions))
        x = x.masked_fill(is_pad.unsqueeze(-1), 0.0)

        causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=seq_emb.device))
        allowed = causal.unsqueeze(0) & ~is_pad.unsqueeze(1)  # [B, query, key]
        allowed = allowed | torch.eye(L, dtype=torch.bool, device=seq_emb.device)
        blocked = (~allowed).repeat_interleave(self.n_heads, dim=0)  # [B * heads, L, L]
        return self.norm(self.encoder(x, mask=blocked))

    def forward(self, seqs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """seqs [B, L] item indices -> (hidden [B, L, dim], item table [n_items + 1, dim])."""
        table = self.items()
        hidden = self.encode(table[seqs], seqs == PAD)
        return hidden, table

    @torch.no_grad()
    def score_last(self, seqs: torch.Tensor) -> torch.Tensor:
        """Scores for the next item after each sequence: [B, n_items + 1]."""
        hidden, table = self(seqs)
        return hidden[:, -1] @ table.T


def build_model(cfg: dict, split, features: torch.Tensor | None = None) -> nn.Module:
    m = cfg["model"]
    kind = m["kind"]
    if kind == "popularity":
        return Popularity(split.n_items).fit(split.train_seqs)
    if kind == "sasrec_id":
        items = IdItems(split.n_items, m["dim"])
    elif kind in ("sasrec_content", "sasrec_id_content"):
        if features is None:
            raise ValueError(f"{kind} needs content features (model.features in the config)")
        if features.shape[0] != split.n_items + 1:
            raise ValueError(f"features have {features.shape[0]} rows, expected {split.n_items + 1}")
        if kind == "sasrec_content":
            items = ContentItems(features, m["dim"], m["dropout"])
        else:
            items = IdContentItems(features, split.n_items, m["dim"], torch.as_tensor(split.is_cold),
                                   m["id_dropout"], m["dropout"])
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return SASRec(items, m["dim"], cfg["data"]["max_len"], m["n_layers"], m["n_heads"], m["dropout"])
