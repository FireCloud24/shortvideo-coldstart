"""Popularity baseline and SASRec (Kang & McAuley, 2018).

SASRec is written so that *where item vectors come from* is a separate module
(an "item table"). The sequence model only ever sees a table of shape
[n_items + 1, dim] (row 0 is padding): it looks up the user's history in it and
scores every item by a dot product with it. Swapping the table swaps the
experiment: learned IDs here, frozen content features later.
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
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return SASRec(items, m["dim"], cfg["data"]["max_len"], m["n_layers"], m["n_heads"], m["dropout"])
