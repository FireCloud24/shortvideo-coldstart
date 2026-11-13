"""The popularity correction: train on score + log(popularity share), serve on the score alone."""
import math

import numpy as np
import torch
from torch import nn

from src.data import PAD
from src.train import popularity_log_prior, run, train_epoch
from tests.test_content_models import content_setup


def test_log_prior_is_the_log_share_of_training_interactions():
    seqs = [np.array([1, 1, 2]), np.array([1, 3])]          # item 1: 3 of 5, items 2 and 3: 1 of 5
    prior = popularity_log_prior(seqs, n_items=4)
    assert abs(prior[1].item() - math.log(3 / 5)) < 1e-6
    assert abs(prior[2].item() - math.log(1 / 5)) < 1e-6
    assert prior[PAD] == 0 and prior[4] == 0                 # never seen: no prior
    assert abs(torch.exp(prior[1:4]).sum().item() - 1.0) < 1e-6


class OneScorePerItem(nn.Module):
    """The simplest model with something to learn: one free score per item, the same
    for every user. Left alone, the best it can do is learn how popular each item is."""

    def __init__(self, n_items):
        super().__init__()
        self.scores = nn.Parameter(torch.zeros(n_items + 1, 1))

    def forward(self, seqs):
        return torch.ones(*seqs.shape, 1), self.scores


def fit_scores(log_prior):
    torch.manual_seed(0)
    n_items = 4
    # Next-item targets: item 1 is 8 times as common as items 2, 3 and 4.
    targets = torch.tensor([[1]] * 80 + [[2]] * 10 + [[3]] * 10 + [[4]] * 10)
    inputs = torch.ones_like(targets)
    model = OneScorePerItem(n_items)
    train_items = torch.tensor([False, True, True, True, True])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(60):
        train_epoch(model, inputs, targets, train_items, optimizer, 55, "cpu", torch.Generator().manual_seed(0),
                    log_prior)
    return model.scores.detach().flatten()[1:]


def test_without_the_correction_the_model_learns_popularity():
    scores = fit_scores(None)
    assert scores[0] - scores[1:].max() > 1.5                # log(8) is about 2.08


def test_with_the_correction_the_served_scores_carry_no_popularity():
    prior = popularity_log_prior([np.array([1] * 80 + [2] * 10 + [3] * 10 + [4] * 10)], n_items=4)
    scores = fit_scores(prior)
    assert (scores.max() - scores.min()).abs() < 0.2          # the prior explained it all


def test_run_records_the_correction_and_still_evaluates(tmp_path):
    cfg, split, feats = content_setup(tmp_path, "sasrec_id_content")
    cfg["train"].update({"pop_correction": 1.0, "epochs": 2})
    result = run(cfg, seed=0, split=split, features=feats, log=lambda *_: None)
    assert result["config"]["train"]["pop_correction"] == 1.0
    assert result["metrics"]["warm"]["recall@10"] is not None and result["exposure"]["n_users"] > 0
