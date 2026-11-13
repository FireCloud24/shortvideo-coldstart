"""Content and ID+content item encoders on tiny synthetic data."""
import json

import pandas as pd
import pytest
import torch

from src.data import PAD, build_split
from src.models import build_model
from src.train import evaluate, fit, run
from tests.test_models import N_ITEMS, cyclic_df, tiny_cfg


def content_setup(tmp_path, kind):
    """Cyclic data ('next item = previous + 1') plus two cold items.

    Features are one-hot in the item's place in the cycle. Cold item 13 is given
    item 3's features and cold item 14 item 9's: they are new videos that look
    like items 3 and 9. Users whose history ends on item 2 go on to cold 13
    (as they would to 3); users ending on 8 go on to cold 14.
    """
    df = cyclic_df()
    late = []
    for u in range(24):
        last = (u + 5) % N_ITEMS + 1
        if last == 2:
            late.append((u + 1, 13, 10**9 + u))
        if last == 8:
            late.append((u + 1, 14, 10**9 + u))
    df = pd.concat([df, pd.DataFrame(late, columns=df.columns)], ignore_index=True)
    split = build_split(df, cold_item_frac=2 / 14)
    feats = torch.zeros(15, 12)
    for i in range(1, 13):
        feats[i, i - 1] = 1.0
    feats[13, 3 - 1] = 1.0
    feats[14, 9 - 1] = 1.0
    cfg = tiny_cfg(tmp_path, kind=kind, id_dropout=0.5)
    return cfg, split, feats


def test_content_table_has_zero_padding_and_frozen_features(tmp_path):
    cfg, split, feats = content_setup(tmp_path, "sasrec_content")
    model = build_model(cfg, split, feats)
    table = model.items()
    assert table.shape == (15, 32) and torch.all(table[PAD] == 0)
    assert "items.features" not in dict(model.named_parameters())
    assert not model.items.features.requires_grad
    with pytest.raises(ValueError, match="rows"):
        build_model(cfg, split, feats[:-1])
    with pytest.raises(ValueError, match="needs content features"):
        build_model(cfg, split, None)


def test_id_content_drops_the_id_for_cold_items(tmp_path):
    cfg, split, feats = content_setup(tmp_path, "sasrec_id_content")
    model = build_model(cfg, split, feats).eval()
    table = model.items()
    content = model.items.content()
    cold = torch.as_tensor(split.is_cold)
    assert torch.allclose(table[cold], content[cold]), "cold items must be pure content"
    warm = ~cold
    warm[PAD] = False
    assert not torch.allclose(table[warm], content[warm]), "warm items should also use their ID"


def test_id_dropout_only_applies_in_training(tmp_path):
    cfg, split, feats = content_setup(tmp_path, "sasrec_id_content")
    model = build_model(cfg, split, feats)
    model.eval()
    assert torch.equal(model.items(), model.items())
    model.train()
    model.items.content.proj[2].p = 0.0  # isolate ID dropout from ordinary dropout
    torch.manual_seed(0)
    a = model.items()
    b = model.items()
    assert not torch.equal(a, b)


@pytest.mark.parametrize("kind", ["sasrec_content", "sasrec_id_content"])
def test_content_models_rank_cold_items_that_ids_cannot(tmp_path, kind):
    """The project in miniature: with informative content features a cold item
    is ranked correctly; nothing was ever trained on it."""
    cfg, split, feats = content_setup(tmp_path, kind)
    assert split.n_cold == 2 and len(split.cold) == 4

    device = torch.device("cpu")
    torch.manual_seed(0)
    model = build_model(cfg, split, feats)
    fit(model, split, cfg, device, seed=0, log=lambda *_: None)
    ranks = evaluate(model, split.cold, split, "cold", 8, device)
    assert ranks.tolist() == [1, 1, 1, 1], "each user's cold target should beat the other cold item"


def test_run_loads_features_from_disk_and_writes_per_item_results(tmp_path):
    import numpy as np

    from src.features import save_features

    cfg, split, feats = content_setup(tmp_path, "sasrec_content")
    # On disk, features are indexed by ORIGINAL item id (here ids equal indices).
    save_features(tmp_path / "features", "toy", feats.numpy().astype(np.float32), {"model": "one-hot"})
    cfg["data"]["features_dir"] = str(tmp_path / "features")
    cfg["model"]["features"] = ["toy"]
    cfg["train"]["epochs"] = 3

    result = run(cfg, seed=1, split=split, log=lambda *_: None)
    assert result["metrics"]["cold"]["n"] == 4
    out = tmp_path / "runs" / "tiny" / "seed1"
    per_item = json.loads((out / "cold_by_item.json").read_text())
    assert {row["item"] for row in per_item} == {13, 14}
    assert sum(row["n"] for row in per_item) == 4


def test_id_dropout_of_one_is_the_content_only_model(tmp_path):
    cfg, split, feats = content_setup(tmp_path, "sasrec_id_content")
    cfg["model"]["id_dropout"] = 1.0
    model = build_model(cfg, split, feats)
    for mode in (model.eval, model.train):
        mode()
        model.items.content.proj[2].p = 0.0
        assert torch.equal(model.items(), model.items.content()), "every ID must be hidden, always"
    # Same scores as a content-only model holding the same content weights.
    content_cfg, _, _ = content_setup(tmp_path, "sasrec_content")
    content_model = build_model(content_cfg, split, feats)
    shared = {k: v for k, v in model.state_dict().items() if not k.startswith("items.")}
    content_model.load_state_dict(shared, strict=False)  # the sequence model
    content_model.items.load_state_dict(model.items.content.state_dict())
    model.eval(), content_model.eval()
    seqs = torch.tensor([[PAD, 1, 2, 3], [4, 5, 6, 7]])
    assert torch.allclose(model.score_last(seqs), content_model.score_last(seqs), atol=1e-6)


def test_id_dropout_of_zero_never_hides_a_warm_id(tmp_path):
    cfg, split, feats = content_setup(tmp_path, "sasrec_id_content")
    cfg["model"]["id_dropout"] = 0.0
    model = build_model(cfg, split, feats).train()
    model.items.content.proj[2].p = 0.0
    assert torch.equal(model.items(), model.items())
