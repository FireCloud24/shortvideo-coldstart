"""LoRA / adapter fine-tuning on a tiny offline Chinese-CLIP (CPU)."""
import numpy as np
import pytest
import torch

from src import features as F
from src import finetune as ft
from src.data import build_split
from tests.test_models import N_ITEMS, cyclic_df

pytest.importorskip("peft")

WORDS = ["new", "video", "cat", "dog", "game", "song", "food"]
TITLES = [""] + [f"{WORDS[i % 7]} {WORDS[(i * 3 + 1) % 7]} {WORDS[(i // 7) % 7]}" for i in range(N_ITEMS)]


@pytest.fixture()
def split():
    return build_split(cyclic_df(), cold_item_frac=0.0)


def snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def test_lora_text_trains_only_lora_and_keeps_base_weights(tiny_cnclip, split, tmp_path):
    clip, tokenizer, _ = tiny_cnclip
    base_before = snapshot(clip)
    peft_clip = ft.add_lora(clip, "text", r=4, dropout=0.0)

    trainable = [n for n, p in peft_clip.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n and "text_model" in n for n in trainable)

    encoder = ft.TextItemEncoder(peft_clip, tokenizer, TITLES)
    model = ft.InBatchRecommender(encoder, feat_dim=24, dim=16, max_len=4, n_layers=1, dropout=0.0)
    before = F.CnClipEncoder("tiny", "cpu", model=peft_clip, tokenizer=tokenizer).encode_texts(TITLES[1:])

    cost = ft.finetune(model, split, max_len=4, epochs=30, batch_size=8, lr=5e-3, device=torch.device("cpu"),
                       seed=0, log=lambda *_: None)
    assert cost["steps"] == 30 * 3 and np.isfinite(cost["last_loss"])
    assert cost["last_loss"] < cost["first_loss"]
    assert 0 < cost["n_trainable_encoder_params"] < cost["n_trainable_params"]

    # Base weights are untouched; only the LoRA matrices moved.
    # (peft wraps a targeted layer, so "query.weight" becomes "query.base_layer.weight".)
    after_state = {k.removeprefix("base_model.model.").replace(".base_layer.", "."): v
                   for k, v in peft_clip.state_dict().items() if "lora_" not in k}
    assert set(after_state) == set(base_before)
    for name, value in base_before.items():
        assert torch.equal(after_state[name], value), name

    after = F.CnClipEncoder("tiny", "cpu", model=peft_clip, tokenizer=tokenizer).encode_texts(TITLES[1:])
    assert not np.allclose(before, after), "fine-tuning should change the text embeddings"

    # Save, then reload onto a fresh copy of the base model the way scripts/embed.py does.
    peft_clip.save_pretrained(tmp_path / "lora_text")
    from transformers import ChineseCLIPModel

    fresh = ChineseCLIPModel(clip.config)
    fresh.load_state_dict(base_before)
    reloaded = F.CnClipEncoder("tiny", "cpu", lora_dir=str(tmp_path / "lora_text"), model=fresh, tokenizer=tokenizer)
    assert np.allclose(reloaded.encode_texts(TITLES[1:]), after, atol=1e-5)


def test_lora_image_tower_runs_and_handles_missing_covers(tiny_cnclip, split, tmp_path):
    from PIL import Image

    clip, _, image_processor = tiny_cnclip
    rng = np.random.default_rng(0)
    covers = [None]
    for i in range(1, N_ITEMS + 1):
        if i == 5:
            covers.append(None)  # an item without a cover
            continue
        path = tmp_path / f"{i}.jpg"
        Image.fromarray(rng.integers(0, 255, (36, 36, 3), dtype=np.uint8)).save(path)
        covers.append(path)

    peft_clip = ft.add_lora(clip, "image", r=4)
    trainable = [n for n, p in peft_clip.named_parameters() if p.requires_grad]
    assert trainable and all("vision_model" in n for n in trainable)

    encoder = ft.ImageItemEncoder(peft_clip, image_processor, covers)
    assert encoder(torch.tensor([4, 5, 6])).shape == (3, 24)
    model = ft.InBatchRecommender(encoder, feat_dim=24, dim=16, max_len=4, n_layers=1, dropout=0.0)
    cost = ft.finetune(model, split, max_len=4, epochs=1, batch_size=8, lr=1e-3, device=torch.device("cpu"),
                       seed=0, max_steps=2, log=lambda *_: None)
    assert cost["steps"] == 2 and np.isfinite(cost["last_loss"])


def test_adapter_starts_as_identity_and_exports_by_original_id(split):
    torch.manual_seed(0)
    table = torch.zeros(N_ITEMS + 1, 6)
    table[1:] = torch.nn.functional.normalize(torch.randn(N_ITEMS, 6), dim=1)
    encoder = ft.CachedItemEncoder(table, hidden=8)
    items = torch.arange(1, N_ITEMS + 1)
    assert torch.allclose(encoder(items), table[1:]), "untrained adapter must not change the features"

    model = ft.InBatchRecommender(encoder, feat_dim=6, dim=16, max_len=4, n_layers=1, dropout=0.0)
    cost = ft.finetune(model, split, max_len=4, epochs=20, batch_size=8, lr=1e-2, device=torch.device("cpu"),
                       seed=0, log=lambda *_: None)
    assert cost["last_loss"] < cost["first_loss"]
    assert not torch.allclose(encoder(items), table[1:])

    # Pretend original ids are 10x the index: rows must land at the original id.
    item_ids = np.concatenate([[-1], np.arange(1, N_ITEMS + 1) * 10])
    out = ft.export_adapted_features(encoder, item_ids, n_rows=N_ITEMS * 10 + 1)
    assert out.shape == (N_ITEMS * 10 + 1, 6)
    assert np.allclose(out[30], encoder(torch.tensor([3])).detach().numpy()[0], atol=1e-6)
    assert np.all(out[0] == 0) and np.all(out[5] == 0)
    assert F.check_features(out, out.shape[0])["zero_rows"] == N_ITEMS * 9


def test_in_batch_loss_matches_a_hand_built_case():
    """One user, history [a, b] -> targets [b, c]. With an encoder that returns
    fixed vectors, the loss must equal cross-entropy over the three batch items."""

    class Fixed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.vecs = torch.nn.Parameter(torch.eye(8)[:6])

        def forward(self, items):
            return self.vecs[items]

    torch.manual_seed(0)
    model = ft.InBatchRecommender(Fixed(), feat_dim=8, dim=8, max_len=2, n_layers=1, dropout=0.0).eval()
    seqs = torch.tensor([[2, 4]])
    targets = torch.tensor([[4, 5]])
    loss = model.loss(seqs, targets)

    feats = torch.nn.functional.normalize(model.item_encoder(torch.tensor([2, 4, 5])), dim=1)
    table = model.proj(feats)  # local items 1, 2, 3
    hidden = model.trunk.encode(table[[0, 1]].unsqueeze(0), torch.zeros(1, 2, dtype=torch.bool))[0]
    expected = torch.nn.functional.cross_entropy(hidden @ table.T, torch.tensor([1, 2]))
    assert torch.allclose(loss, expected, atol=1e-6)
