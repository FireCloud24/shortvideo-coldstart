"""Feature cache and encoder plumbing (offline; the real models run on Colab)."""
import json

import numpy as np
import pytest
import torch

from src import features as F


def test_read_titles_keeps_commas_in_titles(tmp_path):
    path = tmp_path / "titles.csv"
    path.write_text("1, first video\n2, eat, drink, sleep\n\n10,last\n", encoding="utf-8")
    assert F.read_titles(path) == {1: "first video", 2: "eat, drink, sleep", 10: "last"}


def test_find_covers_maps_numeric_stems(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ["1.jpg", "sub/22.JPG", "readme.txt", "cover.jpg"]:
        (tmp_path / name).write_bytes(b"x")
    covers = F.find_covers(tmp_path)
    assert sorted(covers) == [1, 22]


def test_check_features_rejects_bad_arrays():
    good = np.ones((4, 3), dtype=np.float32)
    good[2] = 0
    assert F.check_features(good, 4) == {"shape": [4, 3], "zero_rows": 1, "dtype": "float32"}
    with pytest.raises(ValueError, match="shape"):
        F.check_features(good, 5)
    bad = good.copy()
    bad[1, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        F.check_features(bad, 4)


def test_save_then_load_table_is_indexed_by_item_index(tmp_path):
    # Original ids 1..5; the split only uses ids 2, 4, 5 -> indices 1, 2, 3.
    a = np.arange(12, dtype=np.float32).reshape(6, 2) + 1
    b = np.ones((6, 3), dtype=np.float32)
    F.save_features(tmp_path, "a", a, {"model": "test"})
    F.save_features(tmp_path, "b", b, {"model": "test"})
    assert json.loads((tmp_path / "a.meta.json").read_text())["shape"] == [6, 2]

    item_ids = np.array([-1, 2, 4, 5])
    table = F.load_feature_table(tmp_path, ["a", "b"], item_ids)
    assert table.shape == (4, 5)
    assert torch.all(table[0] == 0), "padding row must be zero"
    expected = torch.tensor(a[4]) / torch.tensor(a[4]).norm()
    assert torch.allclose(table[2, :2], expected)
    assert torch.allclose(table[1:, :2].norm(dim=1), torch.ones(3))
    assert torch.allclose(table[1:, 2:].norm(dim=1), torch.ones(3))

    with pytest.raises(ValueError, match="rows"):
        F.load_feature_table(tmp_path, ["a"], np.array([-1, 2, 99]))
    with pytest.raises(ValueError):
        F.load_feature_table(tmp_path, [], item_ids)


def test_scatter_and_batching():
    calls = []

    def encode(batch):
        calls.append(len(batch))
        return np.array([[x, 2 * x] for x in batch], dtype=np.float32)

    vecs = F.embed_in_batches(encode, [1, 2, 3, 4, 5], batch_size=2, log_every=0)
    assert calls == [2, 2, 1] and vecs.shape == (5, 2)
    out = F.scatter_rows([5, 1, 3, 2, 4], vecs, n_rows=7)
    assert out[5].tolist() == [1, 2] and out[4].tolist() == [5, 10]
    assert out[0].tolist() == out[6].tolist() == [0, 0]


def as_tensor(features):
    """Older transformers return the projected features as a tensor; newer ones
    wrap them in an output object whose ``pooler_output`` holds the same thing."""
    return features if isinstance(features, torch.Tensor) else features.pooler_output


def test_cnclip_encoder_matches_the_library_and_passes_checks(tiny_cnclip, tmp_path):
    from PIL import Image

    model, tokenizer, image_processor = tiny_cnclip
    enc = F.CnClipEncoder("tiny", device="cpu", model=model, tokenizer=tokenizer, image_processor=image_processor)

    texts = ["new cat video", "game", "food song dog"]
    text_vecs = enc.encode_texts(texts)
    tok = enc.tokenize(texts)
    with torch.no_grad():
        reference = as_tensor(model.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"]))
    assert np.allclose(text_vecs, reference.numpy(), atol=1e-5)

    rng = np.random.default_rng(0)
    paths = []
    for i in range(3):
        p = tmp_path / f"{i + 1}.jpg"
        Image.fromarray(rng.integers(0, 255, (40, 50, 3), dtype=np.uint8)).save(p)
        paths.append(p)
    image_vecs = enc.encode_images(paths)
    pixels = image_processor(images=[F.open_image(p) for p in paths], return_tensors="pt")["pixel_values"]
    with torch.no_grad():
        reference = as_tensor(model.get_image_features(pixel_values=pixels))
    assert np.allclose(image_vecs, reference.numpy(), atol=1e-5)

    for vecs in (text_vecs, image_vecs):
        assert vecs.shape == (3, 24)
        assert F.check_features(F.scatter_rows([1, 2, 3], vecs, 4), 4)["zero_rows"] == 0


def test_english_clip_text_encoder_returns_one_projected_vector_per_title():
    transformers = pytest.importorskip("transformers")
    from transformers import CLIPConfig, CLIPModel

    from src.features import EnClipTextEncoder

    torch.manual_seed(0)
    config = CLIPConfig(
        text_config=dict(vocab_size=20, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                         intermediate_size=64, max_position_embeddings=16, eos_token_id=19, bos_token_id=18,
                         pad_token_id=0),
        vision_config=dict(image_size=32, patch_size=16, hidden_size=32, num_hidden_layers=1,
                           num_attention_heads=2, intermediate_size=64),
        projection_dim=24)

    def tokenizer(texts, **_):  # word count -> that many tokens, then end-of-text, then padding
        ids = torch.zeros(len(texts), 8, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, text in enumerate(texts):
            n = min(len(text.split()), 6)
            ids[row, :n] = torch.arange(1, n + 1)
            ids[row, n] = 19
            mask[row, : n + 1] = 1
        return {"input_ids": ids, "attention_mask": mask}

    enc = EnClipTextEncoder("tiny", "cpu", model=CLIPModel(config), tokenizer=tokenizer)
    out = enc.encode_texts(["a cat video", "dog", "a cat video"])
    assert out.shape == (3, 24) and np.isfinite(out).all()
    assert np.allclose(out[0], out[2]) and not np.allclose(out[0], out[1])
