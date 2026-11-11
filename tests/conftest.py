"""Shared fixtures: a tiny randomly initialised Chinese-CLIP that needs no download."""
import pytest
import torch


@pytest.fixture()
def tiny_cnclip(tmp_path):
    """(model, tokenizer, image_processor) with the real Chinese-CLIP architecture,
    2 small layers per tower, random weights."""
    transformers = pytest.importorskip("transformers")
    from transformers import BertTokenizer, ChineseCLIPConfig, ChineseCLIPImageProcessor, ChineseCLIPModel

    torch.manual_seed(0)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "new", "video", "cat", "dog", "game", "song", "food"]
    vocab_file = tmp_path / "vocab.txt"
    vocab_file.write_text("\n".join(vocab), encoding="utf-8")
    tokenizer = BertTokenizer(str(vocab_file))

    small = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
    config = ChineseCLIPConfig(
        # Default init (std 0.02) makes every title embed to almost the same vector
        # (cosine > 0.9999), which gives a model nothing to learn from. A wide
        # init and no dropout make this toy tower informative.
        text_config=dict(vocab_size=len(vocab), max_position_embeddings=64, hidden_dropout_prob=0.0,
                         attention_probs_dropout_prob=0.0, initializer_range=0.5, **small),
        vision_config=dict(image_size=32, patch_size=16, **small),
        projection_dim=24,
    )
    model = ChineseCLIPModel(config).eval()
    image_processor = ChineseCLIPImageProcessor(size={"height": 32, "width": 32},
                                                crop_size={"height": 32, "width": 32})
    return model, tokenizer, image_processor
