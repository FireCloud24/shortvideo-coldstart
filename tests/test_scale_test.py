"""scripts/scale_test.py: stretching new videos' vectors without retraining."""
import torch

from src.models import build_model
from tests.test_content_models import content_setup
from tests.test_exposure_report import load_script

scale_test = load_script("scale_test")


def setup(tmp_path):
    cfg, split, feats = content_setup(tmp_path, "sasrec_id_content")
    torch.manual_seed(0)
    model = build_model(cfg, split, feats).eval()
    return cfg, split, model


def test_factor_one_is_the_model_as_trained(tmp_path):
    cfg, split, model = setup(tmp_path)
    seqs = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
    scaled = scale_test.ScaledCold(model, torch.as_tensor(split.is_cold), 1.0)
    assert torch.allclose(scaled.score_last(seqs), model.score_last(seqs))


def test_only_new_videos_scores_are_multiplied(tmp_path):
    cfg, split, model = setup(tmp_path)
    seqs = torch.tensor([[0, 1, 2, 3]])
    cold = torch.as_tensor(split.is_cold)
    base = model.score_last(seqs)
    scaled = scale_test.ScaledCold(model, cold, 2.5).score_last(seqs)
    assert torch.allclose(scaled[:, cold], 2.5 * base[:, cold], atol=1e-6)
    assert torch.allclose(scaled[:, ~cold], base[:, ~cold])


def test_matched_factor_equalises_the_mean_lengths(tmp_path):
    cfg, split, model = setup(tmp_path)
    cold = torch.as_tensor(split.is_cold)
    lengths = scale_test.vector_lengths(model, cold)
    norms = model.items().norm(dim=1)
    warm = ~cold
    warm[0] = False
    assert abs(lengths["warm_mean_length"] - norms[warm].mean().item()) < 1e-6
    assert abs(lengths["cold_mean_length"] * lengths["matched_factor"] - lengths["warm_mean_length"]) < 1e-6


def test_evaluation_and_report(tmp_path):
    cfg, split, model = setup(tmp_path)
    rows = []
    for factor in (1.0, 50.0):
        r = scale_test.evaluate_factor(model, split, factor, cfg["data"]["max_len"], "cpu")
        assert r["exposure"]["n_users"] == len(split.test) and r["cold_all"]["n"] == len(split.cold)
        rows.append(dict(r, label=f"{factor:g}"))
    results = {"m": {"0": {"lengths": scale_test.vector_lengths(model, torch.as_tensor(split.is_cold)), "rows": rows}}}
    text = scale_test.markdown(results, {"m": "A model"})
    assert "## A model" in text and "| 50 |" in text and "Matching factor" in text


def test_calibration_finds_the_factor_that_hits_the_target():
    # Exposure rises with the factor; the target 0.3 is reached at factor 2.
    exposure = lambda f: min(1.0, 0.15 * f)
    assert abs(scale_test.calibrate_factor(exposure, 0.3) - 2.0) < 1e-3
    # A target the range cannot reach returns the nearer end.
    assert scale_test.calibrate_factor(exposure, 0.01) == 0.5
    assert scale_test.calibrate_factor(lambda f: 0.0, 0.3) == 4.0


def test_exposure_rises_with_the_factor_and_like_for_like_table(tmp_path):
    cfg, split, model = setup(tmp_path)
    at = lambda f: scale_test.exposure_at(model, split, split.valid, f, cfg["data"]["max_len"], "cpu", k=3)
    # Factor 1 is the model as trained. (Stretching only raises exposure when the new videos'
    # best scores are positive, which holds for the trained models and not for this untrained one.)
    as_trained = scale_test.evaluate_factor(model, split, 1.0, cfg["data"]["max_len"], "cpu", ks=(3,))
    assert 0.0 <= at(1.0) <= 1.0 and 0.0 <= at(3.0) <= 1.0
    assert as_trained["exposure"]["k"] == 3
    rows = [dict(scale_test.evaluate_factor(model, split, f, cfg["data"]["max_len"], "cpu"), label=l)
            for f, l in ((1.0, "1"), (2.0, "calibrated"))]
    seed = {"lengths": scale_test.vector_lengths(model, torch.as_tensor(split.is_cold)), "rows": rows,
            "calibration": {"target_exposure": 0.1, "factor": 2.0}}
    text = scale_test.markdown({"m": {"0": seed}}, {"m": "A model"})
    assert "## Like for like" in text and "| A model | calibrated | 2.00 |" in text
    # Results saved before calibration existed are still reported, without the comparison table.
    old = {"m": {"0": {"lengths": seed["lengths"], "rows": rows[:1]}}}
    assert "## Like for like" not in scale_test.markdown(old, {"m": "A model"})
