"""scripts/exposure.py: the exposure table is built only from saved runs."""
import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exposure = load_script("exposure")


def write_run(root, name, seed, hist, warm=0.1, extra=None):
    out = root / "runs" / name / f"seed{seed}"
    out.mkdir(parents=True)
    n, k = sum(hist), len(hist) - 1
    result = {
        "name": name, "seed": seed,
        "metrics": {"warm": {"n": n, "recall@10": warm, "ndcg@10": warm / 2}},
        "exposure": {"k": k, "n_users": n, "users_by_cold_slots": hist,
                     "cold_slot_share": sum(c * h for c, h in enumerate(hist)) / (n * k),
                     "users_with_any_cold": sum(hist[1:]) / n, "cold_catalogue_share": 0.1},
        "config": {"model": {"kind": "sasrec_id_content", "id_dropout": 0.3}, "train": {}},
    }
    result.update(extra or {})
    (out / "metrics.json").write_text(json.dumps(result))


def hist(zero, one, three):
    h = [0] * 11
    h[0], h[1], h[3] = zero, one, three
    return h


def test_exposure_rows_pool_seeds(tmp_path):
    write_run(tmp_path, "sasrec_id", 0, hist(100, 0, 0))
    write_run(tmp_path, "content_cnclip_image", 0, hist(60, 30, 10))
    write_run(tmp_path, "content_cnclip_image", 1, hist(80, 20, 0))
    rows = {r["name"]: r for r in exposure.exposure_rows(exposure.load_runs(tmp_path / "runs"))}
    assert rows["sasrec_id"]["share"] == 0.0 and rows["sasrec_id"]["users_with_any"] == 0.0
    r = rows["content_cnclip_image"]
    # seed 0: (30 + 3*10) / 1000 = 0.06; seed 1: 20 / 1000 = 0.02; mean 0.04
    assert abs(r["share"] - 0.04) < 1e-12 and r["seeds"] == 2
    assert abs(r["users_with_any"] - 60 / 200) < 1e-12        # pooled users
    assert abs(r["users_with_3_plus"] - 10 / 200) < 1e-12
    assert r["low"] <= 0.04 <= r["high"]


def test_exposure_main_writes_table_and_figure(tmp_path):
    write_run(tmp_path, "sasrec_id", 0, hist(100, 0, 0))
    write_run(tmp_path, "content_cnclip_image", 0, hist(60, 30, 10))
    assert exposure.main(["--results", str(tmp_path)]) == 0
    text = (tmp_path / "exposure.md").read_text(encoding="utf-8")
    assert "| Cover only | 1 | 6.00% |" in text and "| IDs only | 1 | 0.00% |" in text
    assert (tmp_path / "exposure.png").stat().st_size > 0


def test_runs_without_exposure_are_left_out(tmp_path):
    out = tmp_path / "runs" / "popularity" / "seed0"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps({"name": "popularity", "seed": 0, "metrics": {}}))
    assert exposure.exposure_rows(exposure.load_runs(tmp_path / "runs")) == []
    assert exposure.main(["--results", str(tmp_path)]) == 0
