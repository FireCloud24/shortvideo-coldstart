"""report.py: the table is built only from saved runs, with the right arithmetic."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("report", Path(__file__).resolve().parents[1] / "scripts" / "report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

SPLIT = {"n_items": 100, "n_warm_items": 80, "n_cold_items": 20, "n_train_users": 5,
         "n_train_interactions": 50, "n_valid": 5, "n_warm_test": 5, "n_cold_test": 8}


def write_run(root, name, seed, warm, cold, per_item):
    out = root / "runs" / name / f"seed{seed}"
    out.mkdir(parents=True)
    metrics = {s: {"n": 5, "recall@10": v, "ndcg@10": v / 2} for s, v in
               [("valid", warm), ("warm", warm), ("warm_matched", warm * 2), ("cold", cold),
                ("cold_all", cold / 4)]}
    (out / "metrics.json").write_text(json.dumps({
        "name": name, "seed": seed, "split": SPLIT, "metrics": metrics, "train_seconds": 12.0,
        "training": {"epochs_run": 20 + seed, "best_epoch": 15 + seed}, "epochs_cap": 100,
        "git_head": "abc1234def"}))
    # Top-10 ranks of the 8 cold examples (0 = outside the top 10); seed and model change them.
    cold_ranks = np.array([1, 0, 0, 2, 0, 0, 0, 0], dtype=np.uint8)
    if name != "sasrec_id":
        cold_ranks = np.array([1, 1, 3, 1, 0, 2, 0, 1], dtype=np.uint8)
    np.savez_compressed(out / "top10_ranks.npz", warm=np.zeros(5, dtype=np.uint8),
                        warm_matched=np.zeros(5, dtype=np.uint8), cold=cold_ranks, cold_all=cold_ranks)
    (out / "cold_by_item.json").write_text(json.dumps(per_item))


@pytest.fixture()
def results(tmp_path):
    items = lambda r1, r2: [{"item": 1, "n": 6, "recall@10": r1, "ndcg@10": r1}, {"item": 2, "n": 2, "recall@10": r2, "ndcg@10": r2}]
    write_run(tmp_path, "sasrec_id", 0, 0.10, 0.02, items(0.0, 0.0))
    write_run(tmp_path, "sasrec_id", 1, 0.20, 0.04, items(0.0, 0.5))
    write_run(tmp_path, "content_cnclip_image", 0, 0.30, 0.40, items(0.5, 1.0))
    return tmp_path


def test_mean_and_sample_std():
    assert report.mean_std([0.1, 0.2]) == pytest.approx((0.15, 0.0707106781))
    assert report.mean_std([0.3]) == (pytest.approx(0.3), None)
    assert report.fmt([0.1, 0.2]) == "0.1500 ± 0.0707"
    assert report.fmt([0.3]) == "0.3000"
    assert report.fmt([None]) == "n/a"


def test_results_table_rows_and_numbers(results):
    runs = report.load_runs(results / "runs")
    assert report.ordered(runs) == ["sasrec_id", "content_cnclip_image"]
    md = report.results_markdown(runs, [])
    main_table = md.split("## Main table")[1].split("## ")[0]
    rows = [line for line in main_table.splitlines() if line.startswith("| `")]
    assert len(rows) == 2
    id_cells = [c.strip() for c in rows[0].strip("|").split("|")]
    assert id_cells[:4] == ["`sasrec_id`", "2", "0.1500 ± 0.0707", "0.0750 ± 0.0354"]
    assert id_cells[4] == "0.3000 ± 0.1414"  # warm, matched recall
    assert id_cells[6] == "0.0300 ± 0.0141"  # cold recall
    header = next(line for line in md.splitlines() if line.startswith("| Model"))
    chance = next(line for line in md.splitlines() if line.startswith("| *random ranker"))
    assert header.count("|") == rows[0].count("|") == chance.count("|")
    assert "0.1250" in chance and "0.1000" in chance  # 10/80 and 10/100
    assert chance.count("0.5000") == 2                  # 10/20 for both "warm, matched" and "cold"
    assert "Warning" not in md


def test_mixed_splits_are_flagged(results):
    runs = report.load_runs(results / "runs")
    runs["sasrec_id"][0]["split"] = {**SPLIT, "n_cold_items": 21}
    assert "Warning" in report.results_markdown(runs, [])


def test_slices_pool_examples_over_seeds(results):
    runs = report.load_runs(results / "runs")
    meta = {"category": {1: "Game", 2: "Music"}, "title length": {1: "1-5 words", 2: "6-10 words"},
            "_likes": {1: 10.0, 2: 20.0}}
    tables, counts = report.slice_tables(runs, meta)
    # sasrec_id, Music (item 2): seeds give recall 0.0 and 0.5 on 2 examples each -> 0.25
    assert tables["category"]["Music"]["sasrec_id"] == pytest.approx(0.25)
    assert tables["category"]["Game"]["content_cnclip_image"] == pytest.approx(0.5)
    assert counts["category"] == {"Game": 6, "Music": 2}
    md = report.slices_markdown(tables, counts, report.ordered(runs))
    assert "| Music | 2 | 0.2500 | 1.0000 |" in md


def test_main_writes_files_from_saved_runs_only(results, tmp_path):
    assert report.main(["--results-dir", str(results), "--raw-dir", str(tmp_path / "no_data")]) == 0
    assert (results / "results.md").exists() and (results / "warm_vs_cold.png").stat().st_size > 1000
    assert not (results / "error_slices.md").exists()  # no raw metadata -> skipped, not invented
    empty = tmp_path / "empty"
    empty.mkdir()
    assert report.main(["--results-dir", str(empty)]) == 1


def test_training_length_table_shows_each_seed(results):
    runs = report.load_runs(results / "runs")
    md = report.results_markdown(runs, [])
    assert "| `sasrec_id` | 100 | 20/21 | 15/16 |" in md
    assert "| `content_cnclip_image` | 100 | 20 | 15 |" in md


def test_paired_bootstrap_is_paired_and_reproducible():
    rng = np.random.default_rng(0)
    base = rng.random(500)
    better = base + 0.05                      # every example improves by the same amount
    out = report.paired_bootstrap(better, base)
    assert out["diff"] == pytest.approx(0.05) and out["low"] == pytest.approx(0.05) == pytest.approx(out["high"])
    noisy = report.paired_bootstrap(base, rng.permutation(base))   # same values, unpaired: no real difference
    assert noisy["low"] < 0 < noisy["high"]
    assert report.paired_bootstrap(base, better) == report.paired_bootstrap(base, better)
    with pytest.raises(ValueError):
        report.paired_bootstrap(base, base[:10])


def test_paired_rows_use_the_saved_per_example_ranks(results, monkeypatch):
    monkeypatch.setattr(report, "PAIRS", [("content_cnclip_image", "sasrec_id", "cold"),
                                          ("content_cnclip_image", "missing_model", "cold")])
    runs = report.load_runs(results / "runs")
    rows = report.paired_rows(runs)
    assert len(rows) == 1 and rows[0]["seeds"] == [0] and rows[0]["n"] == 8
    ndcg = lambda ranks: np.mean([1 / np.log2(r + 1) if r else 0.0 for r in ranks])
    expected = ndcg([1, 1, 3, 1, 0, 2, 0, 1]) - ndcg([1, 0, 0, 2, 0, 0, 0, 0])
    assert rows[0]["diff"] == pytest.approx(expected)
    assert any("| `content_cnclip_image` | `sasrec_id` | Cold | 8 |" in line for line in report.paired_markdown(rows))


def test_feature_table_is_read_from_meta_files(results):
    features = results / "features"
    features.mkdir()
    (features / "cnclip_image.meta.json").write_text(json.dumps(
        {"model": "toy-clip", "n_items_embedded": 95, "seconds": 12.5, "shape": [101, 512], "zero_rows": 5}))
    md = report.results_markdown(report.load_runs(results / "runs"), [], features_dir=features)
    assert "| `cnclip_image` | toy-clip | 512 | 95 | 5 | 12.5 |" in md
