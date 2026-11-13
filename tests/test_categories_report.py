"""scripts/categories.py: per-category cold recall with intervals over videos."""
import json

import numpy as np

from tests.test_exposure_report import load_script

categories = load_script("categories")


def write_items(root, name, seed, rows):
    out = root / "runs" / name / f"seed{seed}"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps({"name": name, "seed": seed, "metrics": {}}))
    (out / "cold_by_item.json").write_text(json.dumps(
        [{"item": i, "n": n, "recall@10": r, "ndcg@10": r} for i, n, r in rows]))


def setup(root):
    # Food: items 1, 2. Film: items 3, 4. The cover does well on food, the title on film.
    write_items(root, "content_cnclip_image", 0, [(1, 10, 1.0), (2, 30, 0.6), (3, 20, 0.0), (4, 20, 0.2)])
    write_items(root, "content_cnclip_image", 1, [(1, 10, 0.8), (2, 30, 0.6), (3, 20, 0.0), (4, 20, 0.0)])
    write_items(root, "content_cnclip_text", 0, [(1, 10, 0.5), (2, 30, 0.2), (3, 20, 0.5), (4, 20, 0.5)])
    (root / "tags.csv").write_text("1,Food\n2,Food\n3,Film\n4,Film\n", encoding="utf-8")


def test_per_item_recall_averages_seeds(tmp_path):
    setup(tmp_path)
    runs = categories.load_runs(tmp_path / "runs")
    cover = categories.per_item_recall(runs["content_cnclip_image"])
    assert cover[1] == (10, 0.9) and abs(cover[4][1] - 0.1) < 1e-12


def test_category_rows_weight_videos_by_their_examples(tmp_path):
    setup(tmp_path)
    runs = categories.load_runs(tmp_path / "runs")
    rows = categories.category_rows(categories.per_item_recall(runs["content_cnclip_image"]),
                                    categories.per_item_recall(runs["content_cnclip_text"]),
                                    {1: "Food", 2: "Food", 3: "Film", 4: "Film"}, n_samples=300)
    food, film = rows                                  # sorted by the cover's lead
    assert food["category"] == "Food" and food["videos"] == 2 and food["examples"] == 40
    assert abs(food["cover"] - (10 * 0.9 + 30 * 0.6) / 40) < 1e-12
    assert abs(food["title"] - (10 * 0.5 + 30 * 0.2) / 40) < 1e-12
    assert abs(film["diff"] - (0.05 - 0.5)) < 1e-12
    for r in rows:
        assert r["cover_ci"][0] <= r["cover"] <= r["cover_ci"][1]
        assert r["diff_ci"][0] <= r["diff"] <= r["diff_ci"][1]


def test_small_categories_are_pooled(tmp_path):
    cover = {i: (10 - i, 0.5) for i in range(1, 6)}
    cats = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}
    rows = categories.category_rows(cover, cover, cats, top=2, n_samples=50)
    assert [r["category"] for r in rows][-1] == categories.OTHER
    assert {r["category"] for r in rows} == {"A", "B", categories.OTHER}
    assert sum(r["videos"] for r in rows) == 5


def test_interval_resamples_videos_not_examples():
    # One video carries every example that succeeds: the interval must be wide.
    n = np.array([100.0, 1.0, 1.0, 1.0])
    recall = np.array([1.0, 0.0, 0.0, 0.0])
    (low, high), = categories.bootstrap_videos(n, recall, n_samples=500)
    assert low == 0.0 and high > 0.9


def test_main_writes_table_and_figure_or_skips(tmp_path):
    assert categories.main(["--results", str(tmp_path), "--categories", str(tmp_path / "tags.csv")]) == 0
    assert not (tmp_path / "categories.md").exists()   # no runs yet: nothing written
    setup(tmp_path)
    assert categories.main(["--results", str(tmp_path), "--categories", str(tmp_path / "tags.csv")]) == 0
    text = (tmp_path / "categories.md").read_text(encoding="utf-8")
    assert "| Food | 2 | 40 | 0.675 |" in text
    assert (tmp_path / "categories.png").stat().st_size > 0
