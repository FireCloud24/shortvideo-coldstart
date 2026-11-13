"""scripts/pick_lr.py: the learning rate is chosen on validation only."""
import json

from tests.test_exposure_report import load_script

pick_lr = load_script("pick_lr")


def write(root, family, lr, valid, cold=0.0, best=10, cap=300):
    out = root / f"{family}_lr{lr}" / "seed0"
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(json.dumps({
        "config": {"train": {"lr": lr}}, "epochs_cap": cap, "train_seconds": 1.0,
        "metrics": {"valid": {"ndcg@10": valid}, "cold": {"ndcg@10": cold}},
        "training": {"best_epoch": best, "epochs_run": best + 5}}))


def test_family_comes_from_the_config_name():
    assert pick_lr.family_of("content_cnclip_image") == "content_cnclip_image_text"
    assert pick_lr.family_of("content_qwen_vl") == "content_cnclip_image_text"
    assert pick_lr.family_of("id_content_dropout_050_popcorrect") == "id_content_cnclip_image_text"
    assert pick_lr.family_of("sasrec_id") == "sasrec_id"
    assert pick_lr.family_of("popularity") is None and pick_lr.family_of("knn_cnclip_image") is None


def test_best_validation_wins_whatever_the_cold_result(tmp_path):
    fam = "content_cnclip_image_text"
    write(tmp_path, fam, 0.001, valid=0.030, cold=0.9)   # best on cold, which must not matter
    write(tmp_path, fam, 0.003, valid=0.045, cold=0.1)
    write(tmp_path, fam, 0.01, valid=0.040, cold=0.5)
    assert pick_lr.pick(pick_lr.search_runs(tmp_path, fam)) == 0.003


def test_tie_goes_to_the_smaller_rate_and_no_runs_to_the_default(tmp_path):
    write(tmp_path, "sasrec_id", 0.003, valid=0.05)
    write(tmp_path, "sasrec_id", 0.001, valid=0.05)
    assert pick_lr.pick(pick_lr.search_runs(tmp_path, "sasrec_id")) == 0.001
    assert pick_lr.pick(pick_lr.search_runs(tmp_path, "content_cnclip_image_text")) == pick_lr.DEFAULT_LR


def test_main_prints_the_rate_and_writes_the_table(tmp_path, capsys):
    write(tmp_path, "sasrec_id", 0.001, valid=0.05)
    write(tmp_path, "sasrec_id", 0.01, valid=0.06, best=300)
    assert pick_lr.main(["sasrec_id", "--search-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "0.01"
    assert pick_lr.main(["popularity", "--search-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == "0.001"
    assert pick_lr.main(["--table", "--search-dir", str(tmp_path)]) == 0
    text = (tmp_path / "chosen.md").read_text(encoding="utf-8")
    assert "| IDs only | 0.01 | 0.0600 | 300 (cap) | 305 | yes |" in text
