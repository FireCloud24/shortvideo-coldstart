"""download.py: skip-if-present and manifest behaviour, with a fake server (no network)."""
import hashlib
import importlib.util
import io
import json
import zipfile
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "download", Path(__file__).resolve().parents[1] / "scripts" / "download.py"
)
download = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class FakeResponse(io.BytesIO):
    status = 200

    def __init__(self, body: bytes):
        super().__init__(body)
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make_opener(calls):
    def opener(req):
        calls.append(req.full_url)
        return FakeResponse(b"content of " + req.full_url.encode())

    return opener


def test_manifest_written_and_rerun_skips(tmp_path):
    calls = []
    names = ["a.csv", "b.csv"]
    manifest = download.download_all(tmp_path, names, make_opener(calls))
    assert len(calls) == 2

    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk == manifest
    for name in names:
        body = (tmp_path / name).read_bytes()
        assert on_disk["files"][name]["bytes"] == len(body)
        assert on_disk["files"][name]["sha256"] == hashlib.sha256(body).hexdigest()

    download.download_all(tmp_path, names, make_opener(calls))
    assert len(calls) == 2, "second run must not download anything"


def test_truncated_file_is_fetched_again(tmp_path):
    calls = []
    download.download_all(tmp_path, ["a.csv"], make_opener(calls))
    (tmp_path / "a.csv").write_bytes(b"cut")
    download.download_all(tmp_path, ["a.csv"], make_opener(calls))
    assert len(calls) == 2


def test_adopt_records_a_hand_downloaded_file_without_fetching(tmp_path):
    calls = []
    (tmp_path / "a.csv").write_bytes(b"by hand")
    manifest = download.download_all(tmp_path, ["a.csv"], make_opener(calls), adopt=True)
    assert calls == []
    assert manifest["files"]["a.csv"]["sha256"] == hashlib.sha256(b"by hand").hexdigest()


def test_extract_covers_flattens_and_ignores_other_files(tmp_path):
    with zipfile.ZipFile(tmp_path / download.COVERS, "w") as z:
        z.writestr("covers/1.jpg", b"one")
        z.writestr("2.jpg", b"two")
        z.writestr("../evil.jpg", b"x")
        z.writestr("notes.txt", b"n")
    assert download.extract_covers(tmp_path) == 3
    assert sorted(p.name for p in (tmp_path / "covers").iterdir()) == ["1.jpg", "2.jpg", "evil.jpg"]
    assert not (tmp_path.parent / "evil.jpg").exists()
