import json

import pytest

from scripts import run_enumeration as guard


@pytest.mark.parametrize(("time", "rss", "storage", "expected"), [
    (1, 1, 1, None), (178, 1, 1, "wall"),
    (2, 2 * 1024**3, 1, "rss"), (2, 1, 50 * 1024**2, "storage"),
])
def test_guard_budgets(time, rss, storage, expected):
    assert guard.over_limit(time, rss, storage) == expected


def test_exclusive_launch_receipt(tmp_path):
    receipt = tmp_path / "attempt.json"
    guard.write_once(receipt, {"attempt": 1})
    with pytest.raises(FileExistsError):
        guard.write_once(receipt, {"attempt": 2})
    assert json.loads(receipt.read_text()) == {"attempt": 1}


def test_no_incomplete_analysis_audited(tmp_path):
    (tmp_path / "main_resources.json").write_text('{"status":"stopped"}')
    with pytest.raises(ValueError, match="Completed"):
        guard.supervise("audit", "a" * 40, tmp_path)
    assert not (tmp_path / "audit_attempt.json").exists()


def test_accepted_git_blob_and_sha256(tmp_path, monkeypatch):
    import hashlib

    source = b"accepted evidence\n"
    path = tmp_path / "known.txt"
    path.write_bytes(source)
    blob = hashlib.sha1(b"blob 18\0" + source).hexdigest()
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    monkeypatch.setattr(guard, "git_hashes", lambda commit: {"known.txt": blob})
    assert guard.check_committed_files(["known.txt"], "fixture")["known.txt"] == {
        "git_blob": blob, "sha256": hashlib.sha256(source).hexdigest()}
    path.write_bytes(b"modified\n")
    with pytest.raises(ValueError, match="Source mismatch"):
        guard.check_committed_files(["known.txt"], "fixture")


def test_guard_rejects_wrong_freeze_before_science(monkeypatch):
    monkeypatch.setattr(guard, "git", lambda *args: b"f" * 40)
    with pytest.raises(ValueError, match="preregistration commit"):
        guard.verify("0" * 40)
