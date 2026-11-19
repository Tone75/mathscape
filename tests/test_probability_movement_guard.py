"""Synthetic checks of attempt reservation and fixed resource boundaries."""
import pytest

from scripts.run_probability_movement import ACCEPTED, PUBLIC, RSS, WALL, input_paths, limit, write_once


@pytest.mark.parametrize("elapsed,rss,size,want", [
    (1,20,100,None), (WALL-1,1,0,"wall"), (1,RSS,2,"rss"),
    (1,1,PUBLIC,None), (1,1,PUBLIC+1,"storage"),
])
def test_limits(elapsed,rss,size,want):
    assert limit(elapsed,rss,size) == want


def test_unique_sources_and_originals():
    paths=input_paths()
    assert len(ACCEPTED)==40 and len(set(paths))==len(paths)
    for group,count in (("/references/",60),("/comparisons/",60),("/streams/",36),("/panels/",6)):
        assert sum(group in p for p in paths)==count


def test_exclusive_attempt(tmp_path):
    target=tmp_path/"main_attempt.json"
    write_once(target, {"attempt":1})
    with pytest.raises(FileExistsError):
        write_once(target, {"attempt":2})
    assert '"attempt": 1' in target.read_text()


def test_header_schema_records_dtype_and_dynamic_support_shape(tmp_path, monkeypatch):
    import numpy as np
    from scripts import run_probability_movement as guard

    monkeypatch.setattr(guard, "ROOT", tmp_path)
    folder = tmp_path/"refs"
    folder.mkdir()
    np.savez(folder/"a.npz", support=np.array([0.0,1.0]), count=np.array(2))
    np.savez(folder/"b.npz", support=np.array([0.0,0.5,1.0]), count=np.array(3))
    fields = guard.npz_schema(["refs/a.npz", "refs/b.npz"])["refs"]
    assert fields["support"] == [
        {"shape":[2],"dtype":"<f8","order":"C"}, {"shape":[3],"dtype":"<f8","order":"C"}]
    assert fields["count"] == [{"shape":[],"dtype":"<i8","order":"C"}]
    np.savez(folder/"different.npz", support=np.array([0.0]), extra=np.array(1), count=np.array(1))
    with pytest.raises(ValueError,match="fields differ"):
        guard.npz_schema(["refs/a.npz", "refs/different.npz"])


def test_invalid_preanalysis_head_stops_before_loading_any_scientific_input(tmp_path, monkeypatch):
    from scripts import run_probability_movement as guard

    monkeypatch.setattr(guard, "OUT", tmp_path)
    with pytest.raises(ValueError, match="freeze HEAD"):
        guard.verify("fictional-input")
    assert list(tmp_path.iterdir()) == []
