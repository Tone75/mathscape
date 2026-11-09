"""Absolute external deadlines, without real subprocesses or random draws."""

from pathlib import Path
import signal
import subprocess

import pytest

from scripts import run_finite_k_guarded as guard


class Process:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.events = []

    def wait(self, timeout):
        self.events.append(("wait", timeout))
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def send_signal(self, value):
        self.events.append(("signal", value))

    def kill(self):
        self.events.append(("kill",))


def run_fake(outcomes, times):
    process, launches = Process(outcomes), []
    clock = iter(times)

    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return process

    result = guard.supervise(["python", "fixed-runner"], cwd=Path("."), started=0.,
                             clock=lambda: next(clock), popen=popen)
    assert len(launches) == 1
    assert launches[0][1]["start_new_session"]
    return result, process.events


def test_success_launches_once_and_counts_setup_time():
    result, events = run_fake([0], [2., 80.])
    assert events == [("wait", 293.)]
    assert result["supervised_wall_seconds"] == 80.
    assert result["exit_code"] == 0
    assert not result["termination_sent"] and not result["kill_sent"]


def test_deadline_requests_graceful_stop_without_retry():
    expired = subprocess.TimeoutExpired("fixed-runner", 295)
    result, events = run_fake([expired, 1], [0., 295., 295.4])
    assert events == [("wait", 295.), ("signal", signal.SIGTERM), ("wait", 3.)]
    assert result["exit_code"] == 124 and result["termination_sent"]
    assert not result["kill_sent"]


def test_stuck_child_is_killed_before_outer_cap():
    expired = subprocess.TimeoutExpired("fixed-runner", 295)
    result, events = run_fake([expired, expired, -9], [0., 295., 298., 298.1])
    assert events == [("wait", 295.), ("signal", signal.SIGTERM), ("wait", 3.), ("kill",), ("wait", 1.)]
    assert result["exit_code"] == 124 and result["kill_sent"]
    assert result["supervised_wall_seconds"] < 300


def test_nonzero_child_status_is_not_retried_or_masked():
    result, events = run_fake([2], [0., 0.1])
    assert result["exit_code"] == 2 and len(events) == 1


def test_interruption_terminates_only_owned_child():
    result, events = run_fake([KeyboardInterrupt(), 1], [0., 10., 10., 10.1])
    assert result["interrupted"] and result["termination_sent"]
    assert events[1] == ("signal", signal.SIGTERM)
    assert events[2] == ("wait", 3.)


def test_existing_attempt_rejected_before_child_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "ROOT", tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    output = tmp_path / "output"
    output.mkdir()
    (output / "attempt.json").write_text("preserved")
    monkeypatch.setattr(guard, "supervise", lambda *args, **kwargs: pytest.fail("must not launch"))
    with pytest.raises(FileExistsError):
        guard.main(["--manifest", str(manifest), "--output", str(output), "--source-commit", "a" * 40])
    assert (output / "attempt.json").read_text() == "preserved"


def test_outer_record_refuses_overwrite_and_private_paths_are_not_in_command(tmp_path):
    path = tmp_path / "record.json"
    guard.write_new_json(path, {"command": ["python", "scripts/run_finite_k_benchmark.py"]})
    with pytest.raises(FileExistsError):
        guard.write_new_json(path, {})
    assert str(tmp_path) not in path.read_text()
