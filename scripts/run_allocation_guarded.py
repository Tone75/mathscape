"""Run the sole RQ-M08 CLI under an external standard-library wall guard.

This launcher imports no numerical engine and creates exactly one child. The
runner has a 294-second soft stop; this parent requests termination at 295,
kills its own child at 298, and reserves time for the external 300-second cap.
The actual full CLI is measured independently with /usr/bin/time -p.
"""

from time import perf_counter
_START = perf_counter()

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOFT_SECONDS = 295.0
KILL_SECONDS = 298.0
FINISH_SECONDS = 299.0


def supervise(command, *, cwd, clock=perf_counter, popen=subprocess.Popen,
              started=None):
    """Wait for one child, escalating only that child's termination on expiry.

    Timeouts are absolute deadlines from launcher entry, including setup and
    child startup. Fake clocks/processes let tests exercise the deadlines
    without launching an experiment or sleeping.
    """
    started = clock() if started is None else started
    child = popen(command, cwd=cwd, start_new_session=True)
    result = dict(child_launches=1, termination_sent=False, kill_sent=False, interrupted=False)

    def remaining(deadline):
        return max(0.001, deadline - (clock() - started))

    try:
        code = child.wait(timeout=remaining(SOFT_SECONDS))
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        result["interrupted"] = isinstance(error, KeyboardInterrupt)
        result["termination_sent"] = True
        child.send_signal(signal.SIGTERM)
        kill_deadline = min(KILL_SECONDS, clock() - started + 3) if result["interrupted"] else KILL_SECONDS
        try:
            code = child.wait(timeout=remaining(kill_deadline))
        except subprocess.TimeoutExpired:
            result["kill_sent"] = True
            child.kill()
            code = child.wait(timeout=remaining(FINISH_SECONDS))
    result.update(child_exit_code=code, supervised_wall_seconds=clock() - started)
    result["exit_code"] = 124 if result["termination_sent"] else code
    return result


def project_path(path):
    path = Path(path).resolve()
    if ROOT not in path.parents:
        raise ValueError("Input/output must be inside the project")
    return path


def write_new_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "evidence/rq_m08/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "evidence/rq_m08")
    args = parser.parse_args(argv)
    if len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit):
        parser.error("source-commit must be a full hexadecimal commit SHA")
    manifest, output = project_path(args.manifest), project_path(args.output)
    if not manifest.is_file():
        raise FileNotFoundError("Preregistered manifest is missing")
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in
           ("outer_attempt.json", "outer_resources.json", "attempt.json", "resources.json", "panels", "summary.csv")):
        raise FileExistsError("Existing experiment attempt/results are preserved")
    relative_manifest, relative_output = str(manifest.relative_to(ROOT)), str(output.relative_to(ROOT))
    arguments = ["scripts/run_allocation_benchmark.py", "--run", "--manifest", relative_manifest,
                 "--output", relative_output, "--source-commit", args.source_commit]
    attempt = dict(question_id="RQ-M08", source_git_sha=args.source_commit,
                   started_utc=datetime.now(timezone.utc).isoformat(), planned_child_launches=1,
                   command=["python", *arguments], soft_termination_seconds=SOFT_SECONDS,
                   hard_kill_seconds=KILL_SECONDS, finish_deadline_seconds=FINISH_SECONDS,
                   outer_cap_seconds=300, scope="one guarded main attempt; no scientific retry")
    write_new_json(output / "outer_attempt.json", attempt)
    result = dict(attempt, status="stopped", child_launches=0)
    try:
        result.update(supervise([sys.executable, *arguments], cwd=ROOT, started=_START))
        result["status"] = "completed" if result["exit_code"] == 0 else "stopped"
    except Exception as error:
        # A class name identifies launcher errors without disclosing local paths.
        result.update(exit_code=1, launcher_error=type(error).__name__)
        raise
    finally:
        result.update(launcher_wall_seconds=perf_counter() - _START,
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      wall_scope="module entry before imports through child completion; interpreter startup/final record write measured by external time")
        write_new_json(output / "outer_resources.json", result)
    print(json.dumps({key: result[key] for key in ("status", "exit_code", "launcher_wall_seconds")}))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
