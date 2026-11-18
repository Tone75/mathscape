"""Freeze the known RQ-M10 inputs or supervise one RQ-M11 analysis/audit.

Only the worker imports scientific code. Reservations and resource records are
exclusive so an interrupted or completed attempt remains available for review.
"""
from time import monotonic
START = monotonic()

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/rq_m10"
EVIDENCE = ROOT / "evidence/rq_m11"
ACCEPTED = "bd01ae7ff55b1e8593a4b3bd003587db2732c050"
WALL = 180
RSS = 2 * 1024**3
PUBLIC = 50 * 1024**2
FROZEN = (
    "scripts/run_enumeration.py", "scripts/analyze_enumerated_reference.py",
    "scripts/audit_enumerated_reference.py", "mathscape/enumerated_reference.py",
    "tests/test_enumerated_reference.py", "tests/test_enumerated_audit.py",
    "tests/test_enumeration_guard.py", "tests/test_enumeration_integration.py",
    "evidence/rq_m11/SPEC.md",
    "evidence/rq_m11/CODEX_RECEIPT.md", "evidence/rq_m11/view_schema.md",
)

def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_once(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")

def environment():
    requirements = (ROOT / "requirements.txt").read_text().splitlines()
    packages = {line.split("==")[0]: version(line.split("==")[0]) for line in requirements if line}
    return dict(python=platform.python_version(), system=platform.system(),
                architecture=platform.machine(), os_release=platform.release(),
                packages=packages)

def git_hashes(commit):
    tree = git("ls-tree", "-r", "-z", commit).decode().split("\0")
    return {row.split("\t", 1)[1]: row.split("\t", 1)[0].split()[2] for row in tree if row}

def check_committed_files(paths, commit):
    tree = git_hashes(commit)
    result = {}
    for name in sorted(paths):
        data = (ROOT / name).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if tree.get(name) != blob:
            raise ValueError(f"Source mismatch at {name}")
        result[name] = {"git_blob": blob, "sha256": hashlib.sha256(data).hexdigest()}
    return result

def manifest():
    reference = json.loads((SOURCE / "manifest.json").read_text())
    paths = set(reference["source_files_sha256"]) | {"mathscape/__init__.py"}
    for name in ("manifest.json", "originals/receipt.json", "hypotheses.json", "REPORT.md",
                 "resources.json", "independent_audit.json"):
        paths.add(str((SOURCE / name).relative_to(ROOT)))
    for folder, expected in (("panels", 6), ("originals", 6), ("streams", 36)):
        files = sorted((SOURCE / folder).glob("*.npz"))
        if len(files) != expected:
            raise ValueError(f"Unexpected number of {folder} files")
        paths.update(str(f.relative_to(ROOT)) for f in files)
    inputs = check_committed_files(paths, ACCEPTED)
    records = json.loads((SOURCE / "originals/receipt.json").read_text())["records"]
    return {
        "question": "RQ-M11-v1", "accepted_head": ACCEPTED, "known_inputs": inputs,
        "original_ids_and_receipt": records, "environment": environment(),
        "frozen_sources": {name: digest(ROOT / name) for name in FROZEN},
        "panel_axes": ["outer_id", "budget_repeat_id", "method", "stream_id"],
        "panel_shape": [10,20,3,120], "panel_method_order": ["t","percentile","bca"],
        "sample_axis": [10,5], "selected_indices_axis": [4999,5],
        "n": 5, "alpha": .05, "finite_B": 4999, "ordered_n5": 3125,
        "planned_groups": {"gamma": 30, "lognormal": 30}, "series_per_original": 2400,
        "unique_saved_series": 144000, "new_draws": 0, "prng_replays": 0,
        "selected_finite_streams": {"outer":0,"repeat":0,"stream_ids":[0,29,30,59,60,119]},
        "contract": {"mean": "binary64 np.mean(x[indices], axis=1), base5 row order",
                     "tie": "exact numeric equality, half tie, signed zeros equal",
                     "reference": "integer cumulative counts vs exact q.as_integer_ratio(); q0=min, q1=max",
                     "linear": "unweighted all 3125 elements, same adjusted q",
                     "primary": "both BCa endpoint RMSE / sample SE <= 0.10, inclusive before rounding",
                     "group": "all 30 complete, count >= 27 supports; invalid or missing holds the group",
                     "overall": "fail if either complete group fails; both support required; otherwise hold",
                     "audit": "unverified if incomplete or any discrepancy; all 60 x 2400 saved comparisons; 36 indices connections",
                     "numeric": {"rtol":2e-10,"atol":2e-12,"identity_rtol":2e-12,"identity_atol":2e-12},
                     "exact": ["ID","hash","frequency","support","tie","CDF selection","classification","rank"]},
        "resources": {"main_seconds":WALL,"audit_seconds":WALL,"rss_bytes_each":RSS,
                      "additional_public_bytes":PUBLIC,"main_launches":1,"smoke":0,
                      "correction_only_extra_attempts":1,"main_cumulative_seconds":360},
        "output_schema": "view_schema.md, SPEC.md and the frozen primary writer specify persisted fields and axes",
    }

def verify(freeze):
    if len(freeze) != 40 or git("rev-parse", "HEAD").decode().strip() != freeze:
        raise ValueError("Run from the full preregistration commit")
    subprocess.run(["git", "merge-base", "--is-ancestor", ACCEPTED, freeze], cwd=ROOT, check=True)
    stored = json.loads((EVIDENCE / "manifest.json").read_text())
    check_committed_files((*FROZEN, "evidence/rq_m11/manifest.json"), freeze)
    if manifest() != stored:
        raise ValueError("Frozen manifest, environment or accepted inputs changed")
    return stored

def directory_size(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

def other_public_bytes():
    # Include the full changed source/document sizes, a conservative measure
    # of public additions. The science directory is measured live by the guard.
    changed = git("diff", "--name-only", "-z", ACCEPTED).split(b"\0")
    new = git("ls-files", "--others", "--exclude-standard", "-z").split(b"\0")
    paths = {ROOT / name.decode() for name in changed + new if name}
    return sum(path.stat().st_size for path in paths if EVIDENCE not in path.parents and path.is_file())


def peak_bytes(value):
    return int(value if sys.platform == "darwin" else value * 1024)

def worker(action, freeze, output):
    verified = verify(freeze)
    if action == "main":
        from scripts.analyze_enumerated_reference import analyze
        write_once(output / "input_receipt.json", {
            "accepted_head": ACCEPTED, "preanalysis_commit": freeze,
            "manifest_sha256": digest(EVIDENCE / "manifest.json"),
            "inputs": verified["known_inputs"], "originals": verified["original_ids_and_receipt"],
        })
        analyze(SOURCE, output)
    else:
        from scripts.audit_enumerated_reference import audit
        report = audit(SOURCE, output)
        write_once(output / "independent_audit.json", report)

def over_limit(elapsed, rss, evidence_bytes):
    if elapsed >= WALL - 2:
        return "wall"
    if rss >= RSS:
        return "rss"
    if evidence_bytes >= PUBLIC:
        return "storage"
    return None

def supervise(action, freeze, output):
    if action == "audit":
        main_record = json.loads((output / "main_resources.json").read_text())
        if main_record["status"] != "completed" or main_record["preanalysis_commit"] != freeze:
            raise ValueError("Completed matching main result required")
    overhead = other_public_bytes()
    record = {"phase": action, "preanalysis_commit": freeze, "new_draws": 0,
              "wall_cap_seconds": WALL, "peak_rss_cap_bytes": RSS}
    write_once(output / f"{action}_attempt.json", record)
    code = "from scripts.run_enumeration import worker; import sys; from pathlib import Path; worker(sys.argv[1], sys.argv[2], Path(sys.argv[3]))"
    child = subprocess.Popen([sys.executable, "-c", code, action, freeze, str(output)], cwd=ROOT)
    cause = None
    try:
        while child.poll() is None:
            # The worker runs numerical work in-process. Short-lived Git/ps
            # readers only verify metadata. wait/RUSAGE also captures its high-water mark.
            ps = subprocess.run(["ps","-o","rss=","-p",str(child.pid)], capture_output=True)
            resident = int(ps.stdout.strip() or b"0") * 1024
            cause = over_limit(monotonic() - START, resident, overhead + directory_size(EVIDENCE))
            if cause:
                child.kill()
                break
            time.sleep(.05)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        elapsed = monotonic() - START
        peak = max(peak_bytes(resource.getrusage(who).ru_maxrss)
                   for who in (resource.RUSAGE_CHILDREN, resource.RUSAGE_SELF))
        size = directory_size(EVIDENCE)
        success = child.returncode == 0 and not cause and elapsed < WALL and peak <= RSS and size + overhead <= PUBLIC
        record.update(status="completed" if success else "stopped", exit_code=child.returncode,
                      guard_stop=cause, wall_seconds=elapsed, peak_rss_bytes=peak,
                      evidence_bytes_at_finish=size, other_public_bytes_at_start=overhead)
        write_once(output / f"{action}_resources.json", record)
    return 0 if success else 1

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze","main","audit"))
    parser.add_argument("--commit")
    args = parser.parse_args()
    if args.phase == "freeze":
        write_once(EVIDENCE / "manifest.json", manifest())
        return 0
    if not args.commit:
        parser.error("--commit is required to run")
    return supervise(args.phase, args.commit, EVIDENCE)

if __name__ == "__main__":
    raise SystemExit(main())
