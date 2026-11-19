"""Freeze the accepted inputs, or supervise a single deterministic M12 stage."""
from time import monotonic

START = monotonic()

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_enumeration import check_committed_files, digest, environment, git, write_once

ACCEPTED = "0deb14fa313dc27380afbb43e6b132386aa40206"
OUT = ROOT / "evidence/rq_m12"
WALL = 180
RSS = 2 * 1024**3
PUBLIC = 50 * 1024**2
WORKERS = {"main": "analyze_probability_movement.py", "audit": "audit_probability_movement.py"}


def input_paths():
    old = "evidence/rq_m10"
    base = "evidence/rq_m11"
    paths = {
        "requirements.txt", "mathscape/__init__.py", "scripts/run_enumeration.py",
        "scripts/audit_enumerated_reference.py", "scripts/analyze_enumerated_reference.py",
        "mathscape/enumerated_reference.py", f"{old}/originals/receipt.json", f"{old}/manifest.json",
        *(f"{base}/{name}" for name in ("manifest.json", "input_receipt.json", "original_metrics.csv",
                                        "endpoint_metrics.csv", "reference_metrics.csv", "summary.csv",
                                        "hypotheses.json", "representatives.json", "independent_audit.json")),
    }
    for pop in ("gamma", "lognormal"):
        for seed in range(3):
            cell = f"{pop}_seed{seed}"
            paths.update(f"{old}/{folder}/{cell}.npz" for folder in ("originals", "panels"))
            paths.update(f"{old}/streams/{cell}_outer0_repeat0_stream{k}.npz" for k in (0,29,30,59,60,119))
            for outer in range(10):
                paths.update(f"{base}/{folder}/{cell}_outer{outer}.npz" for folder in ("references", "comparisons"))
    return sorted(paths)


def frozen_sources():
    sources = ["scripts/run_probability_movement.py", *(f"scripts/{x}" for x in WORKERS.values()),
               *(f"evidence/rq_m12/{x}" for x in ("SPEC.md", "CODEX_RECEIPT.md", "schema.md", "view_schema.md")),
               *(f"tests/test_probability_movement{x}.py" for x in ("", "_audit", "_guard", "_integration"))]
    if (ROOT / "mathscape/probability_movement.py").exists():
        sources.append("mathscape/probability_movement.py")
    return sorted(sources)


def npz_schema(paths):
    """Read NPY headers only. Retain all shape/dtype variants of each saved field."""
    from numpy.lib import format as npformat

    layouts = {}
    for name in paths:
        if not name.endswith(".npz"):
            continue
        family = str(Path(name).parent)
        fields = layouts.setdefault(family, {})
        expected_keys = set(fields)
        with zipfile.ZipFile(ROOT / name) as archive:
            keys = set()
            for member in archive.namelist():
                if not member.endswith(".npy"):
                    raise ValueError(f"Unexpected member in {name}")
                key = member[:-4]
                keys.add(key)
                with archive.open(member) as handle:
                    version = npformat.read_magic(handle)
                    if version == (1, 0):
                        shape, fortran, dtype = npformat.read_array_header_1_0(handle)
                    elif version == (2, 0):
                        shape, fortran, dtype = npformat.read_array_header_2_0(handle)
                    else:
                        raise ValueError(f"Unsupported NPY version in {name}")
                value = {"shape":list(shape),"dtype":dtype.str,"order":"F" if fortran else "C"}
                options = fields.setdefault(key, [])
                if value not in options:
                    options.append(value)
            if expected_keys and keys != expected_keys:
                raise ValueError(f"NPZ fields differ within {family}")
    return layouts


def manifest():
    paths = input_paths()
    files = check_committed_files(paths, ACCEPTED)
    origins = json.loads((ROOT / "evidence/rq_m11/input_receipt.json").read_bytes())["originals"]
    planned = [(p,s,o) for p in ("gamma","lognormal") for s in range(3) for o in range(10)]
    if [(r["distribution"],r["master_seed"],r["outer_id"]) for r in origins] != planned:
        raise ValueError("Unexpected accepted original IDs or order")
    return {
        "question_id":"RQ-M12", "accepted_head":ACCEPTED,
        "source_inputs":files, "originals":origins, "environment":environment(),
        "frozen_sources":{name:digest(ROOT/name) for name in frozen_sources()},
        "accepted_npz_schema":npz_schema(paths), "sample_size":5, "B":4999, "alpha":0.05,
        "comparison_axes":["r*120+k", "bca/percentile", "lower/upper"],
        "m10_panel_axes":["outer", "r", "t/percentile/bca", "k"],
        "originals_per_population":30, "paired_streams_per_original":2400,
        "total_distinct_saved_series":144000,"BCa_endpoint_observations":288000,
        "source_distribution":"3125 ordered binary64 bootstrap means per saved M10 original",
        "confirmatory_quantity":"u=max RMS((Q(q_B)-Q(q*))/sample_SE) across the two BCa endpoints",
        "decision":"All 30 complete; u <= 0.10 on both ends in >=27 originals per group. Missing/invalid -> hold. Both groups support or any non-support, else hold.",
        "validity_and_schema":"SPEC.md, schema.md", "saved_view":"view_schema.md",
        "limits":{"wall_seconds_per_stage":WALL,"peak_rss_bytes":RSS,"full_changed_files_bytes":PUBLIC,
                  "main":1,"independent_audit":1,"confirmed_code_bug_retries_per_affected_stage":1,
                  "new_originals":0,"new_streams":0,"PRNG_replays":0,"M11_reruns":0},
        "comparison_tolerances":{"rtol":2e-10,"atol":2e-12,"identities_rtol":2e-12,"identities_atol":2e-12,
                                 "exact":["ID","hash","support point","selected index","direction","classification","pass"]},
    }


def verify(commit):
    if len(commit) != 40 or git("rev-parse", "HEAD").decode().strip() != commit:
        raise ValueError("Expected the analysis freeze HEAD")
    subprocess.run(["git", "merge-base", "--is-ancestor", ACCEPTED, commit], cwd=ROOT, check=True)
    check_committed_files([*frozen_sources(), "evidence/rq_m12/manifest.json"], commit)
    stored = json.loads((OUT/"manifest.json").read_bytes())
    if manifest() != stored:
        raise ValueError("Accepted inputs, environment, schema or working sources differ from freeze")
    return stored


def worker(stage, commit):
    checked = verify(commit)
    if stage == "main":
        write_once(OUT/"input_receipt.json", dict(question_id="RQ-M12",accepted_head=ACCEPTED,
                   preanalysis_commit=commit, manifest_sha256=digest(OUT/"manifest.json"),
                   inputs=checked["source_inputs"], originals=checked["originals"]))
    else:
        receipt = json.loads((OUT/"input_receipt.json").read_bytes())
        main = json.loads((OUT/"main_resources.json").read_bytes())
        if (main["status"] != "completed" or main["preanalysis_commit"] != commit or
            receipt["preanalysis_commit"] != commit or receipt["inputs"] != checked["source_inputs"] or
            receipt["originals"] != checked["originals"] or receipt["manifest_sha256"] != digest(OUT/"manifest.json")):
            raise ValueError("Completed, matching main attempt required")
    script = ROOT / "scripts" / WORKERS[stage]
    module = "scripts." + script.stem
    os.execv(sys.executable, [sys.executable, "-m", module, "--root", str(ROOT), "--output", str(OUT)])


def static_public_bytes():
    changed = git("diff","--name-only","-z",ACCEPTED) + git("ls-files","--others","--exclude-standard","-z")
    paths = {ROOT/name.decode() for name in changed.split(b"\0") if name}
    return sum(p.stat().st_size for p in paths if p.is_file() and OUT not in p.parents)


def limit(elapsed, resident, size):
    if elapsed >= WALL - 1:
        return "wall"
    if resident >= RSS:
        return "rss"
    if size > PUBLIC:
        return "storage"
    return None


def supervise(stage, commit):
    if stage == "audit" and not (OUT/"main_resources.json").is_file():
        raise ValueError("Run the frozen main first")
    record = dict(phase=stage, preanalysis_commit=commit, wall_cap=WALL, rss_cap_bytes=RSS,
                  public_cap_bytes=PUBLIC, new_random_series=0)
    write_once(OUT/f"{stage}_attempt.json", record)
    other = static_public_bytes()
    child = subprocess.Popen([sys.executable, __file__, stage, "--worker", "--commit", commit], cwd=ROOT)
    stop = None
    try:
        while child.poll() is None:
            pstat = subprocess.run(["ps", "-o", "rss=", "-p", str(child.pid)],capture_output=True)
            resident = int(pstat.stdout.strip() or b"0") * 1024
            current = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
            stop = limit(monotonic()-START, resident, current+other)
            if stop:
                child.kill()
                break
            time.sleep(.05)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()
        elapsed = monotonic()-START
        peak = int(max(resource.getrusage(w).ru_maxrss for w in (resource.RUSAGE_CHILDREN, resource.RUSAGE_SELF)))
        if sys.platform != "darwin":
            peak *= 1024
        size = other + sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
        success = child.returncode == 0 and not stop and elapsed < WALL and peak < RSS and size <= PUBLIC
        write_once(OUT/f"{stage}_resources.json", dict(record,status="completed" if success else "stopped",
                   external_worker_exit=child.returncode, exit_code=child.returncode, guard_stop=stop,
                   wall_seconds=elapsed, peak_rss_bytes=peak, public_bytes_at_finish=size))
    return 0 if success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze","main","audit"))
    parser.add_argument("--commit")
    parser.add_argument("--worker", action="store_true",help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stage == "freeze":
        write_once(OUT/"manifest.json", manifest())
    elif not args.commit:
        parser.error("--commit is required for a scientific stage")
    elif args.worker:
        worker(args.stage, args.commit)
    else:
        raise SystemExit(supervise(args.stage, args.commit))
