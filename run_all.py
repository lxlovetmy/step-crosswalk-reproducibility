#!/usr/bin/env python3
"""One-command reproduction of the aggregate tables and figures."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ANALYSIS = PACKAGE_ROOT / "analysis"


def run(command: list[str], label: str, env: dict[str, str]) -> None:
    print(f"\n[{datetime.now().isoformat(timespec='seconds')}] START {label}", flush=True)
    subprocess.run(command, cwd=ANALYSIS, env=env, check=True)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] PASS  {label}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_root, output = args.data_root.resolve(), args.output_dir.resolve()
    if output == PACKAGE_ROOT or PACKAGE_ROOT in output.parents:
        parser.error("--output-dir must be outside the package")
    if output.exists() and any(output.iterdir()):
        parser.error("--output-dir must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    (output / "logs").mkdir(exist_ok=True)

    sys.path.insert(0, str(ANALYSIS))
    from release_common import verify_input_hashes, write_json

    started = datetime.now(timezone.utc)
    input_checks = verify_input_hashes(data_root, PACKAGE_ROOT / "data" / "expected_inputs.csv")
    write_json(output / "logs" / "input_hashes.json", input_checks)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MPLCONFIGDIR"] = str(output / ".mplconfig")
    py = sys.executable
    reps = "2" if args.mode == "smoke" else "300"

    commands = [
        ([py, str(ANALYSIS / "build_stage2_all7_crosswalk_matrix.py"), "--raw-dir", str(data_root), "--out-dir", str(output)], "core crosswalk matrix"),
        ([py, str(ANALYSIS / "build_stage2c_all7_subgroup_stability.py"), "--raw-dir", str(data_root), "--out-dir", str(output)], "subgroup curves"),
        ([py, str(ANALYSIS / "build_sample_alignment.py"), "--raw-dir", str(data_root), "--out-dir", str(output)], "sample alignment"),
        ([py, str(ANALYSIS / "build_stage3_release_assembly.py"), "--out-dir", str(output)], "direction-level E/H assembly"),
        ([py, str(ANALYSIS / "build_crosswalk_resource.py"), "--raw-dir", str(data_root), "--out-dir", str(output), "--resource-version", "1.1.1"], "exact crosswalk mappings and direction metadata"),
        ([py, str(ANALYSIS / "build_stage4_crosswalk_uncertainty_ci.py"), "--raw-dir", str(data_root), "--out-dir", str(output), "--bootstrap-reps", reps, "--progress-every", "10"], "E/H bootstrap uncertainty"),
        ([py, str(ANALYSIS / "build_stage6_common_support.py"), "--raw-dir", str(data_root), "--out-dir", str(output), "--bootstrap-reps", reps, "--progress-every", "10"], "common-support sensitivity"),
        ([py, str(ANALYSIS / "build_stage6_cycle_threshold.py"), "--raw-dir", str(data_root), "--out-dir", str(output), "--bootstrap-reps", reps, "--progress-every", "10"], "cross-cycle and fixed-threshold checks"),
    ]
    try:
        for command, label in commands:
            run(command, label, env)
        if args.mode == "full":
            run([
                py, str(ANALYSIS / "build_wear_threshold_sensitivity.py"),
                "--raw-dir", str(data_root), "--baseline-dir", str(output),
                "--output-dir", str(output / "wear"),
            ], "wear-time sensitivity", env)
            run([
                py, str(ANALYSIS / "build_tables_figures.py"),
                "--raw-dir", str(data_root), "--generated-dir", str(output),
            ], "submission tables and figures", env)
            run([
                py, str(ANALYSIS / "validate_release.py"),
                "--package-root", str(PACKAGE_ROOT), "--generated-dir", str(output),
                "--report", str(output / "release_validation.json"),
            ], "release validation", env)
        summary = {
            "status": "PASS", "mode": args.mode, "started_utc": started.isoformat(),
            "finished_utc": datetime.now(timezone.utc).isoformat(), "bootstrap_replicates": int(reps),
            "input_hashes_verified": len(input_checks), "output_dir": str(output),
        }
        write_json(output / "run_summary.json", summary)
        print("REPRODUCTION_RUN=PASS", flush=True)
        return 0
    except subprocess.CalledProcessError as error:
        write_json(output / "run_summary.json", {
            "status": "BLOCKED", "mode": args.mode, "failed_command": error.cmd,
            "returncode": error.returncode, "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        print(f"REPRODUCTION_RUN=BLOCKED ({error.returncode})", file=sys.stderr, flush=True)
        return error.returncode or 2


if __name__ == "__main__":
    raise SystemExit(main())
