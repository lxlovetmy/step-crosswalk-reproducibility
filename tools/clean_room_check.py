#!/usr/bin/env python3
"""Run the full pipeline in a new temporary output directory and save evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    root = args.package_root.resolve()
    started = datetime.now(timezone.utc); clock = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="step_crosswalk_cleanroom.", dir="/private/tmp") as temporary:
        output = Path(temporary) / "full"
        completed = subprocess.run([
            sys.executable, str(root / "run_all.py"), "--mode", "full",
            "--data-root", str(args.data_root.resolve()), "--output-dir", str(output),
        ])
        validation = {}
        validation_path = output / "release_validation.json"
        if validation_path.is_file():
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
        size = sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) if output.exists() else 0
        payload = {
            "status": "PASS" if completed.returncode == 0 and validation.get("status") == "RELEASE_CANDIDATE_PASS" else "FAIL",
            "started_utc": started.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": round(time.monotonic() - clock, 3), "returncode": completed.returncode,
            "output_bytes": size, "validation_status": validation.get("status"),
            "validation_check_count": validation.get("check_count"), "validation_failure_count": validation.get("failure_count"),
            "temporary_output_removed_after_report": True,
        }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"CLEAN_ROOM_CHECK={payload['status']}")
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
