#!/usr/bin/env python3
"""Shared release constants and small integrity helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable


ALGORITHMS = ["acti", "adept", "oak", "scrf", "scssl", "vs", "vsrev"]
ALGORITHM_LABELS = {
    "acti": "ActiLife",
    "adept": "ADEPT",
    "oak": "Oak",
    "scrf": "Stepcount-RF",
    "scssl": "Stepcount-SSL",
    "vs": "Verisense",
    "vsrev": "Verisense-revised",
}
DIRECTED_PAIRS = [(source, target) for source in ALGORITHMS for target in ALGORITHMS if source != target]
OAK_ANCHORS = [7000.0, 8000.0, 10000.0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    columns = fieldnames or list(materialized[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_input_hashes(data_root: Path, manifest_path: Path) -> list[dict[str, str]]:
    rows = read_csv(manifest_path)
    observed: list[dict[str, str]] = []
    for row in rows:
        relative = Path(row["relative_path"])
        parts = relative.parts
        if len(parts) < 3 or parts[:2] != ("data", "raw"):
            raise ValueError(f"Invalid input path in manifest: {relative}")
        path = data_root.joinpath(*parts[2:])
        actual = sha256(path)
        if actual != row["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {path.name}: {actual}")
        observed.append({"relative_path": str(relative), "sha256": actual, "status": "PASS"})
    return observed


def assert_pair_set(rows: list[dict[str, object]], label: str) -> None:
    observed = {(str(row["source_algorithm"]), str(row["target_algorithm"])) for row in rows}
    expected = set(DIRECTED_PAIRS)
    if observed != expected:
        raise RuntimeError(
            f"{label} directed-pair mismatch: missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
