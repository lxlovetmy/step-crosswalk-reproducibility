#!/usr/bin/env python3
"""Create or verify the package-internal SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_PARTS = {"__pycache__", ".git", ".pytest_cache", ".mypy_cache"}
EXCLUDED_NAMES = {"MANIFEST.sha256", ".DS_Store"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path.name not in EXCLUDED_NAMES
        and not (set(path.relative_to(root).parts) & EXCLUDED_PARTS)
    )


def render(root: Path) -> str:
    return "".join(f"{sha256(path)}  {path.relative_to(root).as_posix()}\n" for path in inventory(root))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", type=Path)
    group.add_argument("--check", type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    expected = render(root)
    if args.write:
        target = args.write if args.write.is_absolute() else root / args.write
        target.write_text(expected, encoding="utf-8")
        print(f"MANIFEST_WRITE=PASS files={len(inventory(root))}")
        return 0
    target = args.check if args.check.is_absolute() else root / args.check
    actual = target.read_text(encoding="utf-8")
    if actual != expected:
        print("MANIFEST_CHECK=FAIL")
        return 2
    print(f"MANIFEST_CHECK=PASS files={len(inventory(root))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
