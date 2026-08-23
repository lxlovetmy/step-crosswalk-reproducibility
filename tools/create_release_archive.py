#!/usr/bin/env python3
"""Create a versioned ZIP after verifying the internal manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, output = args.package_root.resolve(), args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing archive: {output}")
    subprocess.run([sys.executable, str(root / "tools" / "build_manifest.py"), "--package-root", str(root), "--check", "MANIFEST.sha256"], check=True)
    files = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts and path.name != ".DS_Store")
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, Path(root.name) / path.relative_to(root))
    with zipfile.ZipFile(output) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failure: {bad}")
    print(f"ARCHIVE_CREATE=PASS files={len(files)} bytes={output.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
