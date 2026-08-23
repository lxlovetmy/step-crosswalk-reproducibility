import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PrivacyTests(unittest.TestCase):
    def test_no_raw_word_or_symlink(self):
        forbidden = {".docx", ".xpt", ".sav", ".dta", ".sas7bdat"}
        for path in ROOT.rglob("*"):
            self.assertFalse(path.is_symlink(), path)
            if path.is_file():
                self.assertNotIn(path.suffix.lower(), forbidden, path)
                self.assertFalse(path.name.endswith(".csv.xz"), path)

    def test_no_private_absolute_path(self):
        needle = "/Users/" + "luoxiang"
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".py",".md",".json",".csv",".yml",".toml",".cff",".txt"} and path.stat().st_size < 2_000_000:
                self.assertNotIn(needle, path.read_text(encoding="utf-8", errors="ignore"), path)

    def test_retired_cluster_analysis_absent(self):
        self.assertFalse((ROOT / "analysis" / "build_pam_auxiliary.py").exists())
        for path in ROOT.rglob("*"):
            if path.is_file():
                self.assertNotIn("pam", path.name.lower(), path)


if __name__ == "__main__": unittest.main()
