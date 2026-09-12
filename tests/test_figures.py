import csv
import hashlib
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


class FigureTests(unittest.TestCase):
    def test_eight_current_visual_references(self):
        paths = [ROOT/"results"/"reference"/"figures"/f"Figure{i}_current.png" for i in range(1,5)]
        paths += [ROOT/"results"/"reference"/"figures"/f"FigureS{i}_current.png" for i in range(1,5)]
        self.assertEqual(len(paths),8)
        for path in paths:
            with self.subTest(path=path.name):
                with Image.open(path) as image:
                    self.assertGreater(image.width,1000); self.assertGreater(image.height,600)
        self.assertFalse((ROOT/"results"/"reference"/"figures"/"FigureS5_current.png").exists())

    def test_figure4_authority(self):
        path=ROOT/"results"/"reference"/"figures"/"Figure4_current.png"
        manifest = ROOT / "provenance" / "reference_manifest.csv"
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        row = next(item for item in rows if item["relative_path"] == "results/reference/figures/Figure4_current.png")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), row["sha256"])
        self.assertEqual(path.stat().st_size, int(row["bytes"]))


if __name__ == "__main__": unittest.main()
