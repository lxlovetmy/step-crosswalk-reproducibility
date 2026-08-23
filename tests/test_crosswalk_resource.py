import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from crosswalk_converter import (  # noqa: E402
    STATUS_ABOVE_RANGE,
    STATUS_BELOW_RANGE,
    STATUS_IN_RANGE,
    CrosswalkResource,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class CrosswalkResourceTests(unittest.TestCase):
    def make_resource(self, directory: Path) -> CrosswalkResource:
        metadata = directory / "metadata.csv"
        knots = directory / "knots.csv"
        write_rows(
            metadata,
            [
                {
                    "resource_version": "test",
                    "source_algorithm": "a",
                    "target_algorithm": "b",
                    "released_source_p05_steps": 1.0,
                    "released_source_p95_steps": 9.0,
                    "exact_knot_nodes": 3,
                }
            ],
        )
        write_rows(
            knots,
            [
                {
                    "source_algorithm": "a",
                    "target_algorithm": "b",
                    "knot_index": 1,
                    "source_input_steps": 0.0,
                    "predicted_target_steps": 0.0,
                },
                {
                    "source_algorithm": "a",
                    "target_algorithm": "b",
                    "knot_index": 2,
                    "source_input_steps": 5.0,
                    "predicted_target_steps": 10.0,
                },
                {
                    "source_algorithm": "a",
                    "target_algorithm": "b",
                    "knot_index": 3,
                    "source_input_steps": 10.0,
                    "predicted_target_steps": 20.0,
                },
            ],
        )
        return CrosswalkResource(metadata, knots)

    def test_strict_direction_and_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource = self.make_resource(Path(tmp))
            inside = resource.convert_one("a", "b", 2.5)
            self.assertEqual(inside.status, STATUS_IN_RANGE)
            self.assertAlmostEqual(inside.converted_target_steps, 5.0)
            self.assertEqual(resource.convert_one("a", "b", 0.5).status, STATUS_BELOW_RANGE)
            self.assertEqual(resource.convert_one("a", "b", 9.5).status, STATUS_ABOVE_RANGE)
            with self.assertRaises(KeyError):
                resource.convert_one("b", "a", 2.5)

    def test_rounding_occurs_after_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource = self.make_resource(Path(tmp))
            result = resource.convert_one("a", "b", 2.75, round_whole_step=True)
            self.assertEqual(result.status, STATUS_IN_RANGE)
            self.assertEqual(result.converted_target_steps, 6.0)

    def test_bundled_release_uses_exact_resource_only(self):
        tables = ROOT / "results" / "reference" / "tables"
        resource = CrosswalkResource(
            tables / "crosswalk_direction_metadata.csv",
            tables / "crosswalk_exact_knots.csv",
        )
        self.assertEqual(len(resource.available_directions()), 42)
        self.assertTrue(
            all(row["resource_version"] == "1.1.1" for row in resource.metadata.values())
        )
        self.assertFalse((tables / "individual_crosswalk_grid.csv").exists())
        self.assertFalse((tables / "crosswalk_grid_fidelity.csv").exists())

    def test_bundled_release_boundary_contract_all_directions(self):
        tables = ROOT / "results" / "reference" / "tables"
        resource = CrosswalkResource(
            tables / "crosswalk_direction_metadata.csv",
            tables / "crosswalk_exact_knots.csv",
        )
        checks = 0
        for source, target in resource.available_directions():
            metadata = resource.metadata[(source, target)]
            low = float(metadata["released_source_p05_steps"])
            high = float(metadata["released_source_p95_steps"])
            probes = (
                (math.nextafter(low, -math.inf), STATUS_BELOW_RANGE),
                (low, STATUS_IN_RANGE),
                ((low + high) / 2.0, STATUS_IN_RANGE),
                (high, STATUS_IN_RANGE),
                (math.nextafter(high, math.inf), STATUS_ABOVE_RANGE),
            )
            for value, expected_status in probes:
                result = resource.convert_one(source, target, value)
                self.assertEqual(result.status, expected_status)
                if expected_status == STATUS_IN_RANGE:
                    self.assertIsNotNone(result.converted_target_steps)
                    self.assertTrue(math.isfinite(result.converted_target_steps))
                else:
                    self.assertIsNone(result.converted_target_steps)
                checks += 1
        self.assertEqual(checks, 210)


if __name__ == "__main__":
    unittest.main()
