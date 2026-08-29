import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results" / "reference" / "tables"


class ContractTests(unittest.TestCase):
    def test_configuration_contracts(self):
        config = json.loads((ROOT / "config" / "analysis.json").read_text())
        self.assertEqual(config["cohort"]["expected_participants"], 8646)
        self.assertEqual(config["cohort"]["expected_valid_person_days"], 57080)
        self.assertEqual(config["cohort"]["minimum_age_years"], 20)
        self.assertEqual(config["individual_crosswalk"]["expected_directed_pairs"], 42)
        self.assertEqual(config["individual_crosswalk"]["canonical_machine_readable_representation"], "exact_isotonic_thresholds")
        self.assertEqual(config["bootstrap"]["replicates"], 300)
        self.assertEqual(config["release_domain_audit"]["bootstrap_replicates"], 0)
        self.assertEqual(
            config["release_domain_audit"]["interpretation"],
            "sensitivity_not_validated_range_or_improvement_test",
        )

    def test_reference_row_contracts(self):
        expected = {
            "sample_alignment.csv": 21,
            "crosswalk_exact_knots.csv": 11416,
            "crosswalk_direction_metadata.csv": 42,
            "pair_summary.csv": 42, "common_support_detail.csv": 378,
            "common_support_summary.csv": 42, "continuous_bootstrap.csv": 126,
            "cross_cycle.csv": 84, "cross_cycle_ci.csv": 1176,
            "fixed_threshold.csv": 252, "fixed_threshold_ci.csv": 1764,
            "release_domain_whole_cohort.csv": 42,
            "release_domain_cross_cycle.csv": 84,
        }
        for name, rows in expected.items():
            with self.subTest(name=name): self.assertEqual(len(pd.read_csv(REF / name)), rows)

    def test_exact_resource_contract(self):
        knots = pd.read_csv(REF / "crosswalk_exact_knots.csv")
        metadata = pd.read_csv(REF / "crosswalk_direction_metadata.csv")
        counts = knots.groupby(["source_algorithm", "target_algorithm"]).size().sort_index()
        expected = metadata.set_index(["source_algorithm", "target_algorithm"])["exact_knot_nodes"].astype(int).sort_index()
        self.assertEqual(counts.to_dict(), expected.to_dict())
        self.assertTrue(metadata.resource_version.astype(str).eq("1.1.1").all())
        monotone = knots.sort_values(["source_algorithm", "target_algorithm", "knot_index"]).groupby(["source_algorithm", "target_algorithm"]).apply(
            lambda frame: frame.source_input_steps.is_monotonic_increasing
            and frame.source_input_steps.is_unique
            and frame.predicted_target_steps.is_monotonic_increasing,
            include_groups=False,
        )
        self.assertTrue(monotone.all())
        self.assertFalse((REF / "individual_crosswalk_grid.csv").exists())
        self.assertFalse((REF / "crosswalk_grid_fidelity.csv").exists())


if __name__ == "__main__": unittest.main()
