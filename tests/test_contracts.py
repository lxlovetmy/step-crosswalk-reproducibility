import csv
import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results" / "reference" / "tables"


class ContractTests(unittest.TestCase):
    def test_run_all_formal_stage_order(self):
        runner = (ROOT / "run_all.py").read_text(encoding="utf-8")
        stages = [
            "build_stage2_all7_crosswalk_matrix.py",
            "build_stage2c_all7_subgroup_stability.py",
            "build_sample_alignment.py",
            "build_stage3_release_assembly.py",
            "build_crosswalk_resource.py",
            "build_stage4_crosswalk_uncertainty_ci.py",
            "build_stage6_common_support.py",
            "build_stage6_cycle_threshold.py",
            "build_release_domain_audit.py",
            "build_wear_threshold_sensitivity.py",
            "build_tables_figures.py",
            "validate_release.py",
        ]
        positions = [runner.index(stage) for stage in stages]
        self.assertEqual(positions, sorted(positions))
        self.assertIn('if args.mode == "full":', runner)
        self.assertIn('"--seed", stage4_seed', runner)
        self.assertGreaterEqual(runner.count('"--seed", stage6_seed'), 2)

    def test_table_figure_builder_emits_formal_figure1_name(self):
        source = (ROOT / "analysis" / "build_tables_figures.py").read_text(encoding="utf-8")
        self.assertIn('shutil.copyfile(figure1_path,figures/"Figure1.png")', source)

    def test_reference_manifest_covers_reference_tree(self):
        with (ROOT / "provenance" / "reference_manifest.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        recorded = {row["relative_path"]: row for row in rows}
        actual = sorted(
            path for path in (ROOT / "results" / "reference").rglob("*")
            if path.is_file()
        )
        self.assertEqual(set(recorded), {path.relative_to(ROOT).as_posix() for path in actual})
        for path in actual:
            relative = path.relative_to(ROOT).as_posix()
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), recorded[relative]["sha256"])
            self.assertEqual(path.stat().st_size, int(recorded[relative]["bytes"]))

    def test_configuration_contracts(self):
        config = json.loads((ROOT / "config" / "analysis.json").read_text())
        self.assertEqual(config["schema_version"], 4)
        self.assertEqual(config["cohort"]["expected_participants"], 8646)
        self.assertEqual(config["cohort"]["expected_valid_person_days"], 57080)
        self.assertEqual(config["cohort"]["minimum_age_years"], 20)
        self.assertEqual(config["individual_crosswalk"]["expected_directed_pairs"], 42)
        self.assertEqual(config["individual_crosswalk"]["canonical_machine_readable_representation"], "exact_isotonic_thresholds")
        self.assertEqual(config["individual_crosswalk"]["oof_folds"], 5)
        self.assertEqual(config["individual_crosswalk"]["oof_split_unit"], "participant")
        self.assertEqual(config["individual_crosswalk"]["oof_seed"], 20260607)
        self.assertEqual(config["evaluation"]["minimum_subgroup_n"], 200)
        self.assertEqual(
            config["evaluation"]["common_support_grid"],
            {"type": "equally_spaced", "nodes": 101},
        )
        self.assertNotIn("common_support_grids", config["evaluation"])
        self.assertEqual(
            config["evaluation"]["common_support_curve_fitting"],
            "one_full_resample_fit_per_eligible_subgroup_reused_for_complete_and_common_support_H",
        )
        self.assertEqual(config["bootstrap"]["replicates"], 300)
        self.assertEqual(config["bootstrap"]["stage4_seed"], 20260609)
        self.assertEqual(config["bootstrap"]["stage6_seed"], 20260715)
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
            "pair_summary.csv": 42, "common_support_detail.csv": 252,
            "common_support_summary.csv": 42, "continuous_bootstrap.csv": 84,
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
