import ast
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "analysis"
REF = ROOT / "results" / "reference" / "tables"


class CommonSupportReleaseTests(unittest.TestCase):
    def test_formal_source_has_one_common_support_grid(self):
        support = (ANALYSIS / "build_stage6_common_support.py").read_text(encoding="utf-8")
        wear = (ANALYSIS / "build_wear_threshold_sensitivity.py").read_text(encoding="utf-8")
        validator = (ANALYSIS / "validate_release.py").read_text(encoding="utf-8")
        config = (ROOT / "config" / "analysis.json").read_text(encoding="utf-8")
        retired = (
            "COMMON_EMPIRICAL_DEFINITION",
            "common_support_empirical",
            "common_support_pooled_empirical_quantile",
            "pooled_empirical_quantile",
            "crossfit_ensemble_predictions",
            "cross-fitted isotonic ensemble",
        )
        for token in retired:
            self.assertNotIn(token, support)
            self.assertNotIn(token, wear)
            self.assertNotIn(token, config)
        self.assertNotIn('common[["source_algorithm", "target_algorithm", "common_support_empirical_H"]]', validator)
        self.assertNotIn('displayed["common_support_empirical_H"]', validator)
        self.assertIn('COMMON_LINEAR_DEFINITION = "common_support_linear"', support)
        self.assertIn("np.linspace(common_low, common_high, GRID_N)", support)
        self.assertIn("same fitted subgroup curve as original-grid H", support)
        self.assertIn('if __name__ == "__main__":\n    main_release()', support)
        self.assertNotIn("pooled_common_n >= GRID_N", support)
        self.assertNotIn("COMMON_N_LT_", wear)

    def test_common_support_reuses_one_fit_per_eligible_subgroup(self):
        sys.path.insert(0, str(ANALYSIS))
        support = importlib.import_module("build_stage6_common_support")
        x = np.arange(12, dtype=float)
        cohort = SimpleNamespace(
            daily_by_alg={"source": x, "target": 2.0 * x + 1.0},
            subject_ids=np.asarray([f"s{i}" for i in range(12)], dtype=object),
            age_group=np.asarray(["20-39", "40-59", "60+"] * 4, dtype=object),
            sex=np.asarray(["male", "female"] * 6, dtype=object),
            bmi_group=np.asarray(["<25", "25-<30", ">=30"] * 4, dtype=object),
        )
        models = []

        class Model:
            def __init__(self, offset):
                self.offset = offset
                self.grids = []

            def predict(self, grid):
                values = np.asarray(grid, dtype=float)
                self.grids.append(values.copy())
                return values + self.offset

        def fake_fit(_method, fit_x, _fit_y):
            model = Model(float(np.mean(fit_x)))
            models.append(model)
            return model

        with (
            mock.patch.object(support, "MIN_SUBGROUP_CURVE_N", 2),
            mock.patch.object(support, "MIN_COMMON_SUPPORT_LEVEL_N", 2),
            mock.patch.object(support, "grouped_oof_mae_per_iqr", return_value=0.1),
            mock.patch.object(support, "fit_crosswalk", side_effect=fake_fit),
        ):
            result = support.compute_pair(
                cohort,
                np.arange(12),
                "source",
                "target",
                123,
                keep_detail=True,
            )

        self.assertEqual(len(models), 8)  # 3 age + 2 sex + 3 BMI curves
        self.assertTrue(all(len(model.grids) == 2 for model in models))
        self.assertTrue(all(len(grid) == 101 for model in models for grid in model.grids))
        common_rows = [
            row for row in result.detail_rows
            if row["grid_definition"] == support.COMMON_LINEAR_DEFINITION
        ]
        self.assertEqual(len(common_rows), 3)
        self.assertTrue(all(np.isfinite(float(row["p95_between_subgroup_spread_predicted_target"])) for row in common_rows))
        self.assertTrue(all(int(row["pooled_observations_within_common_support"]) < 101 for row in common_rows))
        self.assertTrue(all(float(row["grid_min"]) == float(row["common_support_low"]) for row in common_rows))
        self.assertTrue(all(float(row["grid_max"]) == float(row["common_support_high"]) for row in common_rows))

    def test_subject_copies_never_cross_stage6_folds(self):
        sys.path.insert(0, str(ANALYSIS))
        support = importlib.import_module("build_stage6_common_support")
        subject_ids = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"], dtype=object)
        for train, test in support.grouped_folds(subject_ids, 20260715):
            train_ids = set(subject_ids[train])
            test_ids = set(subject_ids[test])
            self.assertFalse(train_ids & test_ids)

    def test_bootstrap_retains_complete_psu_clusters_with_multiplicity(self):
        sys.path.insert(0, str(ANALYSIS))
        bootstrap = importlib.import_module("build_stage4_crosswalk_uncertainty_ci")
        strata = np.asarray([1, 1, 1, 1, 1, 2, 2, 2], dtype=float)
        psu = np.asarray([1, 1, 2, 2, 2, 1, 2, 2], dtype=float)
        sampled = bootstrap.design_bootstrap_indices(
            strata,
            psu,
            np.random.default_rng(20260715),
        )
        for stratum in np.unique(strata):
            for cluster in np.unique(psu[strata == stratum]):
                members = np.flatnonzero((strata == stratum) & (psu == cluster))
                multiplicities = [int(np.sum(sampled == member)) for member in members]
                self.assertEqual(len(set(multiplicities)), 1)

    def test_grouped_folds_only_serve_oof_error(self):
        tree = ast.parse((ANALYSIS / "build_stage6_common_support.py").read_text(encoding="utf-8"))
        callers = []
        for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
            if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "grouped_folds" for node in ast.walk(function)):
                callers.append(function.name)
        self.assertEqual(callers, ["grouped_oof_mae_per_iqr"])

    def test_release_subgroup_minimum_is_200(self):
        sys.path.insert(0, str(ANALYSIS))
        source = importlib.import_module("build_mvp_crosswalk_stability")
        support = importlib.import_module("build_stage6_common_support")
        self.assertEqual(source.MIN_SUBGROUP_CURVE_N, 200)
        self.assertEqual(support.MIN_SUBGROUP_CURVE_N, 200)
        self.assertEqual(support.MIN_COMMON_SUPPORT_LEVEL_N, 200)

    def test_reference_output_schema_is_single_grid(self):
        detail = pd.read_csv(REF / "common_support_detail.csv")
        summary = pd.read_csv(REF / "common_support_summary.csv")
        bootstrap = pd.read_csv(REF / "continuous_bootstrap.csv")
        wear_pair = pd.read_csv(REF / "wear_pair.csv")
        wear_comparison = pd.read_csv(REF / "wear_comparison.csv")
        self.assertEqual(len(detail), 252)
        self.assertEqual(detail.groupby("grid_definition").size().to_dict(), {"common_support_linear": 126, "original_grid_stage3": 126})
        self.assertTrue(detail.loc[detail.grid_definition.eq("common_support_linear"), "grid_n"].eq(101).all())
        self.assertEqual(len(summary), 42)
        self.assertIn("common_support_linear_H", summary)
        self.assertNotIn("common_support_empirical_H", summary)
        self.assertEqual(len(bootstrap), 84)
        self.assertEqual(bootstrap.groupby("support_definition").size().to_dict(), {"common_support_linear": 42, "original_grid_stage3": 42})
        self.assertTrue(bootstrap[["E_bootstrap_success", "H_bootstrap_success", "bootstrap_requested"]].eq(300).all().all())
        self.assertFalse(any("empirical" in column for column in wear_pair.columns))
        self.assertFalse(wear_comparison.astype(str).apply(lambda column: column.str.contains("common-support empirical H", regex=False).any()).any())


if __name__ == "__main__":
    unittest.main()
