import unittest
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd


REF = Path(__file__).resolve().parents[1] / "results" / "reference" / "tables"


class NumericReferenceTests(unittest.TestCase):
    def test_e_h_display_statistics(self):
        frame = pd.read_csv(REF / "pair_summary.csv")
        def median_display(column):
            value = np.median(np.round(frame[column].to_numpy(float), 3))
            return Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        self.assertEqual(median_display("E"), Decimal("0.243"))
        self.assertEqual(median_display("H"), Decimal("0.285"))
        self.assertEqual((round(frame.E.min(),3),round(frame.E.max(),3)),(.123,.402))
        self.assertEqual((round(frame.H.min(),3),round(frame.H.max(),3)),(.142,.546))

    def test_fixed_threshold_medians(self):
        frame = pd.read_csv(REF / "fixed_threshold.csv")
        frame = frame.loc[frame.analysis_weighting.eq("unweighted_primary") & ~frame.target_threshold_degenerate.astype(bool)]
        medians = frame.groupby("threshold_target_steps").discordant_reclassification_pct.median().round(2).to_dict()
        self.assertEqual(medians, {7000.0:8.30,8000.0:9.84,10000.0:11.23})

    def test_wear_cohorts(self):
        frame = pd.read_csv(REF / "wear_cohort.csv")
        observed = {int(r.wear_threshold):(int(r.n_subjects),int(r.n_valid_person_days)) for r in frame.itertuples()}
        self.assertEqual(observed,{600:(9023,60782),720:(8936,59845),960:(8646,57080),1296:(8066,49564)})

    def test_common_support_display_convention(self):
        pair = pd.read_csv(REF / "pair_summary.csv")
        common = pd.read_csv(REF / "common_support_summary.csv")
        frame = pair[["source_algorithm", "target_algorithm", "H"]].merge(
            common[["source_algorithm", "target_algorithm", "common_support_linear_H"]],
            on=["source_algorithm", "target_algorithm"],
        )
        exact = frame.common_support_linear_H.to_numpy(float) - frame.H.to_numpy(float)
        quantum = Decimal("0.001")
        displayed = [
            Decimal(str(float(common_value))).quantize(quantum, rounding=ROUND_HALF_UP)
            - Decimal(str(float(full_value))).quantize(quantum, rounding=ROUND_HALF_UP)
            for common_value, full_value in zip(frame.common_support_linear_H, frame.H)
        ]
        ordered = sorted(displayed)
        midpoint = len(ordered) // 2
        median = ((ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)).quantize(quantum, rounding=ROUND_HALF_UP)
        self.assertEqual(((exact < 0).sum(), sum(x < 0 for x in displayed), sum(x == 0 for x in displayed), sum(x > 0 for x in displayed)), (33, 32, 6, 4))
        self.assertEqual(median, Decimal("-0.024"))


if __name__ == "__main__": unittest.main()
