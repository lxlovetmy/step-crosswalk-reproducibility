import unittest
import sys
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))

from build_release_domain_audit import error_summary, subject_grouped_folds, target_iqr


class ReleaseDomainAuditTests(unittest.TestCase):
    def test_subjects_never_cross_fold_boundaries(self):
        subject_ids = np.asarray(["a", "a", "b", "c", "d", "e", "f", "g", "h", "i"])
        seen = np.zeros(len(subject_ids), dtype=int)
        for train, test in subject_grouped_folds(subject_ids, 20260607):
            self.assertFalse(np.any(train & test))
            for subject in set(subject_ids.tolist()):
                locations = test[subject_ids == subject]
                self.assertTrue(np.all(locations) or not np.any(locations))
            seen += test.astype(int)
        np.testing.assert_array_equal(seen, np.ones(len(subject_ids), dtype=int))

    def test_complete_sample_iqr_is_retained_for_restricted_error(self):
        target = np.asarray([0.0, 10.0, 20.0, 30.0, 40.0])
        prediction = np.asarray([0.0, 12.0, 18.0, 33.0, 40.0])
        denominator = target_iqr(target)
        result = error_summary(prediction[1:4], target[1:4], denominator)
        self.assertAlmostEqual(denominator, 20.0)
        self.assertAlmostEqual(result["mae"], 7.0 / 3.0)
        self.assertAlmostEqual(result["E"], (7.0 / 3.0) / 20.0)


if __name__ == "__main__":
    unittest.main()
