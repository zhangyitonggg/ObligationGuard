import itertools
import unittest

from obligationguard.errors import DataError
from obligationguard.matching import maximum_matching
from obligationguard.metrics import InstanceScore, aggregate


class MatchingTests(unittest.TestCase):
    def test_augmenting_path_recovers_optimum(self):
        pairs = maximum_matching([[True, True], [True, False]], 2)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(set(pairs), {(0, 1), (1, 0)})

    def test_duplicate_predictions_cannot_double_count(self):
        self.assertEqual(len(maximum_matching([[True], [True]], 1)), 1)

    def test_matches_exhaustive_assignment_on_small_graphs(self):
        for bits in itertools.product((False, True), repeat=9):
            matrix = [list(bits[i:i+3]) for i in range(0, 9, 3)]
            optimum = 0
            for assignment in itertools.product((-1, 0, 1, 2), repeat=3):
                chosen = [x for x in assignment if x >= 0]
                if len(chosen) != len(set(chosen)):
                    continue
                if all(j < 0 or matrix[i][j] for i, j in enumerate(assignment)):
                    optimum = max(optimum, len(chosen))
            self.assertEqual(len(maximum_matching(matrix, 3)), optimum)

    def test_empty_and_invalid_matrices(self):
        self.assertEqual(maximum_matching([], 3), [])
        self.assertEqual(maximum_matching([[], []], 0), [])
        with self.assertRaises(DataError):
            maximum_matching([[1]], 1)
        with self.assertRaises(DataError):
            maximum_matching([[True]], 2)


class MetricTests(unittest.TestCase):
    def test_micro_metrics_and_separate_classification(self):
        result = aggregate([
            InstanceScore("positive-a", 1, 1, 1),
            InstanceScore("positive-b", 3, 2, 1),
            InstanceScore("negative-a", 2, 0, 0),
            InstanceScore("negative-b", 0, 0, 0),
        ])
        self.assertEqual(result["precision"], 0.5)
        self.assertAlmostEqual(result["recall"], 2/3)
        self.assertEqual(result["exact_match"], 0.5)
        self.assertEqual(result["ca_positive"], 1.0)
        self.assertEqual(result["ca_negative"], 0.5)

    def test_empty_predictions_on_positive_instances(self):
        result = aggregate([InstanceScore("positive", 0, 2, 0)])
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["exact_match"], 0.0)
        self.assertEqual(result["ca_positive"], 0.0)
        self.assertIsNone(result["ca_negative"])

    def test_wrong_extra_prediction_prevents_exact_match(self):
        self.assertFalse(InstanceScore("a", 2, 1, 1).exact_match)

    def test_invalid_and_duplicate_scores(self):
        with self.assertRaises(DataError):
            InstanceScore("a", 1, 1, 2)
        with self.assertRaises(DataError):
            aggregate([InstanceScore("a", 0, 0, 0)] * 2)
