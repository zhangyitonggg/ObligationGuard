import unittest

from obligationguard.ablations import model_settings
from obligationguard.config import load_config
from obligationguard.preliminary import safety_failures


class PreliminaryTests(unittest.TestCase):
    def test_labels_are_independent_and_denominator_is_functionally_correct_runs(self):
        def row(functional, security, forbidden=False, unfulfilled=False):
            return {"functional_pass": functional, "security_pass": security, "attribution": {"forbidden_action": {"present": forbidden}, "unfulfilled_obligation": {"present": unfulfilled}}}
        result = safety_failures([row(True, False, True, True), row(True, False, False, True), row(True, True), row(False, False)])
        self.assertEqual(result["evaluated_executions"], 3)
        self.assertAlmostEqual(result["unsafe_rate"], 2/3)
        self.assertAlmostEqual(result["forbidden_action"], 1/3)
        self.assertAlmostEqual(result["unfulfilled_obligation"], 2/3)


class AblationTests(unittest.TestCase):
    def test_training_runs_cover_all_rq3_comparisons(self):
        paper = load_config("configs/paper.toml")
        runs = model_settings(paper)
        self.assertEqual(len(runs), 8)
        self.assertEqual(len({row["name"] for row in runs}), 8)
        for row in runs:
            self.assertEqual(row["settings"]["training"]["epochs"], 2)
            self.assertEqual(row["settings"]["training"]["learning_rate"], 1e-5)
