import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from obligationguard.backends import Completion, ResponsesBackend, completion_result
from obligationguard.errors import ModelOutputError
from obligationguard.evaluation import evaluate, predict
from obligationguard.io import read_jsonl, write_jsonl
from obligationguard.schema import Instance, Obligation


class OutputTests(unittest.TestCase):
    def scratch(self):
        root = Path(os.environ.get("OG_TEST_SCRATCH", "D:/Codex/scratch/obligationguard-tests" if os.name == "nt" else "/tmp/obligationguard-tests"))
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def sample(self):
        return Instance("fixture", "Fixture task", ({"step": 1},), (Obligation("Required action"),))

    def test_unfinished_reasoning_is_retained_scored_empty_and_not_regenerated(self):
        class Backend:
            calls = 0
            def identity(self): return {"model": "fixture-model"}
            def complete(self, prompt):
                self.calls += 1
                return completion_result("<think>unfinished", {"original": "<think>unfinished"}, {})
        with self.scratch() as location:
            root, backend = Path(location), Backend()
            predictions = root / "predictions.jsonl"
            predict([self.sample()], backend, predictions)
            predict([self.sample()], backend, predictions)
            row = next(read_jsonl(predictions))
            self.assertEqual(row["text"], "<think>unfinished")
            self.assertEqual(row["raw_response"]["original"], row["text"])
            self.assertTrue(row["parse_error"])
            self.assertEqual(backend.calls, 1)
            result = evaluate([self.sample()], predictions, backend, root / "judgments.jsonl", root / "metrics.json")
            self.assertEqual(result["parse_failures"], 1)
            self.assertEqual(result["metrics"]["recall"], 0.0)
            self.assertEqual(backend.calls, 1)

    def test_responses_incomplete_status_is_retained_without_another_generation(self):
        raw = {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": []}
        with patch.dict(os.environ, {"FIXTURE_API_KEY": "fixture"}), patch("obligationguard.backends._post", side_effect=[{"input_tokens": 10}, raw]) as request:
            backend = ResponsesBackend({"model": "fixture-model", "api_key_env": "FIXTURE_API_KEY", "max_output_tokens": 20})
            completion = backend.complete("Fixture prompt")
            self.assertTrue(completion.error)
            self.assertEqual(completion.raw, raw)
            self.assertEqual(request.call_count, 2)

    def test_incomplete_judge_cannot_produce_a_match_and_its_output_is_saved(self):
        class Judge:
            def identity(self): return {"model": "fixture-judge"}
            def complete(self, prompt): return Completion('{"match":true}', {"status": "incomplete"}, {}, "incomplete judge response")
        with self.scratch() as location:
            root = Path(location)
            predictions = root / "predictions.jsonl"
            write_jsonl(predictions, [{"instance_id": "fixture", "text": json.dumps(self.sample().target())}])
            with self.assertRaises(ModelOutputError):
                evaluate([self.sample()], predictions, Judge(), root / "judgments.jsonl", root / "metrics.json")
            self.assertFalse((root / "metrics.json").exists())
            saved = next(read_jsonl(root / "judgments_errors.jsonl"))
            self.assertEqual(saved["raw_response"], {"status": "incomplete"})
