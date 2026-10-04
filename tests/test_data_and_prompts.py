import hashlib
from importlib.resources import files
import json
import unittest

from obligationguard.datasets import validate_training_split
from obligationguard.errors import ConfigurationError, DataError, ModelOutputError
from obligationguard.prepare import import_trajectory
from obligationguard.prompting import identification, render
from obligationguard.schema import Instance, Obligation, parse_prediction
from obligationguard.training import encode_target_only


def instance(identifier="a", scenario="s", obligations=()):
    return Instance(identifier, "Complete the task", ({"step": 1, "action": "Inspect", "observation": "Visible state"},), obligations, scenario_id=scenario)


class DataTests(unittest.TestCase):
    def test_import_preserves_task_trajectory_and_original_labels(self):
        original = {"trajectory_id": "t", "seed_id": "s", "input": {"task": "Original task", "trajectory": [{"step": 1, "action": "Original action", "observation": "Original observation"}]}, "target": {"unfulfilled_obligations": [{"required_action": "Required closure", "why_required": "Original explanation", "security_consequence_if_omitted": "Original consequence"}]}}
        imported = import_trajectory(original)
        self.assertEqual(imported.task, original["input"]["task"])
        self.assertEqual(list(imported.trajectory), original["input"]["trajectory"])
        self.assertEqual(imported.obligations[0].required_safety_action, "Required closure")
        self.assertEqual(imported.metadata["original_target"], original["target"])

    def test_scenario_split_can_be_checked_explicitly(self):
        with self.assertRaises(DataError):
            validate_training_split([instance("a", "s")], [instance("b", "s")], 1, 1, split_by_scenario=True)
        validate_training_split([instance("a", "s")], [instance("b", "s")], 1, 1)

    def test_source_scenarios_prevent_composition_leakage(self):
        composed = Instance("a", "Task", ({"step": 1},), (), "composed", ("source",))
        with self.assertRaises(DataError):
            validate_training_split([composed], [instance("b", "source")], 1, 1, split_by_scenario=True)

    def test_instance_identity_cannot_cross_split(self):
        with self.assertRaises(DataError):
            validate_training_split([instance("a")], [instance("a")], 1, 1)

    def test_prediction_count_and_identifier_are_validated(self):
        self.assertEqual(parse_prediction({"task_id": "a", "obligation_count": 0, "obligations": []}, "a"), ())
        with self.assertRaises(ModelOutputError):
            parse_prediction({"task_id": "b", "obligation_count": 0, "obligations": []}, "a")
        with self.assertRaises(ModelOutputError):
            parse_prediction({"task_id": "a", "obligation_count": 1, "obligations": []}, "a")


class PromptTests(unittest.TestCase):
    def test_identifier_and_label_changes_cannot_change_the_model_input(self):
        positive = instance("source-U-r01", obligations=(Obligation("Secret target label"),))
        negative = instance("source-negative")
        self.assertEqual(identification(positive), identification(negative))
        self.assertEqual(positive.guard_input()["task_id"], negative.guard_input()["task_id"])
        self.assertEqual(positive.target()["task_id"], negative.target()["task_id"])
        for identifier in (positive.instance_id, negative.instance_id):
            self.assertNotIn(identifier, identification(positive))
            self.assertNotIn(identifier, identification(negative))

    def test_prompt_content_matches_extracted_hashes(self):
        resources = files("obligationguard").joinpath("prompts")
        metadata = json.loads(resources.joinpath("sources.json").read_text())
        self.assertEqual(len(metadata), 11)
        for name, source in metadata.items():
            content = resources.joinpath(name + ".txt").read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), source["sha256"])

    def test_replacement_does_not_interpret_braces_in_input(self):
        prompt = render("obligation_feedback", PREDICTED_OBLIGATIONS="literal {USER_TASK}")
        self.assertIn("literal {USER_TASK}", prompt)

    def test_model_input_contains_no_annotation_metadata(self):
        sample = instance(obligations=(Obligation("Secret target label"),))
        self.assertNotIn("Secret target label", identification(sample))

    def test_missing_placeholders_fail(self):
        with self.assertRaises(ConfigurationError):
            render("semantic_matching", USER_TASK="Task")


class CharacterTokenizer:
    eos_token = "<END>"
    def apply_chat_template(self, messages, **kwargs):
        user = messages[0]["content"]
        if len(messages) == 1:
            return "USER:" + user + "ASSISTANT:"
        return "USER:" + user + "ASSISTANT:<think>\n\n</think>\n\n" + messages[1]["content"] + self.eos_token
    def encode(self, text, **kwargs):
        return [ord(x) for x in text]


class TargetMaskTests(unittest.TestCase):
    def test_supervised_tokens_never_contain_the_source_identifier(self):
        sample = instance("source-U-r01", obligations=(Obligation("Required action"),))
        result = encode_target_only(sample, CharacterTokenizer(), 10000, {"enable_thinking": True})
        sequence = "".join(chr(x) for x in result["input_ids"])
        self.assertNotIn(sample.instance_id, sequence)
        supervised = "".join(chr(x) for x in result["labels"] if x != -100)
        self.assertEqual(json.loads(supervised[:-5])["task_id"], "task")

    def test_only_ground_truth_json_and_end_token_receive_loss(self):
        sample = instance(obligations=(Obligation("Required action"),))
        result = encode_target_only(sample, CharacterTokenizer(), 10000, {"enable_thinking": True})
        supervised = "".join(chr(x) for x in result["labels"] if x != -100)
        self.assertTrue(supervised.endswith("<END>"))
        self.assertEqual(json.loads(supervised[:-5]), sample.target())
        self.assertNotIn("<think>", supervised)
        self.assertGreater(result["labels"].count(-100), 0)

    def test_overlength_sequences_are_not_silently_truncated(self):
        with self.assertRaises(DataError):
            encode_target_only(instance(), CharacterTokenizer(), 20, {"enable_thinking": True})
