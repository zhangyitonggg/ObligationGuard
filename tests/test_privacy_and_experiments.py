import json
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from obligationguard.catalog import EVALUATED_MODELS
from obligationguard.config import load_config
from obligationguard.container_state import capture_environment
from obligationguard.datasets import validate_benchmark
from obligationguard.errors import ConfigurationError
from obligationguard.privacy import inspect_text, remove_private_record
from obligationguard.schema import Instance, Obligation


class PrivacyTests(unittest.TestCase):
    def test_private_addresses_and_directory_identifiers_are_deleted(self):
        address = "private-user" + "@" + "private-domain.invalid"
        directory = "/ho" + "me/PrivateUser"
        value = {"task": "Contact " + address, "trajectory": [{"path": directory + "/temporary.txt"}], "metadata": {address: "private-key-name"}}
        cleaned = remove_private_record(value)
        self.assertEqual(cleaned["task"], "Contact ")
        self.assertEqual(cleaned["trajectory"][0]["path"], "/temporary.txt")
        self.assertEqual(cleaned["metadata"], {})
        self.assertFalse(inspect_text(json.dumps(cleaned)))

    def test_scanner_matches_identity_terms_in_field_names(self):
        term = "Private" + "Identity"
        self.assertTrue(inspect_text('{"' + term + '_field":true}', (term,)))

    def test_reserved_example_addresses_are_not_real_contacts(self):
        self.assertFalse(inspect_text("actor" + "@" + "example.com"))

    def test_service_credentials_are_removed_without_dropping_obligations(self):
        tokens = ("hvs." + "A" * 32, "sk_live_" + "a" * 24,
                  "ghp_" + "a" * 36, "AKIA" + "A" * 16)
        for token in tokens:
            with self.subTest(prefix=token[:4]):
                value = {"trajectory": [{"observation": "token=" + token}],
                         "obligations": ["Revoke the issued token"], token: "private field"}
                self.assertEqual(inspect_text(json.dumps(value))["credentials"], 2)
                cleaned = remove_private_record(value)
                self.assertEqual(cleaned["trajectory"], [{"observation": "token="}])
                self.assertEqual(cleaned["obligations"], value["obligations"])
                self.assertNotIn(token, cleaned)
                self.assertFalse(inspect_text(json.dumps(cleaned)))

    def test_code_decorators_and_api_routes_are_preserved(self):
        value = {"content": r"\n@pytest.fixture\n@dataclasses.dataclass", "route": "/users/staff_u"}
        self.assertEqual(remove_private_record(value), value)
        self.assertFalse(inspect_text(json.dumps(value)))

    def test_home_identifier_removal_preserves_shell_delimiters(self):
        value = "`/ho" + "me/PrivateUser/file.txt`; next command"
        self.assertEqual(remove_private_record(value), "`/file.txt`; next command")

    def test_identity_field_is_deleted_and_remaining_text_is_not_replaced(self):
        term = "Private" + "Identity"
        cleaned = remove_private_record({term + "_field": "value", "content": "written by " + term}, (term,))
        self.assertEqual(cleaned, {"content": "written by "})

    def test_changed_image_cannot_reuse_visual_review(self):
        from pathlib import Path
        import tempfile
        from obligationguard.privacy import inspect_image
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "figure.png"
            path.write_bytes(b"changed content")
            self.assertEqual(inspect_image(path, "old digest")["unreviewed_image_files"], 1)


class ConfigurationTests(unittest.TestCase):
    def test_registry_covers_every_evaluated_model_and_thinking_is_enabled(self):
        runtime = load_config("configs/runtime.toml")
        self.assertEqual(set(runtime["evaluation_models"]), set(EVALUATED_MODELS))
        self.assertTrue(runtime["training_runtime"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(runtime["models"]["judge"]["parameters"]["temperature"], 0)
        self.assertNotIn("temperature", runtime["evaluation_models"]["Kimi-K3"]["parameters"])
        self.assertNotIn("temperature", runtime["evaluation_models"]["Claude-Opus-4.8"]["parameters"])

    def test_benchmark_does_not_require_creation_steps_or_safety_categories(self):
        examples = []
        sizes = [size for size, count in ((1, 24), (2, 21), (3, 36), (4, 30), (5, 9)) for _ in range(count)]
        for i, size in enumerate(sizes):
            domain = "issue_resolution" if i < 40 else "feature_development" if i < 56 else "terminal_operation"
            for positive in (True, False):
                examples.append(Instance(f"fixture-{i}-{positive}", f"Fixture task {i}", ({"step": 1},), tuple(Obligation(f"Fixture action {j}") for j in range(size)) if positive else (), task_domain=domain, pair_id=f"fixture-pair-{i}"))
        self.assertEqual(validate_benchmark(examples)["obligations"], 339)


class SnapshotTests(unittest.TestCase):
    def environment(self):
        class Config:
            executable = "docker"
            def model_dump(self, **kwargs): return {"executable": "docker"}
        return SimpleNamespace(container_id="fixture-container", config=Config())

    def test_checkpoint_freezes_processes_before_committing_filesystem(self):
        from pathlib import Path
        calls = []
        def command(*arguments):
            calls.append(arguments)
            return '[{"Mounts":[]}]' if arguments[1] == "inspect" else "fixture-image"
        with patch("obligationguard.container_state.docker_command", side_effect=command), patch.object(Path, "mkdir"):
            result = capture_environment(self.environment(), Path("checkpoint"), "fixture-tag", "checkpoint")
        self.assertEqual(calls[1][1:3], ("checkpoint", "create"))
        self.assertEqual(calls[2][1], "commit")
        self.assertEqual(result["mode"], "checkpoint")

    def test_idle_mode_rejects_a_persistent_process(self):
        from pathlib import Path
        with patch("obligationguard.container_state.docker_command", side_effect=['[{"Mounts":[]}]', "COMMAND\nsleep\npython"]), patch.object(Path, "mkdir"):
            with self.assertRaises(ConfigurationError):
                capture_environment(self.environment(), Path("checkpoint"), "fixture-tag", "idle_container")

    def test_external_mounts_cannot_be_omitted_from_a_snapshot(self):
        from pathlib import Path
        with patch("obligationguard.container_state.docker_command", return_value='[{"Mounts":[{}]}]'):
            with self.assertRaises(ConfigurationError):
                capture_environment(self.environment(), Path("checkpoint"), "fixture-tag", "checkpoint")
