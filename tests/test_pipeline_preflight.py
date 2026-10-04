import unittest
from unittest.mock import patch

from obligationguard.backends import ChatBackend
from obligationguard.errors import ConfigurationError
from obligationguard.pipeline import pipeline
from obligationguard.serving import ManagedChatBackend


class PreflightTests(unittest.TestCase):
    def test_invalid_stages_never_start_training(self):
        for stages in (["unknown"], ["rq1", "rq1"], "rq1", [None], [{}]):
            with self.subTest(stages=stages), patch("obligationguard.pipeline.readiness") as checks, patch("obligationguard.pipeline.launch_training") as train:
                with self.assertRaises(ConfigurationError):
                    pipeline({}, {"pipeline": {"stages": stages}})
                checks.assert_not_called()
                train.assert_not_called()

    def test_missing_requirements_never_start_training(self):
        with patch("obligationguard.pipeline.readiness", return_value={"benchmark": False}), patch("obligationguard.pipeline.launch_training") as train:
            with self.assertRaises(ConfigurationError):
                pipeline({}, {"pipeline": {"stages": ["rq1"]}})
            train.assert_not_called()


class ManagedContextTests(unittest.TestCase):
    def config(self):
        return {"backend": "managed_chat", "model": "fixture-model", "max_output_tokens": 20, "max_context_length": 100, "server": {"port": 8100, "gpu_memory_utilization": 0.7, "logs": "fixture-logs"}, "api_chat_template_kwargs": {"enable_thinking": True}}

    def test_server_counts_with_the_same_template_used_for_generation(self):
        with patch("obligationguard.serving.ModelServer.start", return_value="http://127.0.0.1:8100/v1"), patch("obligationguard.serving.ModelServer.close"), patch("obligationguard.backends._post", side_effect=[{"count": 60}, {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}]) as request:
            backend = ManagedChatBackend(self.config())
            completion = backend.complete("fixture prompt")
            self.assertEqual(completion.text, "{}")
            self.assertEqual(request.call_args_list[0].args[0], "http://127.0.0.1:8100/tokenize")
            self.assertEqual(request.call_args_list[0].args[1]["chat_template_kwargs"], {"enable_thinking": True})
            self.assertEqual(request.call_args_list[1].args[1]["chat_template_kwargs"], {"enable_thinking": True})
            self.assertFalse(hasattr(backend, "tokenizer"))
            backend.close()

    def test_overlength_input_is_rejected_before_generation(self):
        from obligationguard.errors import ModelOutputError
        config = {**self.config(), "backend": "chat", "base_url": "http://127.0.0.1:8100/v1", "token_count_backend": "vllm"}
        with patch("obligationguard.backends._post", return_value={"count": 81}) as request:
            backend = ChatBackend(config)
            with self.assertRaises(ModelOutputError):
                backend.complete("fixture prompt")
            self.assertEqual(request.call_count, 1)
