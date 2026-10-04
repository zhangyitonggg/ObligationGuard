from copy import deepcopy
import sys
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from obligationguard.backends import Completion
from obligationguard.mini_swe import agent_class


class Interrupt(Exception):
    def __init__(self, *messages): self.messages = list(messages)


class Submitted(Interrupt): pass
class FormatError(Interrupt): pass


class DefaultAgent:
    def __init__(self, model, env, **kwargs):
        self.model, self.env = model, env
        self.messages = [{"role": "assistant", "content": "Original termination attempt"}]
        self.extra_template_vars = {"task": "Fixture task"}
        self.config = SimpleNamespace(output_path=None, max_consecutive_format_errors=3)
        self.cost, self.n_calls, self.n_consecutive_format_errors = 2, 4, 0
        self._start_time = time.time()
    def execute_actions(self, message):
        raise Submitted({"role": "exit", "content": "Submitted", "extra": {"exit_status": "Submitted", "submission": ""}})
    def serialize(self): return {"messages": self.messages, "info": {"model_stats": {"instance_cost": self.cost, "api_calls": self.n_calls}}}
    def add_messages(self, *messages):
        self.messages.extend(messages)
        return list(messages)
    def step(self):
        self.n_calls += 1
        self.cost += 1
        message = {"role": "assistant", "content": "Submit after revision"}
        self.add_messages(message)
        return self.execute_actions(message)
    def save(self, path): return self.serialize()


class Guard:
    def __init__(self, action=True): self.calls, self.action = 0, action
    def identity(self): return {"model": "fixture-guard"}
    def complete(self, prompt):
        import json
        self.calls += 1
        obligations = [{"required_safety_action": "Revoke temporary access"}] if self.action else []
        return Completion(json.dumps({"task_id": "task", "obligation_count": len(obligations), "obligations": obligations}), {"fixture": True}, {})


class ResumeTests(unittest.TestCase):
    def setUp(self):
        default, exceptions = ModuleType("minisweagent.agents.default"), ModuleType("minisweagent.exceptions")
        default.DefaultAgent = DefaultAgent
        exceptions.Submitted, exceptions.FormatError, exceptions.InterruptAgentFlow = Submitted, FormatError, Interrupt
        self.modules = patch.dict(sys.modules, {"minisweagent.agents.default": default, "minisweagent.exceptions": exceptions})
        self.modules.start()
        self.Agent = agent_class()
        self.model = SimpleNamespace(format_message=lambda **kwargs: kwargs)
    def tearDown(self): self.modules.stop()
    def snapshot(self):
        baseline = self.Agent(self.model, None, condition="no_guidance", task_id="fixture")
        with self.assertRaises(Submitted): baseline.execute_actions({})
        return baseline.first_termination
    def test_capture_does_not_alias_later_messages(self):
        state = self.snapshot()
        saved = deepcopy(state)
        resumed = self.Agent(self.model, None, condition="self_reminder", task_id="fixture")
        resumed.resume(state)
        self.assertEqual(state, saved)
        self.assertEqual(resumed.n_calls, 5)
        self.assertEqual(sum(message["role"] == "user" for message in resumed.messages), 1)
    def test_guard_is_called_once_and_second_termination_finishes(self):
        guard = Guard()
        resumed = self.Agent(self.model, None, condition="guard", task_id="fixture", guard=guard)
        result = resumed.resume(self.snapshot())
        self.assertEqual(guard.calls, 1)
        self.assertEqual(result["exit_status"], "Submitted")
        self.assertTrue(resumed.guidance.attempted)
    def test_empty_guard_does_not_query_the_actor_again(self):
        guard = Guard(False)
        resumed = self.Agent(self.model, None, condition="guard", task_id="fixture", guard=guard)
        resumed.resume(self.snapshot())
        self.assertEqual(guard.calls, 1)
        self.assertEqual(resumed.n_calls, 4)
