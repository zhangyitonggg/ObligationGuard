from __future__ import annotations

from copy import deepcopy
import time
import json
from pathlib import Path
from .backends import Backend
from .evaluation import parse_record
from .guidance import FirstTerminationGuidance
from .identifiers import MODEL_TASK_ID
from .io import fingerprint, write_json
from .errors import DataError
from .prompting import identification
from .schema import Instance


def configuration_path(settings: dict) -> Path:
    """Use an explicit config or the pinned agent package's official default."""
    configured = settings.get("mini_config")
    if configured:
        path = Path(configured)
    else:
        from importlib.resources import files
        path = Path(str(files("minisweagent").joinpath("config/default.yaml")))
    if not path.is_file():
        raise DataError("Mini-SWE-Agent configuration is unavailable")
    return path


def agent_class():
    """Create the Mini-SWE-Agent subclass without importing optional packages at startup."""
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.exceptions import Submitted

    class ObligationGuidedAgent(DefaultAgent):
        def __init__(self, *args, condition: str, task_id: str, guard: Backend | None = None, on_first_termination=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.guidance = FirstTerminationGuidance(condition)
            self.task_id = task_id
            self.guard = guard
            self.on_first_termination = on_first_termination
            self.first_termination = None
            self.termination_messages = None
            self.guard_response = None

        def execute_actions(self, message):
            try:
                return super().execute_actions(message)
            except Submitted as submitted:
                if self.guidance.attempted:
                    raise
                self.first_termination = deepcopy(self.serialize())
                self.termination_messages = deepcopy(submitted.messages)
                self.first_termination["resume"] = {"task": self.extra_template_vars["task"], "template_vars": deepcopy(self.extra_template_vars), "elapsed_seconds": time.time() - self._start_time, "format_errors": self.n_consecutive_format_errors, "termination_messages": self.termination_messages}
                if self.on_first_termination is not None:
                    self.on_first_termination(self, self.first_termination)
                feedback = self.make_feedback()
                if feedback is None:
                    raise
                return self.add_messages(self.model.format_message(role="user", content=feedback))

        def make_feedback(self):
            obligations = ()
            if self.guidance.condition == "guard":
                if self.guard is None:
                    raise ValueError("guard condition requires a guard backend")
                instance = Instance(self.task_id, self.extra_template_vars["task"], tuple(self.messages), ())
                prompt = identification(instance)
                identity = fingerprint({"prompt": prompt, "model": self.guard.identity()})
                path = Path(self.config.output_path).parent / "guard_response.json" if self.config.output_path else None
                if path and path.is_file():
                    self.guard_response = json.loads(path.read_text(encoding="utf-8"))
                    if self.guard_response["fingerprint"] != identity:
                        raise DataError("saved guard feedback uses different inputs or model settings")
                else:
                    completion = self.guard.complete(prompt)
                    self.guard_response = {"fingerprint": identity, "text": completion.text, "raw_response": completion.raw, "request": completion.request, "model_settings": self.guard.identity(), "completion_error": completion.error}
                obligations, error = parse_record(self.guard_response, MODEL_TASK_ID)
                self.guard_response["parse_error"] = error
                if path:
                    write_json(path, self.guard_response)
            return self.guidance.feedback(obligations)

        def resume(self, state: dict):
            from minisweagent.exceptions import FormatError, InterruptAgentFlow
            self.messages = deepcopy(state["messages"])
            self.extra_template_vars = deepcopy(state["resume"]["template_vars"])
            self.cost = state["info"]["model_stats"]["instance_cost"]
            self.n_calls = state["info"]["model_stats"]["api_calls"]
            self.n_consecutive_format_errors = state["resume"]["format_errors"]
            self._start_time = time.time() - state["resume"]["elapsed_seconds"]
            self.first_termination = deepcopy(state)
            self.termination_messages = deepcopy(state["resume"]["termination_messages"])
            feedback = self.make_feedback()
            if feedback is None:
                self.add_messages(*self.termination_messages)
            else:
                self.add_messages(self.model.format_message(role="user", content=feedback))
            while self.messages[-1].get("role") != "exit":
                try:
                    self.step()
                    self.n_consecutive_format_errors = 0
                except FormatError as exc:
                    self.cost += exc.messages[0].get("extra", {}).get("cost", 0.0)
                    self.n_consecutive_format_errors += 1
                    self.add_messages(*exc.messages)
                    if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                        self.add_messages({"role": "exit", "content": "RepeatedFormatError", "extra": {"exit_status": "RepeatedFormatError", "submission": ""}})
                except InterruptAgentFlow as exc:
                    self.add_messages(*exc.messages)
                except Exception as exc:
                    self.handle_uncaught_exception(exc)
                    raise
                finally:
                    self.save(self.config.output_path)
            self.save(self.config.output_path)
            return self.messages[-1].get("extra", {})

    return ObligationGuidedAgent
