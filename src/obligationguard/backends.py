from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from .errors import ConfigurationError, ModelOutputError


@dataclass(frozen=True)
class Completion:
    text: str
    raw: dict
    request: dict
    error: str | None = None


class Backend(Protocol):
    def identity(self) -> dict: ...
    def complete(self, prompt: str) -> Completion: ...


def decode_json_output(text: str) -> dict:
    stripped = final_answer(text).strip()
    if stripped.startswith("```json\n") and stripped.endswith("```"):
        stripped = stripped[8:-3].strip()
    elif stripped.startswith("```\n") and stripped.endswith("```"):
        stripped = stripped[4:-3].strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise ModelOutputError(f"model output is not one complete JSON object: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ModelOutputError("model output must be a JSON object")
    return value


def completion_result(text: str, raw: dict, request: dict, error: str | None = None) -> Completion:
    """Retain generated output even when its final answer is unavailable."""
    if not text:
        error = error or "model returned no output text"
    try:
        answer = final_answer(text)
    except ModelOutputError as exc:
        answer, error = text, error or str(exc)
    return Completion(answer, raw, request, error)


def _post(url: str, body: dict, headers: dict, timeout: int) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ModelOutputError(f"model endpoint returned HTTP {exc.code}: {detail[:1000]}") from exc


class ResponsesBackend:
    """OpenAI Responses interface, with input-token counting before generation."""
    def __init__(self, config: dict):
        self.config = dict(config)
        self.model = config["model"]
        self.base_url = config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.key_variable = config.get("api_key_env", "OPENAI_API_KEY")
        self.max_output_tokens = config["max_output_tokens"]
        self.max_context = config.get("max_context_length", 32768)
        self.parameters = config.get("parameters", {})
        forbidden = {"model", "input", "tools", "previous_response_id", "conversation", "max_output_tokens"} & self.parameters.keys()
        if forbidden:
            raise ConfigurationError(f"Responses parameters cannot override {sorted(forbidden)}")
        if not os.environ.get(self.key_variable):
            raise ConfigurationError(f"set the {self.key_variable} environment variable")

    def identity(self):
        return {key: value for key, value in self.config.items() if key != "api_key"}

    def complete(self, prompt: str) -> Completion:
        headers = {"Authorization": "Bearer " + os.environ[self.key_variable]}
        inputs = [{"role": "user", "content": prompt}]
        count = _post(self.base_url + "/responses/input_tokens", {"model": self.model, "input": inputs}, headers, self.config.get("timeout", 600))["input_tokens"]
        if count + self.max_output_tokens > self.max_context:
            raise ModelOutputError("complete input and output budget exceed the configured context limit")
        request = {"model": self.model, "input": inputs, "max_output_tokens": self.max_output_tokens, "store": False, **self.parameters}
        raw = _post(self.base_url + "/responses", request, headers, self.config.get("timeout", 600))
        if raw.get("status") not in {"completed", "incomplete"}:
            raise ModelOutputError(f"model response did not complete: {raw.get('status')}")
        text = "".join(item["text"] for output in raw.get("output", []) if output.get("type") == "message" for item in output.get("content", []) if item.get("type") == "output_text")
        error = "model response was incomplete: " + str(raw.get("incomplete_details")) if raw.get("status") == "incomplete" else None
        return completion_result(text, raw, request, error)


class ChatBackend:
    """Chat Completions interface with an explicitly configured tokenizer."""
    def __init__(self, config: dict):
        self.config = dict(config)
        self.model = config["model"]
        self.key_variable = config.get("api_key_env")
        if self.key_variable and not os.environ.get(self.key_variable):
            raise ConfigurationError(f"set the {self.key_variable} environment variable")
        if config.get("token_count_backend") not in {"moonshot", "vllm"}:
            from transformers import AutoTokenizer, PreTrainedTokenizerFast
            tokenizer_class = PreTrainedTokenizerFast if config.get("tokenizer_backend") == "fast" else AutoTokenizer
            self.tokenizer = tokenizer_class.from_pretrained(config["tokenizer"], revision=config.get("tokenizer_revision"))

    def identity(self):
        return self.config

    def complete(self, prompt: str) -> Completion:
        config = self.config
        messages = [{"role": "user", "content": prompt}]
        headers = {"Authorization": "Bearer " + os.environ[self.key_variable]} if self.key_variable else {}
        if config.get("token_count_backend") == "moonshot":
            count = _post(config["base_url"].rstrip("/") + "/tokenizers/estimate-token-count", {"model": self.model, "messages": messages}, headers, config.get("timeout", 600))["data"]["total_tokens"]
        elif config.get("token_count_backend") == "vllm":
            endpoint = config["base_url"].rstrip("/").removesuffix("/v1") + "/tokenize"
            request = {"model": self.model, "messages": messages, "add_generation_prompt": True, "add_special_tokens": False, "chat_template_kwargs": config.get("api_chat_template_kwargs", config.get("chat_template_kwargs", {}))}
            count = _post(endpoint, request, headers, config.get("timeout", 600))["count"]
        else:
            count = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **config.get("chat_template_kwargs", {})))
        if count + config["max_output_tokens"] > config.get("max_context_length", 32768):
            raise ModelOutputError("complete input and output budget exceed the configured context limit")
        parameters = config.get("parameters", {})
        if {"model", "messages", "n", "max_tokens"} & parameters.keys():
            raise ConfigurationError("chat parameters cannot override model, messages, n, or max_tokens")
        request = {"model": self.model, "messages": messages, "n": 1, "max_tokens": config["max_output_tokens"], **parameters}
        if "api_chat_template_kwargs" in config:
            request["chat_template_kwargs"] = config["api_chat_template_kwargs"]
        raw = _post(config["base_url"].rstrip("/") + "/chat/completions", request, headers, config.get("timeout", 600))
        choices = raw.get("choices", [])
        if len(choices) != 1:
            raise ModelOutputError("expected exactly one model response")
        text = choices[0]["message"].get("content")
        error = "chat response did not complete: " + str(choices[0].get("finish_reason")) if choices[0].get("finish_reason") != "stop" else None
        return completion_result(text if isinstance(text, str) else "", raw, request, error)


class LocalBackend:
    def __init__(self, config: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.config = dict(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config.get("revision"))
        self.model = AutoModelForCausalLM.from_pretrained(config["model"], revision=config.get("revision"), torch_dtype=torch.bfloat16, device_map="auto", attn_implementation=config.get("attention_implementation", "sdpa"))
        self.model.eval()

    def identity(self):
        return self.config

    def complete(self, prompt: str) -> Completion:
        import torch
        messages = [{"role": "user", "content": prompt}]
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", **self.config.get("chat_template_kwargs", {})).to(self.model.device)
        output_budget = self.config["max_output_tokens"]
        if encoded.shape[-1] + output_budget > self.config.get("max_context_length", 32768):
            raise ModelOutputError("complete input and output budget exceed the configured context limit")
        with torch.inference_mode():
            generated = self.model.generate(encoded, attention_mask=torch.ones_like(encoded), max_new_tokens=output_budget, do_sample=False, pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        tokens = generated[0, encoded.shape[-1]:]
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        error = "local model output reached its token limit" if len(tokens) == output_budget and tokens[-1].item() not in eos_ids else None
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return completion_result(text, {"output_token_ids": tokens.tolist()}, {"messages": messages, "max_new_tokens": output_budget, "do_sample": False}, error)


def final_answer(text: str) -> str:
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        raise ModelOutputError("model returned an unfinished reasoning block")
    return text


class AnthropicBackend:
    def __init__(self, config: dict):
        self.config = dict(config)
        self.key_variable = config.get("api_key_env", "ANTHROPIC_API_KEY")
        if not os.environ.get(self.key_variable):
            raise ConfigurationError(f"set the {self.key_variable} environment variable")

    def identity(self):
        return self.config

    def complete(self, prompt: str) -> Completion:
        config = self.config
        base = config.get("base_url", "https://api.anthropic.com/v1").rstrip("/")
        headers = {"x-api-key": os.environ[self.key_variable], "anthropic-version": "2023-06-01"}
        messages = [{"role": "user", "content": prompt}]
        parameters = config.get("parameters", {})
        if {"model", "messages", "max_tokens", "tools", "system"} & parameters.keys():
            raise ConfigurationError("Anthropic parameters cannot replace model inputs or add tools")
        count = _post(base + "/messages/count_tokens", {"model": config["model"], "messages": messages}, headers, config.get("timeout", 600))["input_tokens"]
        if count + config["max_output_tokens"] > config.get("max_context_length", 32768):
            raise ModelOutputError("complete input and output budget exceed the configured context limit")
        request = {"model": config["model"], "messages": messages, "max_tokens": config["max_output_tokens"], **parameters}
        raw = _post(base + "/messages", request, headers, config.get("timeout", 600))
        text = "".join(block["text"] for block in raw.get("content", []) if block.get("type") == "text")
        error = "Anthropic response did not complete: " + str(raw.get("stop_reason")) if raw.get("stop_reason") != "end_turn" else None
        return completion_result(text, raw, request, error)


def make_backend(config: dict) -> Backend:
    from .serving import ManagedChatBackend
    constructors = {"responses": ResponsesBackend, "chat": ChatBackend, "managed_chat": ManagedChatBackend, "local": LocalBackend, "anthropic": AnthropicBackend}
    backend = config.get("backend")
    if backend not in constructors:
        raise ConfigurationError(f"unknown model backend: {backend}")
    if "astra" in str(config.get("model", "")).lower():
        raise ConfigurationError("the configured model is excluded by the workspace model policy")
    if not isinstance(config.get("model"), str) or not config["model"] or "$" in config["model"]:
        raise ConfigurationError("set an explicit model identifier or checkpoint path")
    if "api_key" in config:
        raise ConfigurationError("configure api_key_env rather than storing credentials in a model configuration")
    if type(config.get("max_output_tokens")) is not int or config["max_output_tokens"] <= 0:
        raise ConfigurationError("max_output_tokens must be explicitly set to a positive integer")
    return constructors[backend](config)
