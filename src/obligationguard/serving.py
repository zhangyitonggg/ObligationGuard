from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.request

from .backends import ChatBackend
from .errors import ConfigurationError
from .io import fingerprint


class ModelServer:
    def __init__(self, config: dict):
        self.config = config
        server = config["server"]
        self.url = f"http://127.0.0.1:{server['port']}/v1"
        self.process = None
        self.stream = None

    def start(self):
        try:
            urllib.request.urlopen(self.url + "/models", timeout=2).close()
        except (urllib.error.URLError, TimeoutError):
            pass
        else:
            raise ConfigurationError("the configured managed-server port is already in use")
        server = self.config["server"]
        executable = server.get("python", sys.executable)
        if "$" in executable:
            raise ConfigurationError("set OBLIGATIONGUARD_INFERENCE_PYTHON to the vLLM environment's Python executable")
        directory = Path(server["logs"])
        directory.mkdir(parents=True, exist_ok=True)
        self.stream = (directory / (fingerprint(self.config) + ".log")).open("a", encoding="utf-8")
        command = [executable, "-m", "vllm.entrypoints.openai.api_server", "--model", self.config["model"], "--host", "127.0.0.1", "--port", str(server["port"]), "--max-model-len", str(self.config.get("max_context_length", 32768)), "--tensor-parallel-size", str(server.get("tensor_parallel_size", 1)), "--gpu-memory-utilization", str(server["gpu_memory_utilization"])]
        if self.config.get("revision"):
            command += ["--revision", self.config["revision"]]
        command += server.get("extra_arguments", [])
        environment = dict(os.environ)
        cache = environment.get("OBLIGATIONGUARD_CACHE_ROOT")
        if cache:
            environment.setdefault("HF_HOME", str(Path(cache) / "huggingface"))
            environment.setdefault("TORCH_HOME", str(Path(cache) / "torch"))
        if "cuda_visible_devices" in server:
            environment["CUDA_VISIBLE_DEVICES"] = str(server["cuda_visible_devices"])
        try:
            self.process = subprocess.Popen(command, stdout=self.stream, stderr=subprocess.STDOUT, env=environment)
            deadline = time.monotonic() + server.get("startup_timeout", 900)
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise ConfigurationError("model server exited during startup; inspect its log")
                try:
                    with urllib.request.urlopen(self.url + "/models", timeout=2) as response:
                        models = json.load(response)
                    if self.config["model"] in {row["id"] for row in models["data"]}:
                        return self.url
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            raise ConfigurationError("model server did not become ready within the configured time")
        except Exception:
            self.close()
            raise

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=30)
        if self.stream is not None:
            self.stream.close()


class ManagedChatBackend(ChatBackend):
    def __init__(self, config: dict):
        self.original_config = dict(config)
        self.server = ModelServer(config)
        base_url = self.server.start()
        try:
            super().__init__({**config, "base_url": base_url, "token_count_backend": "vllm"})
        except Exception:
            self.server.close()
            raise

    def identity(self):
        return self.original_config

    def close(self):
        self.server.close()

    def __del__(self):
        if getattr(self, "server", None) is not None:
            self.server.close()
