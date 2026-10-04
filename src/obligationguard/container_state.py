from __future__ import annotations

import json
from pathlib import Path
import subprocess

from .errors import ConfigurationError


def docker_command(executable: str, *arguments: str) -> str:
    return subprocess.run([executable, *arguments], check=True, capture_output=True, text=True, encoding="utf-8", timeout=600).stdout.strip()


def capture_environment(environment, directory: Path, tag: str, mode: str) -> dict:
    executable = environment.config.executable
    identifier = environment.container_id
    inspection = json.loads(docker_command(executable, "inspect", identifier))[0]
    if inspection.get("Mounts"):
        raise ConfigurationError("shared-state snapshots require a container without external mounts")
    if mode not in {"checkpoint", "idle_container"}:
        raise ConfigurationError("snapshot_mode must be checkpoint or idle_container")
    directory.mkdir(parents=True, exist_ok=True)
    if mode == "checkpoint":
        docker_command(executable, "checkpoint", "create", "--checkpoint-dir", str(directory.resolve()), identifier, "first-termination")
    else:
        processes = docker_command(executable, "top", identifier, "-eo", "comm").splitlines()[1:]
        if not processes or any(process.strip() != "sleep" for process in processes):
            raise ConfigurationError("container has persistent processes; use snapshot_mode = checkpoint to preserve their state")
    image = docker_command(executable, "commit", identifier, tag)
    return {"image": image, "tag": tag, "mode": mode, "checkpoint_directory": str(directory.resolve()) if mode == "checkpoint" else None, "environment": environment.config.model_dump(mode="json")}


def environment_class(snapshot: dict | None = None, *, filesystem_only: bool = False):
    from minisweagent.environments.docker import DockerEnvironment

    class SharedStateEnvironment(DockerEnvironment):
        def _start_container(self):
            if snapshot is None or snapshot["mode"] == "idle_container" or filesystem_only:
                return super()._start_container()
            self.container_id = docker_command(self.config.executable, "create", "-w", self.config.cwd, *self.config.run_args, self.config.image, "sleep", self.config.container_timeout)
            docker_command(self.config.executable, "start", "--checkpoint-dir", snapshot["checkpoint_directory"], "--checkpoint", "first-termination", self.container_id)

        def cleanup(self):
            identifier = getattr(self, "container_id", None)
            if identifier:
                subprocess.run([self.config.executable, "rm", "-f", identifier], check=False, capture_output=True, timeout=60)
                self.container_id = None

    return SharedStateEnvironment


def patch(environment) -> str:
    result = environment.execute({"command": "git add -N . && git diff --binary HEAD"})
    if result["returncode"] != 0:
        raise ConfigurationError("cannot extract a patch from the task repository")
    return result["output"]
