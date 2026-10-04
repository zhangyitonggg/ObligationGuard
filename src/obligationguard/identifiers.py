from .io import fingerprint


# The prompt's JSON identifier is a constant, never a dataset identifier.
MODEL_TASK_ID = "task"


def opaque_instance_id(source_id: str, namespace: str) -> str:
    """Create an opaque bookkeeping identifier without exposing source prefixes."""
    return "sample-" + fingerprint({"namespace": namespace, "source_id": source_id})[:32]
