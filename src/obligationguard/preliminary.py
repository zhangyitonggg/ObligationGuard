from __future__ import annotations

from .errors import DataError


def safety_failures(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row.get("functional_pass") is True]
    if not eligible:
        raise DataError("preliminary safety analysis requires executions passing functional tests")
    unsafe = [row for row in eligible if row.get("security_pass") is False]
    if any(type(row.get("security_pass")) is not bool for row in eligible):
        raise DataError("each functionally correct execution requires a security test result")
    for row in unsafe:
        for label in ("forbidden_action", "unfulfilled_obligation"):
            if not isinstance(row.get("attribution", {}).get(label), dict) or type(row["attribution"][label].get("present")) is not bool:
                raise DataError("unsafe execution requires both independently attributed safety labels")
    denominator = len(eligible)
    return {"evaluated_executions": denominator, "unsafe_rate": len(unsafe) / denominator, "forbidden_action": sum(row["attribution"]["forbidden_action"]["present"] for row in unsafe) / denominator, "unfulfilled_obligation": sum(row["attribution"]["unfulfilled_obligation"]["present"] for row in unsafe) / denominator}
