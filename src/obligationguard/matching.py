from __future__ import annotations

from .errors import DataError


def maximum_matching(matrix: list[list[bool]], truth_count: int) -> list[tuple[int, int]]:
    """Maximum bipartite matching: every prediction and truth participates at most once."""
    if type(truth_count) is not int or truth_count < 0:
        raise DataError("truth_count must be a nonnegative integer")
    if any(len(row) != truth_count or any(type(x) is not bool for x in row) for row in matrix):
        raise DataError("matching matrix must be rectangular and contain booleans")
    assigned: dict[int, int] = {}

    def augment(prediction: int, seen: set[int]) -> bool:
        for truth, matches in enumerate(matrix[prediction]):
            if not matches or truth in seen:
                continue
            seen.add(truth)
            previous = assigned.get(truth)
            if previous is None or augment(previous, seen):
                assigned[truth] = prediction
                return True
        return False

    for prediction in range(len(matrix)):
        augment(prediction, set())
    return sorted((prediction, truth) for truth, prediction in assigned.items())
