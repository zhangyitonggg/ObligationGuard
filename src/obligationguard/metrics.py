from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .errors import DataError


@dataclass(frozen=True)
class InstanceScore:
    instance_id: str
    prediction_count: int
    truth_count: int
    matched_count: int

    def __post_init__(self):
        values = (self.prediction_count, self.truth_count, self.matched_count)
        if any(type(x) is not int or x < 0 for x in values) or self.matched_count > min(values[:2]):
            raise DataError("invalid obligation counts")

    @property
    def exact_match(self) -> bool:
        return self.matched_count == self.prediction_count == self.truth_count


def aggregate(scores: Iterable[InstanceScore]) -> dict:
    scores = list(scores)
    if len({x.instance_id for x in scores}) != len(scores):
        raise DataError("duplicate instance score")
    positive = [x for x in scores if x.truth_count > 0]
    negative = [x for x in scores if x.truth_count == 0]
    matched = sum(x.matched_count for x in positive)
    predictions = sum(x.prediction_count for x in positive)
    truth = sum(x.truth_count for x in positive)
    ratio = lambda n, d: n / d if d else None
    return {
        "precision": ratio(matched, predictions) if predictions else (0.0 if positive else None),
        "recall": ratio(matched, truth),
        "exact_match": ratio(sum(x.exact_match for x in positive), len(positive)),
        "ca_positive": ratio(sum(x.prediction_count > 0 for x in positive), len(positive)),
        "ca_negative": ratio(sum(x.prediction_count == 0 for x in negative), len(negative)),
        "counts": {"instances": len(scores), "positive_instances": len(positive), "negative_instances": len(negative), "matched_obligations": matched, "predicted_obligations_positive": predictions, "ground_truth_obligations_positive": truth},
    }
