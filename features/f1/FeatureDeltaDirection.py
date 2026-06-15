import numpy as np
import frams
from features.Feature import Feature


class FeatureDeltaDirection(Feature):
    def __init__(self):
        ...

    def _part_pos(self, part):
        return np.array([
            part.x._double(),
            part.y._double(),
            part.z._double(),
        ], dtype=float)

    def _part_dir(self, first, second):
        direction = self._part_pos(second) - self._part_pos(first)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.array([0.0, 0.0, 0.0], dtype=float)
        return direction / norm

    def _model_from_seq(self, seq):
        if not seq:
            return None

        try:
            model = frams.Model.newFromString(seq)
        except Exception:
            return None

        if model.is_valid._int() == 0 or model.numparts._int() == 0:
            return None

        return model

    def get_feature_value(self, seq):
        model = self._model_from_seq(seq)
        if model is None:
            return np.array([0.0], dtype=float)

        num_parts = model.numparts._int()
        if num_parts == 1:
            return np.array([0.0], dtype=float)

        first_dir = self._part_dir(model.getPart(0), model.getPart(1))
        last_dir = self._part_dir(model.getPart(num_parts - 2), model.getPart(num_parts - 1))

        dot = float(np.dot(first_dir, last_dir))
        dot = max(-1.0, min(1.0, dot))
        angle = float(np.degrees(np.arccos(dot)))

        return np.array([angle], dtype=float)
