"""Per-channel R_* calibration (Slice 4 / M4).

Fits a univariate map raw R_k → P(correct | R_k) on OOF labels. When a channel
has too little variation or single-class labels, falls back to the identity map
(literature-style pass-through with documented justification).
"""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from rockburst.fusion.stacker import FEATURE_NAMES


class ComponentCalibrators:
    """Independent calibrators for r_data, r_physics, r_model, r_uncertainty, r_stability."""

    CHANNELS = ("r_data", "r_physics", "r_model", "r_uncertainty", "r_stability")

    def __init__(self, conf_eps: float = 1e-3):
        self.conf_eps = float(conf_eps)
        self.models: dict[str, Any] = {}
        self.methods: dict[str, str] = {}
        self.n_fit: dict[str, int] = {}
        self.is_fit = False

    def fit(self, R: np.ndarray, y_correct: np.ndarray) -> "ComponentCalibrators":
        R = np.asarray(R, dtype=float)
        y = np.asarray(y_correct, dtype=int).ravel()
        self.models.clear()
        self.methods.clear()
        self.n_fit.clear()
        for name in self.CHANNELS:
            i = FEATURE_NAMES.index(name)
            if i >= R.shape[1]:
                continue
            x = R[:, i]
            self.n_fit[name] = int(len(x))
            if len(np.unique(y)) < 2 or len(np.unique(np.round(x, 5))) < 3:
                self.models[name] = None
                self.methods[name] = "identity_passthrough"
                continue
            if len(x) < 40 or len(np.unique(np.round(x, 5))) < 5:
                lr = LogisticRegression(solver="lbfgs", max_iter=1000)
                lr.fit(x.reshape(-1, 1), y)
                self.models[name] = lr
                self.methods[name] = "platt"
            else:
                iso = IsotonicRegression(
                    out_of_bounds="clip",
                    y_min=self.conf_eps,
                    y_max=1.0 - self.conf_eps,
                )
                iso.fit(x, y)
                self.models[name] = iso
                self.methods[name] = "isotonic"
        self.is_fit = True
        return self

    def transform_one(self, name: str, raw: float) -> float:
        eps = self.conf_eps
        raw = float(raw)
        if not self.is_fit or name not in self.models or self.models[name] is None:
            return float(np.clip(raw, eps, 1.0 - eps))
        model = self.models[name]
        if self.methods.get(name) == "platt":
            val = float(model.predict_proba([[raw]])[0, 1])
        else:
            val = float(model.predict([raw])[0])
        return float(np.clip(val, eps, 1.0 - eps))

    def transform_vector(self, r_vec: np.ndarray) -> np.ndarray:
        r = np.asarray(r_vec, dtype=float).ravel().copy()
        for name in self.CHANNELS:
            i = FEATURE_NAMES.index(name)
            if i < len(r):
                r[i] = self.transform_one(name, float(r[i]))
        return r

    def report(self) -> dict[str, Any]:
        return {
            "methods": dict(self.methods),
            "n_fit": dict(self.n_fit),
            "note": (
                "identity_passthrough = insufficient variation or single-class OOF; "
                "channel kept as raw heuristic (literature-style constant map)."
            ),
        }

    def save(self, path) -> None:
        joblib.dump(
            {
                "models": self.models,
                "methods": self.methods,
                "n_fit": self.n_fit,
                "conf_eps": self.conf_eps,
                "is_fit": self.is_fit,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "ComponentCalibrators":
        data = joblib.load(path)
        obj = cls(conf_eps=float(data.get("conf_eps", 1e-3)))
        obj.models = data.get("models") or {}
        obj.methods = data.get("methods") or {}
        obj.n_fit = data.get("n_fit") or {}
        obj.is_fit = bool(data.get("is_fit", False))
        return obj


# Candidate fusion masks (Item 9 / Slice 4 M4 recomparison after M1/M2/C3).
FUSION_SET_CANDIDATES: dict[str, tuple[str, ...]] = {
    "A": ("r_model",),
    "B": ("r_model", "n_missing"),
    "C": ("r_data", "r_physics", "r_model", "r_uncertainty", "r_stability", "n_missing"),
    "D": tuple(FEATURE_NAMES),
}
SIMPLICITY_ORDER = ("A", "B", "C", "D")


def mask_for_set(set_id: str) -> np.ndarray:
    feats = FUSION_SET_CANDIDATES[set_id]
    return np.array([n in feats for n in FEATURE_NAMES], dtype=bool)
