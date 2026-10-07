"""Nested Optuna hyperparameter search (equal trial budget per model)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

from rockburst.eval.metrics import quadratic_weighted_kappa
from rockburst.ml.pipelines import build_preprocess

try:
    import optuna
    from optuna.samplers import TPESampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:  # pragma: no cover
    optuna = None

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:  # pragma: no cover
    CatBoostClassifier = None
try:
    from mord import LogisticAT
except ImportError:  # pragma: no cover
    LogisticAT = None


# Documented equal budget for every M1 family
DEFAULT_N_TRIALS = 15

SEARCH_SPACES_DOC = {
    "logistic": {"C": "loguniform(1e-2, 10)"},
    "svm": {
        "kernel": "categorical(linear, rbf)",
        "C": "loguniform(1e-2, 10)",
        "gamma": "loguniform(1e-3, 1) if rbf",
        "note": (
            "Wrapped in CalibratedClassifierCV (sigmoid); reduced space vs full "
            "SVC(probability=True) grid — equal n_trials, not removed."
        ),
    },
    "rf": {"n_estimators": "int(50, 300)", "max_depth": "int(3, 12)"},
    "extratrees": {"n_estimators": "int(50, 300)", "max_depth": "int(3, 12)"},
    "xgboost": {
        "n_estimators": "int(50, 250)",
        "max_depth": "int(2, 6)",
        "learning_rate": "loguniform(0.02, 0.2)",
    },
    "lightgbm": {
        "n_estimators": "int(50, 250)",
        "max_depth": "int(2, 8)",
        "learning_rate": "loguniform(0.02, 0.2)",
    },
    "catboost": {
        "iterations": "int(50, 250)",
        "depth": "int(2, 6)",
        "learning_rate": "loguniform(0.02, 0.2)",
    },
    "ordinal_logistic": {"alpha": "loguniform(1e-2, 10)"},
}


class _FixedTrial:
    def suggest_float(self, n, low, high, log=False):
        return float(np.sqrt(low * high) if log else (low + high) / 2)

    def suggest_int(self, n, low, high):
        return int((low + high) // 2)

    def suggest_categorical(self, n, choices):
        return choices[0]


def _wrap_calibrated_svm(base: Any, seed: int) -> CalibratedClassifierCV:
    # cv=2 keeps nested-Optuna cost tractable; outer/inner loops remain group-aware
    return CalibratedClassifierCV(base, method="sigmoid", cv=2)


def _suggest_estimator(trial: Any, name: str, seed: int) -> Any:
    if name == "logistic":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-2, 10.0, log=True),
            max_iter=2000,
            class_weight="balanced",
        )
    if name == "svm":
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        C = trial.suggest_float("C", 1e-2, 10.0, log=True)
        if kernel == "linear":
            base = LinearSVC(
                C=C,
                class_weight="balanced",
                dual="auto",
                max_iter=8000,
                random_state=seed,
            )
        else:
            gamma = trial.suggest_float("gamma", 1e-3, 1.0, log=True)
            base = SVC(
                C=C,
                kernel="rbf",
                gamma=gamma,
                class_weight="balanced",
            )
        return _wrap_calibrated_svm(base, seed)
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
    if name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            random_state=seed,
            class_weight="balanced",
            n_jobs=1,
        )
    if name == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 250),
            max_depth=trial.suggest_int("max_depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            objective="multi:softprob",
            num_class=4,
            random_state=seed,
            n_jobs=1,
            eval_metric="mlogloss",
        )
    if name == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 250),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            random_state=seed,
            class_weight="balanced",
            verbose=-1,
        )
    if name == "catboost" and CatBoostClassifier is not None:
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 50, 250),
            depth=trial.suggest_int("depth", 2, 6),
            learning_rate=trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            loss_function="MultiClass",
            verbose=False,
            random_seed=seed,
            auto_class_weights="Balanced",
        )
    if name == "ordinal_logistic" and LogisticAT is not None:
        return LogisticAT(alpha=trial.suggest_float("alpha", 1e-2, 10.0, log=True))
    raise ValueError(f"Unknown or unavailable model: {name}")


def _predict_proba_safe(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] < 4:
            pad = np.zeros((proba.shape[0], 4))
            pad[:, : proba.shape[1]] = proba
            return pad
        return proba
    pred = model.predict(X).astype(int)
    proba = np.zeros((len(pred), 4))
    proba[np.arange(len(pred)), np.clip(pred, 0, 3)] = 1.0
    return proba


def available_model_names(quick: bool = False) -> list[str]:
    names = ["logistic", "svm", "rf", "extratrees"]
    if LogisticAT is not None:
        names.append("ordinal_logistic")
    if XGBClassifier is not None:
        names.append("xgboost")
    if LGBMClassifier is not None:
        names.append("lightgbm")
    if CatBoostClassifier is not None:
        names.append("catboost")
    if quick:
        return [n for n in names if n in {"logistic", "rf", "lightgbm", "xgboost"}][:3] or names[:3]
    return names


def nested_optuna_select(
    X,
    y,
    groups,
    *,
    model_name: str,
    imputer_kind: str,
    n_inner: int,
    n_trials: int,
    seed: int,
) -> tuple[Any, dict[str, Any], float]:
    """Inner-loop Optuna; returns (best_estimator, best_params, best_inner_qwk)."""
    if optuna is None:
        est = _suggest_estimator(_FixedTrial(), model_name, seed)
        return est, {}, float("nan")

    n_inner = min(n_inner, max(2, len(np.unique(groups)) // 2))
    sgkf = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)

    def objective(trial: Any) -> float:
        est = _suggest_estimator(trial, model_name, seed)
        qwks = []
        for tr, te in sgkf.split(X, y, groups):
            pipe = Pipeline(
                [("prep", build_preprocess(imputer_kind)), ("model", clone(est))]
            )
            try:
                pipe.fit(X.iloc[tr], y[tr])
                proba = _predict_proba_safe(pipe, X.iloc[te])
                qwks.append(quadratic_weighted_kappa(y[te], proba.argmax(axis=1)))
            except Exception:  # noqa: BLE001
                return -1.0
        return float(np.mean(qwks)) if qwks else -1.0

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_est = _suggest_estimator(study.best_trial, model_name, seed)
    return best_est, dict(study.best_params), float(study.best_value)


def estimator_from_params(name: str, params: dict[str, Any], seed: int) -> Any:
    """Build an estimator from a frozen Optuna param dict (no new search)."""

    class _T:
        def suggest_float(self, n, low, high, log=False):
            return float(params.get(n, np.sqrt(low * high) if log else (low + high) / 2))

        def suggest_int(self, n, low, high):
            return int(params.get(n, (low + high) // 2))

        def suggest_categorical(self, n, choices):
            return params.get(n, choices[0])

    return _suggest_estimator(_T(), name, seed)
