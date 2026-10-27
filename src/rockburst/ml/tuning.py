"""Nested Optuna hyperparameter search (equal trial budget per model)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC

from rockburst.eval.metrics import quadratic_weighted_kappa
from rockburst.ml.ordinal import FrankHallOrdinalClassifier
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


# Slice 6 / M9: equal Optuna budget ≥50 trials per family (quick mode overrides)
DEFAULT_N_TRIALS = 50

SEARCH_SPACES_DOC = {
    "logistic": {"C": "loguniform(1e-2, 10)"},
    "svm": {
        "kernel": "categorical(linear, rbf)",
        "C": "loguniform(1e-3, 100)",
        "gamma": "loguniform(1e-4, 10) if rbf",
        "note": (
            "RM11: StandardScaler in preprocess; CalibratedClassifierCV(sigmoid); "
            "class_weight=balanced on LinearSVC/SVC; equal n_trials."
        ),
    },
    "rf": {
        "n_estimators": "int(50, 400)",
        "max_depth": "int(3, 16)",
        "min_samples_leaf": "int(1, 12)",
        "max_features": "categorical(sqrt, log2, 0.5, 0.8)",
        "note": "RM11: broadened leaf/feature regularisation vs Slice 6.",
    },
    "mlp": {
        "hidden_layer_sizes": "categorical((32,), (64,), (32, 16), (64, 32))",
        "alpha": "loguniform(1e-5, 1e-1)",
        "learning_rate_init": "loguniform(1e-4, 1e-2)",
        "note": (
            "Replaces ExtraTrees (M9): distinct inductive bias (shallow MLP) vs bagged trees; "
            "avoids near-duplicate RF/ExtraTrees pair."
        ),
    },
    "xgboost": {
        "n_estimators": "int(50, 300)",
        "max_depth": "int(2, 8)",
        "learning_rate": "loguniform(0.01, 0.2)",
        "subsample": "uniform(0.6, 1.0)",
        "colsample_bytree": "uniform(0.5, 1.0)",
        "reg_lambda": "loguniform(1e-3, 10)",
        "note": "RM11: subsample / colsample / L2 regularisation.",
    },
    "lightgbm": {
        "n_estimators": "int(50, 300)",
        "max_depth": "int(2, 10)",
        "learning_rate": "loguniform(0.01, 0.2)",
        "subsample": "uniform(0.6, 1.0)",
        "colsample_bytree": "uniform(0.5, 1.0)",
        "reg_lambda": "loguniform(1e-3, 10)",
        "note": "RM11: subsample / colsample / L2 regularisation.",
    },
    "catboost": {
        "iterations": "int(50, 300)",
        "depth": "int(2, 8)",
        "learning_rate": "loguniform(0.01, 0.2)",
        "subsample": "uniform(0.6, 1.0)",
        "l2_leaf_reg": "loguniform(1e-2, 20)",
        "note": "RM11: subsample + L2 leaf regularisation.",
    },
    "ordinal_logistic": {"alpha": "loguniform(1e-2, 10)"},
    "frank_hall_rf": {
        "n_estimators": "int(50, 250)",
        "max_depth": "int(3, 12)",
        "min_samples_leaf": "int(1, 10)",
        "note": "Frank–Hall ordinal reduction with RF binary thresholds (M8/RM11).",
    },
    "frank_hall_logistic": {
        "C": "loguniform(1e-2, 10)",
        "note": "Frank–Hall ordinal reduction with logistic binary thresholds (M8).",
    },
}

class _FixedTrial:
    def suggest_float(self, n, low, high, log=False):
        return float(np.sqrt(low * high) if log else (low + high) / 2)

    def suggest_int(self, n, low, high):
        return int((low + high) // 2)

    def suggest_categorical(self, n, choices):
        return choices[0]


def _wrap_calibrated_svm(base: Any, seed: int) -> CalibratedClassifierCV:
    # cv=3 + sigmoid: better calibrated probs for ordinal metrics; cost OK at equal budget
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def _suggest_estimator(trial: Any, name: str, seed: int) -> Any:
    if name == "logistic":
        return LogisticRegression(
            C=trial.suggest_float("C", 1e-2, 10.0, log=True),
            max_iter=2000,
            class_weight="balanced",
        )
    if name == "svm":
        # RM11: scaled inputs via Pipeline StandardScaler; balanced weights; sigmoid calib
        kernel = trial.suggest_categorical("kernel", ["linear", "rbf"])
        C = trial.suggest_float("C", 1e-3, 100.0, log=True)
        if kernel == "linear":
            base = LinearSVC(
                C=C,
                class_weight="balanced",
                dual="auto",
                max_iter=12000,
                random_state=seed,
            )
        else:
            gamma = trial.suggest_float("gamma", 1e-4, 10.0, log=True)
            base = SVC(
                C=C,
                kernel="rbf",
                gamma=gamma,
                class_weight="balanced",
                random_state=seed,
            )
        return _wrap_calibrated_svm(base, seed)
    if name == "rf":
        max_features = trial.suggest_categorical(
            "max_features", ["sqrt", "log2", 0.5, 0.8]
        )
        return RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 400),
            max_depth=trial.suggest_int("max_depth", 3, 16),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 12),
            max_features=max_features,
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
    if name == "mlp":
        hidden = trial.suggest_categorical(
            "hidden_layer_sizes",
            [(32,), (64,), (32, 16), (64, 32)],
        )
        return MLPClassifier(
            hidden_layer_sizes=hidden,
            alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            learning_rate_init=trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
            random_state=seed,
        )
    if name == "xgboost" and XGBClassifier is not None:
        return XGBClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            objective="multi:softprob",
            num_class=4,
            random_state=seed,
            n_jobs=1,
            eval_metric="mlogloss",
        )
    if name == "lightgbm" and LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 300),
            max_depth=trial.suggest_int("max_depth", 2, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            random_state=seed,
            class_weight="balanced",
            verbose=-1,
        )
    if name == "catboost" and CatBoostClassifier is not None:
        return CatBoostClassifier(
            iterations=trial.suggest_int("iterations", 50, 300),
            depth=trial.suggest_int("depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            bootstrap_type="Bernoulli",
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 20.0, log=True),
            loss_function="MultiClass",
            verbose=False,
            allow_writing_files=False,
            random_seed=seed,
            auto_class_weights="Balanced",
        )
    if name == "ordinal_logistic" and LogisticAT is not None:
        return LogisticAT(alpha=trial.suggest_float("alpha", 1e-2, 10.0, log=True))
    if name == "frank_hall_rf":
        base = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 50, 250),
            max_depth=trial.suggest_int("max_depth", 3, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
            random_state=seed,
            class_weight="balanced_subsample",
            n_jobs=1,
        )
        return FrankHallOrdinalClassifier(base_estimator=base, n_classes=4)
    if name == "frank_hall_logistic":
        base = LogisticRegression(
            C=trial.suggest_float("C", 1e-2, 10.0, log=True),
            max_iter=2000,
            class_weight="balanced",
        )
        return FrankHallOrdinalClassifier(base_estimator=base, n_classes=4)
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
    names = ["logistic", "svm", "rf", "mlp"]
    if LogisticAT is not None:
        names.append("ordinal_logistic")
    names.extend(["frank_hall_logistic", "frank_hall_rf"])
    if XGBClassifier is not None:
        names.append("xgboost")
    if LGBMClassifier is not None:
        names.append("lightgbm")
    if CatBoostClassifier is not None:
        names.append("catboost")
    if quick:
        return [n for n in names if n in {"logistic", "rf", "mlp", "frank_hall_logistic"}][
            :3
        ] or names[:3]
    return names


def _tuning_cache_key(
    *,
    model_name: str,
    imputer_kind: str,
    n_inner: int,
    n_trials: int,
    seed: int,
    near_dup_rel_tol: float,
    config_hash: str,
    y: np.ndarray,
    groups: np.ndarray,
) -> str:
    payload = {
        "model_name": model_name,
        "imputer_kind": imputer_kind,
        "n_inner": n_inner,
        "n_trials": n_trials,
        "seed": seed,
        "near_dup_rel_tol": near_dup_rel_tol,
        "config_hash": config_hash,
        "n": int(len(y)),
        "y_sum": int(np.asarray(y).sum()),
        "n_groups": int(len(np.unique(groups))),
        "phase4_rm11": True,
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


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
    cache_dir: str | Path | None = None,
    near_dup_rel_tol: float = 0.01,
    config_hash: str = "",
) -> tuple[Any, dict[str, Any], float]:
    """Inner-loop Optuna; returns (best_estimator, best_params, best_inner_qwk)."""
    if optuna is None:
        est = _suggest_estimator(_FixedTrial(), model_name, seed)
        return est, {}, float("nan")

    cache_path = None
    if cache_dir is not None:
        key = _tuning_cache_key(
            model_name=model_name,
            imputer_kind=imputer_kind,
            n_inner=n_inner,
            n_trials=n_trials,
            seed=seed,
            near_dup_rel_tol=near_dup_rel_tol,
            config_hash=config_hash or "none",
            y=np.asarray(y),
            groups=np.asarray(groups),
        )
        cache_path = Path(cache_dir) / f"{model_name}__{imputer_kind}__{key}.json"
        if cache_path.exists():
            blob = json.loads(cache_path.read_text())
            params = dict(blob.get("best_params") or {})
            best_est = estimator_from_params(model_name, params, seed)
            return best_est, params, float(blob.get("best_inner_qwk", float("nan")))

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
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"candidate {model_name!r} failed during inner-CV fit"
                ) from exc
        return float(np.mean(qwks)) if qwks else -1.0

    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_est = _suggest_estimator(study.best_trial, model_name, seed)
    params = dict(study.best_params)
    # Optuna may store tuple as list in JSON round-trip; normalise for MLP
    if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], list):
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    score = float(study.best_value)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # JSON-serialise tuples
        ser_params = {
            k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()
        }
        cache_path.write_text(
            json.dumps(
                {
                    "best_params": ser_params,
                    "best_inner_qwk": score,
                    "model_name": model_name,
                },
                indent=2,
            )
        )
    return best_est, params, score


def estimator_from_params(name: str, params: dict[str, Any], seed: int) -> Any:
    """Build an estimator from a frozen Optuna param dict (no new search)."""
    params = dict(params or {})
    if "hidden_layer_sizes" in params and isinstance(params["hidden_layer_sizes"], list):
        params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])

    class _T:
        def suggest_float(self, n, low, high, log=False):
            return float(params.get(n, np.sqrt(low * high) if log else (low + high) / 2))

        def suggest_int(self, n, low, high):
            return int(params.get(n, (low + high) // 2))

        def suggest_categorical(self, n, choices):
            return params.get(n, choices[0])

    return _suggest_estimator(_T(), name, seed)
