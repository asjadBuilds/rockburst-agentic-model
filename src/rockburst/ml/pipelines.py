"""Leakage-safe sklearn pipelines and model zoo."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from rockburst.ml.ordinal import FrankHallOrdinalClassifier
from rockburst.physics.relations import FEATURE_COLS

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


class MissingIndicatorAdder:
    """Add observed_* indicators; fit is no-op (stateless)."""

    def __init__(self, columns: list[str] | None = None):
        self.columns = columns or FEATURE_COLS

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            df = pd.DataFrame(X, columns=self.columns)
        for c in self.columns:
            df[f"observed_{c}"] = (~df[c].isna()).astype(float)
        return df

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        cols = list(self.columns) + [f"observed_{c}" for c in self.columns]
        return np.array(cols)


def make_imputer(kind: str = "median"):
    if kind == "median":
        return SimpleImputer(strategy="median")
    if kind == "knn":
        return KNNImputer(n_neighbors=5)
    if kind == "iterative":
        return IterativeImputer(random_state=42, max_iter=10)
    raise ValueError(kind)


def build_preprocess(imputer_kind: str = "median") -> Pipeline:
    """Impute + scale numeric features; indicators already in frame."""
    num_cols = FEATURE_COLS + [f"observed_{c}" for c in FEATURE_COLS]
    return Pipeline(
        steps=[
            ("imputer", make_imputer(imputer_kind)),
            ("scaler", StandardScaler()),
        ]
    )


def get_model_zoo(quick: bool = False) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "svm": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", dual="auto", max_iter=5000, random_state=42),
            method="sigmoid",
            cv=2,
        ),
        "rf": RandomForestClassifier(
            n_estimators=100 if quick else 300,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1e-3,
            max_iter=200 if quick else 400,
            early_stopping=True,
            random_state=42,
        ),
        "frank_hall_logistic": FrankHallOrdinalClassifier(
            base_estimator=LogisticRegression(
                max_iter=2000, class_weight="balanced", C=1.0
            ),
            n_classes=4,
        ),
        "frank_hall_rf": FrankHallOrdinalClassifier(
            base_estimator=RandomForestClassifier(
                n_estimators=80 if quick else 150,
                max_depth=6,
                random_state=42,
                class_weight="balanced_subsample",
                n_jobs=1,
            ),
            n_classes=4,
        ),
    }
    if LogisticAT is not None:
        models["ordinal_logistic"] = LogisticAT(alpha=1.0)
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=80 if quick else 200,
            max_depth=4,
            learning_rate=0.08,
            objective="multi:softprob",
            num_class=4,
            random_state=42,
            n_jobs=1,
            eval_metric="mlogloss",
        )
    if LGBMClassifier is not None:
        models["lightgbm"] = LGBMClassifier(
            n_estimators=80 if quick else 200,
            random_state=42,
            class_weight="balanced",
            verbose=-1,
        )
    if CatBoostClassifier is not None:
        models["catboost"] = CatBoostClassifier(
            iterations=80 if quick else 200,
            depth=4,
            learning_rate=0.08,
            loss_function="MultiClass",
            verbose=False,
            random_seed=42,
            auto_class_weights="Balanced",
        )
    return models


def prepare_xy(df, feature_cols=None, *, include_observed_indicators: bool = True):
    import pandas as pd

    feature_cols = feature_cols or FEATURE_COLS
    X = df[feature_cols].copy()
    if include_observed_indicators:
        for c in feature_cols:
            X[f"observed_{c}"] = (~X[c].isna()).astype(float)
    y = df["rockburst_grade"].astype(int).to_numpy()
    groups = df["dup_group_id"].to_numpy() if "dup_group_id" in df.columns else np.arange(len(df))
    return X, y, groups


def prepare_xy_sensitivity_variants(df, feature_cols=None):
    """Primary and no-indicator inputs for missingness sensitivity analysis."""
    primary = prepare_xy(df, feature_cols, include_observed_indicators=True)
    no_indicators = prepare_xy(df, feature_cols, include_observed_indicators=False)
    return {"observed_indicators": primary, "no_observed_indicators": no_indicators}
