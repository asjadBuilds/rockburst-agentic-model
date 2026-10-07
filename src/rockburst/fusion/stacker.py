"""Reliability-weighted fusion via constrained non-negative logistic stacker."""

from __future__ import annotations

from typing import Any

import joblib
import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression


FEATURE_NAMES = [
    "r_data",
    "r_physics",
    "r_model",
    "r_uncertainty",
    "r_stability",
]


class FusionNotFitError(RuntimeError):
    """Raised when R_final is requested from an unfitted stacker."""


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def fit_nonneg_logistic(
    R: np.ndarray, y: np.ndarray, *, feature_mask: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Bound-constrained logistic (coef ≥ 0); intercept free. No class_weight."""
    R = np.asarray(R, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = R.shape
    mask = (
        np.ones(d, dtype=bool)
        if feature_mask is None
        else np.asarray(feature_mask, dtype=bool)
    )
    if not mask.any():
        raise ValueError("feature_mask selects no features")
    R_use = R[:, mask]
    d_use = int(mask.sum())

    def nll(theta: np.ndarray) -> float:
        intercept = theta[0]
        coef = theta[1:]
        logits = intercept + R_use @ coef
        p = _sigmoid(logits)
        eps = 1e-12
        return float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    x0 = np.zeros(d_use + 1)
    x0[0] = 0.0
    bounds = [(None, None)] + [(0.0, None)] * d_use
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    intercept = float(res.x[0])
    coef_full = np.zeros(d, dtype=float)
    coef_full[mask] = np.maximum(res.x[1:], 0.0)
    return coef_full, intercept


def collinearity_report(R: np.ndarray, names: list[str] | None = None) -> dict[str, Any]:
    """Correlation and VIF for reliability features (inner-fold diagnostics)."""
    names = names or FEATURE_NAMES
    R = np.asarray(R, dtype=float)
    corr = np.corrcoef(R, rowvar=False)
    # VIF via diagonal of inv(corr) when well-conditioned
    vif: dict[str, float] = {}
    try:
        inv = np.linalg.inv(corr)
        for i, n in enumerate(names):
            vif[n] = float(inv[i, i])
    except np.linalg.LinAlgError:
        for n in names:
            vif[n] = float("inf")
    pair = {
        "r_model_r_uncertainty_corr": float(
            corr[FEATURE_NAMES.index("r_model"), FEATURE_NAMES.index("r_uncertainty")]
        )
    }
    return {"corr": corr.tolist(), "vif": vif, "pair": pair, "names": names}


def choose_feature_mask(
    R: np.ndarray,
    y: np.ndarray,
    *,
    corr_thresh: float = 0.9,
    vif_thresh: float = 5.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Inner-fold: drop redundant r_uncertainty vs r_model; compare drop-one AUCs.

    Pre-registered rule: if |corr|≥0.9 or VIF>5 between r_model and r_uncertainty,
    keep the feature with higher univariate AUC (tie → r_model).
    """
    from sklearn.metrics import roc_auc_score

    R = np.asarray(R, dtype=float)
    y = np.asarray(y, dtype=int)
    d = R.shape[1]
    mask = np.ones(d, dtype=bool)
    report = collinearity_report(R)
    i_m = FEATURE_NAMES.index("r_model")
    i_u = FEATURE_NAMES.index("r_uncertainty")
    corr_mu = abs(report["pair"]["r_model_r_uncertainty_corr"])
    vif_m = report["vif"].get("r_model", 0.0)
    vif_u = report["vif"].get("r_uncertainty", 0.0)
    dropped: list[str] = []
    if corr_mu >= corr_thresh or vif_m > vif_thresh or vif_u > vif_thresh:
        # Univariate AUC
        def _auc(col: int) -> float:
            try:
                return float(roc_auc_score(y, R[:, col]))
            except ValueError:
                return 0.0

        auc_m, auc_u = _auc(i_m), _auc(i_u)
        if auc_u > auc_m:
            mask[i_m] = False
            dropped.append("r_model")
        else:
            mask[i_u] = False
            dropped.append("r_uncertainty")
        report["collinearity_action"] = {
            "dropped": dropped,
            "auc_r_model": auc_m,
            "auc_r_uncertainty": auc_u,
            "reason": f"corr={corr_mu:.3f}, vif_m={vif_m:.2f}, vif_u={vif_u:.2f}",
        }
    else:
        report["collinearity_action"] = {"dropped": [], "reason": "no_redundancy"}

    # Drop-one comparison on constrained fit log-loss (report only; keep mask from collinearity)
    drop_one: dict[str, float] = {}
    full_coef, full_b = fit_nonneg_logistic(R, y, feature_mask=mask)
    p_full = _sigmoid(full_b + R @ full_coef)
    eps = 1e-12
    full_ll = float(
        -np.mean(y * np.log(p_full + eps) + (1 - y) * np.log(1 - p_full + eps))
    )
    drop_one["full"] = full_ll
    for j, name in enumerate(FEATURE_NAMES):
        if not mask[j]:
            continue
        m2 = mask.copy()
        m2[j] = False
        if not m2.any():
            continue
        c2, b2 = fit_nonneg_logistic(R, y, feature_mask=m2)
        p2 = _sigmoid(b2 + R @ c2)
        drop_one[f"drop_{name}"] = float(
            -np.mean(y * np.log(p2 + eps) + (1 - y) * np.log(1 - p2 + eps))
        )
    report["drop_one_logloss"] = drop_one
    report["feature_mask"] = mask.tolist()
    report["zero_weight_flags"] = [
        FEATURE_NAMES[i] for i in range(d) if mask[i] and full_coef[i] <= 1e-8
    ]
    return mask, report


class ReliabilityFusion:
    """Maps reliability vector → soft confidence and decision label.

    Fit uses bound-constrained non-negative logistic regression (no class_weight,
    no post-hoc coefficient clipping). Optional isotonic calibration on inner
    holdout. Unfitted stacker refuses to score (no heuristic weights).
    """

    def __init__(self):
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.is_fit = False
        self.feature_mask: np.ndarray = np.ones(len(FEATURE_NAMES), dtype=bool)
        self.feature_report: dict[str, Any] = {}
        self.calibrator: IsotonicRegression | None = None
        self.decision_thresholds: dict[str, float] = {
            "abstain_r_final": 0.45,
            "engineering_verification": 0.55,
            "accept_with_caution": 0.65,
            "accept_r_final": 0.75,
        }
        # Back-compat for code that still inspects .model.coef_
        self.model = self

    def fit(
        self,
        R: np.ndarray,
        y_correct: np.ndarray,
        *,
        feature_mask: np.ndarray | None = None,
        select_features: bool = True,
    ) -> "ReliabilityFusion":
        """y_correct: 1 if base prediction was correct on OOF, else 0."""
        R = np.asarray(R, dtype=float)
        y = np.asarray(y_correct, dtype=int)
        if len(np.unique(y)) < 2:
            self.is_fit = False
            self.coef_ = None
            return self
        if select_features and feature_mask is None:
            mask, report = choose_feature_mask(R, y)
            self.feature_mask = mask
            self.feature_report = report
        elif feature_mask is not None:
            self.feature_mask = np.asarray(feature_mask, dtype=bool)
        else:
            self.feature_mask = np.ones(R.shape[1], dtype=bool)

        coef, intercept = fit_nonneg_logistic(R, y, feature_mask=self.feature_mask)
        self.coef_ = coef
        self.intercept_ = intercept
        self.is_fit = True
        # Zero-weight flags among active features
        self.feature_report.setdefault(
            "zero_weight_flags",
            [
                FEATURE_NAMES[i]
                for i in range(len(FEATURE_NAMES))
                if self.feature_mask[i] and coef[i] <= 1e-8
            ],
        )
        return self

    def fit_calibrator(self, r_raw: np.ndarray, y_correct: np.ndarray) -> "ReliabilityFusion":
        """Isotonic calibration of raw stacker scores → P(correct), inner data only."""
        r = np.asarray(r_raw, dtype=float)
        y = np.asarray(y_correct, dtype=int)
        if len(np.unique(y)) < 2 or len(r) < 5:
            self.calibrator = None
            return self
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(r, y)
        self.calibrator = iso
        return self

    def _raw_score(self, r_vec: np.ndarray) -> float:
        if not self.is_fit or self.coef_ is None:
            raise FusionNotFitError(
                "ReliabilityFusion is not fitted; refuse heuristic fallback. "
                "Abstain or fit the stacker on OOF live reliabilities."
            )
        r = np.asarray(r_vec, dtype=float).ravel()
        return float(_sigmoid(self.intercept_ + float(r @ self.coef_)))

    def r_final(self, r_vec: np.ndarray) -> float:
        raw = self._raw_score(r_vec)
        if self.calibrator is not None:
            return float(self.calibrator.predict([raw])[0])
        return raw

    def set_decision_thresholds(self, thresholds: dict[str, float]) -> None:
        self.decision_thresholds.update(thresholds)

    def decide(self, r_final: float, cfg: dict[str, Any] | None = None) -> str:
        """Confidence-ascending decision bands (pre-registered).

        abstain < eng_verify < caution <= accept
        """
        thr = dict(self.decision_thresholds)
        if cfg:
            for k in (
                "abstain_r_final",
                "engineering_verification",
                "accept_with_caution",
                "accept_r_final",
            ):
                if k in cfg:
                    thr[k] = float(cfg[k])
        abstain_t = float(thr.get("abstain_r_final", 0.45))
        eng_t = float(thr.get("engineering_verification", 0.55))
        caution_t = float(thr.get("accept_with_caution", 0.65))
        accept_t = float(thr.get("accept_r_final", 0.75))
        # Enforce ordering
        eng_t = max(eng_t, abstain_t + 1e-6)
        caution_t = max(caution_t, eng_t + 1e-6)
        accept_t = max(accept_t, caution_t)
        if r_final < abstain_t:
            return "abstain"
        if r_final < eng_t:
            return "engineering_verification"
        if r_final < accept_t:
            return "accept_with_caution"
        return "accept"

    def save(self, path) -> None:
        joblib.dump(
            {
                "coef_": self.coef_,
                "intercept_": self.intercept_,
                "is_fit": self.is_fit,
                "feature_mask": self.feature_mask,
                "feature_report": self.feature_report,
                "calibrator": self.calibrator,
                "decision_thresholds": self.decision_thresholds,
            },
            path,
        )

    @classmethod
    def load(cls, path) -> "ReliabilityFusion":
        obj = cls()
        data = joblib.load(path)
        # New format
        if "coef_" in data:
            obj.coef_ = data["coef_"]
            obj.intercept_ = float(data.get("intercept_", 0.0))
            obj.is_fit = bool(data.get("is_fit", False))
            obj.feature_mask = np.asarray(
                data.get("feature_mask", np.ones(len(FEATURE_NAMES))), dtype=bool
            )
            obj.feature_report = data.get("feature_report", {})
            obj.calibrator = data.get("calibrator")
            obj.decision_thresholds = data.get(
                "decision_thresholds", obj.decision_thresholds
            )
            return obj
        # Legacy sklearn LogisticRegression blob
        model = data.get("model")
        obj.is_fit = bool(data.get("is_fit", False))
        obj.decision_thresholds = data.get(
            "decision_thresholds", obj.decision_thresholds
        )
        if obj.is_fit and model is not None and hasattr(model, "coef_"):
            obj.coef_ = np.asarray(model.coef_, dtype=float).ravel()
            obj.intercept_ = float(np.asarray(model.intercept_).ravel()[0])
        return obj


def pack_r_vector(
    r_data: float,
    r_physics: float,
    r_model: float,
    r_uncertainty: float,
    r_stability: float,
) -> np.ndarray:
    return np.array(
        [r_data, r_physics, r_model, r_uncertainty, r_stability], dtype=float
    )


def select_decision_thresholds(
    r_final: np.ndarray,
    y_correct: np.ndarray,
    *,
    grid: np.ndarray | None = None,
) -> dict[str, float]:
    """Choose four ordered cutoffs on validation scores only (maximize utility).

    Mapping (pre-registered):
      R < abstain → abstain
      abstain ≤ R < eng → engineering_verification
      eng ≤ R < caution → accept_with_caution
      R ≥ accept → accept
    with abstain < eng < caution ≤ accept.
    """
    r = np.asarray(r_final, dtype=float)
    y = np.asarray(y_correct, dtype=int)
    grid = grid if grid is not None else np.linspace(0.3, 0.9, 13)
    best = {
        "abstain_r_final": 0.45,
        "engineering_verification": 0.55,
        "accept_with_caution": 0.65,
        "accept_r_final": 0.75,
    }
    best_score = -1e18
    for abstain_t in grid:
        for eng_t in grid:
            if eng_t <= abstain_t + 0.05:
                continue
            for caution_t in grid:
                if caution_t <= eng_t + 0.05:
                    continue
                for accept_t in grid:
                    if accept_t < caution_t:
                        continue
                    score = 0.0
                    for ri, yi in zip(r, y):
                        if ri < abstain_t:
                            score += 0.3 if yi == 0 else -0.05
                        elif ri < eng_t:
                            # engineering verification — cautious middle
                            score += 0.4 if yi == 1 else -0.2
                        elif ri < accept_t:
                            score += 0.5 if yi == 1 else -0.3
                        else:
                            score += 1.0 if yi == 1 else -1.0
                    if score > best_score:
                        best_score = score
                        best = {
                            "abstain_r_final": float(abstain_t),
                            "engineering_verification": float(eng_t),
                            "accept_with_caution": float(caution_t),
                            "accept_r_final": float(accept_t),
                        }
    return best


def bootstrap_coef_cis(
    R: np.ndarray,
    y: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 42,
    feature_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Bootstrap intervals for non-negative logistic coefficients."""
    R = np.asarray(R, dtype=float)
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    n = len(y)
    coefs = []
    intercepts = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        c, b = fit_nonneg_logistic(R[idx], y[idx], feature_mask=feature_mask)
        coefs.append(c)
        intercepts.append(b)
    if not coefs:
        return {"coef_mean": None, "coef_ci": None, "intercept_ci": None}
    C = np.asarray(coefs)
    return {
        "coef_mean": C.mean(axis=0).tolist(),
        "coef_ci": np.quantile(C, [0.025, 0.975], axis=0).T.tolist(),
        "intercept_mean": float(np.mean(intercepts)),
        "intercept_ci": [
            float(np.quantile(intercepts, 0.025)),
            float(np.quantile(intercepts, 0.975)),
        ],
        "names": FEATURE_NAMES,
    }
