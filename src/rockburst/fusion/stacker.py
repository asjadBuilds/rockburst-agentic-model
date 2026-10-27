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
    "n_missing",
    "d_observed",
]
# Reliability channels stay non-negative; missingness may reduce trust (free sign).
FREE_SIGN_FEATURES = frozenset({"n_missing", "d_observed"})

# Item 9: locked after A/B/C/D cross-fit on held-out OOF (simplest set tied with best).
CHOSEN_FUSION_SET = "A"
CHOSEN_FUSION_FEATURES = ("r_model",)


def locked_feature_mask() -> np.ndarray:
    """Boolean mask over FEATURE_NAMES for the locked production feature set."""
    return np.array([n in CHOSEN_FUSION_FEATURES for n in FEATURE_NAMES], dtype=bool)


def locked_feature_report() -> dict[str, Any]:
    mask = locked_feature_mask()
    active = [FEATURE_NAMES[i] for i, u in enumerate(mask) if u]
    inactive = [FEATURE_NAMES[i] for i, u in enumerate(mask) if not u]
    return {
        "chosen_set": CHOSEN_FUSION_SET,
        "feature_mask": mask.tolist(),
        "active_features": active,
        "zero_weight_flags": inactive,
        "note": (
            "Locked Item 9: simplest set whose risk-coverage area tied best "
            "(paired CI vs best contains 0); A ≺ B ≺ C ≺ D."
        ),
    }


class FusionNotFitError(RuntimeError):
    """Raised when R_final is requested from an unfitted stacker."""


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def _coef_bounds_for_mask(mask: np.ndarray) -> list[tuple[float | None, float | None]]:
    """Intercept free; R_* ≥ 0; n_missing / d_observed free-signed."""
    bounds: list[tuple[float | None, float | None]] = [(None, None)]
    for i, use in enumerate(mask):
        if not use:
            continue
        name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"f{i}"
        if name in FREE_SIGN_FEATURES:
            bounds.append((None, None))
        else:
            bounds.append((0.0, None))
    return bounds


def fit_nonneg_logistic(
    R: np.ndarray, y: np.ndarray, *, feature_mask: np.ndarray | None = None
) -> tuple[np.ndarray, float]:
    """Bound-constrained logistic; intercept free. Missingness coefs free-signed."""
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
    bounds = _coef_bounds_for_mask(mask)
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    intercept = float(res.x[0])
    coef_full = np.zeros(d, dtype=float)
    coef_use = np.asarray(res.x[1:], dtype=float)
    # Clip only nonneg channels after optimize (numerical safety)
    names_use = [FEATURE_NAMES[i] for i, u in enumerate(mask) if u]
    for j, name in enumerate(names_use):
        if name not in FREE_SIGN_FEATURES:
            coef_use[j] = max(float(coef_use[j]), 0.0)
    coef_full[mask] = coef_use
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
        FEATURE_NAMES[i]
        for i in range(min(d, len(FEATURE_NAMES)))
        if mask[i] and abs(full_coef[i]) <= 1e-8
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
        self.calibrator: Any | None = None
        self.calibration_method: str | None = None
        self.calibration_n: int = 0
        self.interval_members: list[dict[str, Any]] = []
        self.conf_eps: float = 1e-3
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
            # Single-class OOF correctness: still serve a usable confidence
            # (mean correct-rate) rather than leaving the stacker unfitted
            # (which forces 100% abstain in the graph).
            rate = float(np.mean(y)) if len(y) else 0.0
            self.coef_ = np.zeros(R.shape[1], dtype=float)
            i_m = FEATURE_NAMES.index("r_model") if "r_model" in FEATURE_NAMES else 2
            if i_m < len(self.coef_):
                self.coef_[i_m] = 1.0
            # logit(rate) ≈ intercept + mean(r_model); use intercept-only signal
            eps = 1e-6
            rate = float(np.clip(rate, eps, 1 - eps))
            self.intercept_ = float(np.log(rate / (1 - rate)))
            self.coef_[:] = 0.0
            self.is_fit = True
            self.feature_report = {
                "single_class_fallback": True,
                "constant_rate": rate,
            }
            return self
        if feature_mask is not None:
            self.feature_mask = np.asarray(feature_mask, dtype=bool)
            self.feature_report = {"feature_mask": self.feature_mask.tolist()}
        else:
            # Locked Item 9 mask — train / ablation / item4 share this default.
            self.feature_mask = locked_feature_mask()
            self.feature_report = locked_feature_report()
            if select_features:
                self.feature_report["select_features_ignored"] = True

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
                if i < len(coef)
                and self.feature_mask[i]
                and abs(coef[i]) <= 1e-8
            ],
        )
        return self

    def fit_calibrator(
        self,
        r_raw: np.ndarray,
        y_correct: np.ndarray,
        *,
        method: str | None = None,
    ) -> "ReliabilityFusion":
        """Calibrate raw stacker scores → P(correct) on cross-fitted OOF (RC3).

        Method selection (override via ``method``):
          - ``platt``: logistic on raw score (small n / few unique values)
          - ``beta``: logistic on [log s, log(1−s)] (Kull et al.; mid n)
          - ``smoothed_isotonic``: isotonic with y bound away from 0/1

        Prefer calling with the full cross-fitted OOF raw vector (~344), not a
        tiny inner holdout (~70). Scores are clipped to ``conf_eps``…``1-conf_eps``.
        """
        r = np.asarray(r_raw, dtype=float).ravel()
        y = np.asarray(y_correct, dtype=int).ravel()
        self.calibration_n = int(len(r))
        self.calibration_method = None
        self.calibrator = None
        if len(np.unique(y)) < 2 or len(r) < 5:
            return self
        eps = float(self.conf_eps)
        r_clip = np.clip(r, eps, 1.0 - eps)
        n_unique = int(len(np.unique(np.round(r_clip, 6))))
        chosen = method
        if chosen is None:
            # n≈344 is too small for unconstrained isotonic steps. Prefer
            # parametric calibration; isotonic remains an explicit opt-in.
            chosen = "platt" if len(r) < 100 or n_unique < 8 else "beta"
        from sklearn.linear_model import LogisticRegression

        if chosen == "platt":
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(r_clip.reshape(-1, 1), y)
            self.calibrator = lr
            self.calibration_method = "platt"
            return self
        if chosen == "beta":
            # Beta calibration: logit(p) = a log(s) + b log(1-s) + c
            Xb = np.column_stack(
                [np.log(r_clip), np.log(1.0 - r_clip)]
            )
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(Xb, y)
            self.calibrator = ("beta", lr)
            self.calibration_method = "beta"
            return self
        # Smoothed isotonic: bound outputs away from exact 0/1 (reduces plateaus).
        iso = IsotonicRegression(out_of_bounds="clip", y_min=eps, y_max=1.0 - eps)
        iso.fit(r_clip, y)
        self.calibrator = iso
        self.calibration_method = "smoothed_isotonic"
        return self

    def _apply_calibrator(self, raw: float) -> float:
        eps = float(self.conf_eps)
        raw = float(np.clip(raw, eps, 1.0 - eps))
        if self.calibrator is None:
            return raw
        if self.calibration_method == "beta" and isinstance(self.calibrator, tuple):
            _, lr = self.calibrator
            Xb = np.array([[np.log(raw), np.log(1.0 - raw)]], dtype=float)
            return float(lr.predict_proba(Xb)[0, 1])
        if self.calibration_method == "platt" or (
            hasattr(self.calibrator, "predict_proba")
            and not isinstance(self.calibrator, IsotonicRegression)
            and not isinstance(self.calibrator, tuple)
        ):
            return float(self.calibrator.predict_proba([[raw]])[0, 1])
        return float(self.calibrator.predict([raw])[0])

    def _raw_score(self, r_vec: np.ndarray) -> float:
        if not self.is_fit or self.coef_ is None:
            raise FusionNotFitError(
                "ReliabilityFusion is not fitted; refuse heuristic fallback. "
                "Abstain or fit the stacker on OOF live reliabilities."
            )
        r = np.asarray(r_vec, dtype=float).ravel()
        coef = np.asarray(self.coef_, dtype=float).ravel()
        if len(r) != len(coef):
            # Back-compat when FEATURE_NAMES grows (pad missing dims with 0).
            n = max(len(r), len(coef))
            rr = np.zeros(n, dtype=float)
            cc = np.zeros(n, dtype=float)
            rr[: len(r)] = r
            cc[: len(coef)] = coef
            r, coef = rr, cc
        return float(_sigmoid(self.intercept_ + float(r @ coef)))

    def r_final(self, r_vec: np.ndarray) -> float:
        """Calibrated P(correct); always bounded away from exact 0/1 (RC3)."""
        raw = self._raw_score(r_vec)
        eps = float(self.conf_eps)
        val = self._apply_calibrator(raw)
        return float(np.clip(val, eps, 1.0 - eps))

    def r_final_interval(self, r_vec: np.ndarray) -> tuple[float, float, float]:
        """Point estimate plus bootstrap stacker+calibrator interval."""
        point = self.r_final(r_vec)
        if not self.interval_members:
            return point, float("nan"), float("nan")
        values = [self._member_r_final(m, r_vec) for m in self.interval_members]
        return (
            point,
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )

    def _member_r_final(self, member: dict[str, Any], r_vec: np.ndarray) -> float:
        clone_fusion = ReliabilityFusion()
        clone_fusion.coef_ = np.asarray(member["coef_"], dtype=float)
        clone_fusion.intercept_ = float(member["intercept_"])
        clone_fusion.is_fit = True
        clone_fusion.calibrator = member.get("calibrator")
        clone_fusion.calibration_method = member.get("calibration_method")
        clone_fusion.conf_eps = float(self.conf_eps)
        return clone_fusion.r_final(r_vec)

    def fit_bootstrap_interval(
        self,
        R: np.ndarray,
        y_correct: np.ndarray,
        *,
        groups: np.ndarray | None = None,
        n_boot: int = 100,
        seed: int = 42,
        method: str | None = None,
    ) -> "ReliabilityFusion":
        """Bootstrap the complete serving stacker+calibrator pipeline."""
        R = np.asarray(R, dtype=float)
        y = np.asarray(y_correct, dtype=int)
        g = np.asarray(groups) if groups is not None else np.arange(len(y))
        unique = np.unique(g)
        rng = np.random.default_rng(seed)
        members: list[dict[str, Any]] = []
        for _ in range(int(n_boot)):
            chosen = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([np.flatnonzero(g == group) for group in chosen])
            if len(idx) < 5 or len(np.unique(y[idx])) < 2:
                continue
            member = ReliabilityFusion()
            member.conf_eps = float(self.conf_eps)
            member.fit(
                R[idx],
                y[idx],
                feature_mask=self.feature_mask,
                select_features=False,
            )
            raw = np.array([member._raw_score(row) for row in R[idx]])
            member.fit_calibrator(raw, y[idx], method=method)
            members.append(
                {
                    "coef_": np.asarray(member.coef_, dtype=float),
                    "intercept_": float(member.intercept_),
                    "calibrator": member.calibrator,
                    "calibration_method": member.calibration_method,
                }
            )
        self.interval_members = members
        return self

    def set_decision_thresholds(self, thresholds: dict[str, float]) -> None:
        self.decision_thresholds.update(thresholds)

    def decide(self, r_final: float, cfg: dict[str, Any] | None = None) -> str:
        """Map R_final to a decision (confidence-ascending).

        Intended ordering (low → high confidence):
          R < abstain_t                         → abstain
          abstain_t ≤ R < eng_t                 → engineering_verification
          eng_t ≤ R < caution_t                 → accept_with_caution
          R ≥ accept_t                          → accept
        with abstain_t < eng_t < caution_t ≤ accept_t.
        When caution_t < accept_t, scores in [caution_t, accept_t) stay
        accept_with_caution (caution floors accept; label changes at accept_t).
        engineering_verification is NEVER above accept.

        Learned ``decision_thresholds`` are authoritative. Hand-set routing
        keys override only when ``cfg['override_decision_thresholds']`` is true.
        """
        thr = dict(self.decision_thresholds)
        if cfg and bool(cfg.get("override_decision_thresholds")):
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
        # Enforce non-empty bands: abstain < eng < caution < accept (RC3).
        thr_ord = ensure_nonempty_bands(
            {
                "abstain_r_final": abstain_t,
                "engineering_verification": eng_t,
                "accept_with_caution": caution_t,
                "accept_r_final": accept_t,
            }
        )
        abstain_t = thr_ord["abstain_r_final"]
        eng_t = thr_ord["engineering_verification"]
        caution_t = thr_ord["accept_with_caution"]
        accept_t = thr_ord["accept_r_final"]
        if r_final < abstain_t:
            return "abstain"
        if r_final < eng_t:
            return "engineering_verification"
        if r_final < caution_t:
            return "accept_with_caution"
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
                "calibration_method": self.calibration_method,
                "calibration_n": self.calibration_n,
                "interval_members": self.interval_members,
                "conf_eps": self.conf_eps,
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
            obj.calibration_method = data.get("calibration_method")
            obj.calibration_n = int(data.get("calibration_n", 0) or 0)
            obj.interval_members = list(data.get("interval_members") or [])
            obj.conf_eps = float(data.get("conf_eps", obj.conf_eps))
            obj.decision_thresholds = data.get(
                "decision_thresholds", obj.decision_thresholds
            )
            # Infer method for older blobs
            if obj.calibrator is not None and obj.calibration_method is None:
                if isinstance(obj.calibrator, tuple) and obj.calibrator and obj.calibrator[0] == "beta":
                    obj.calibration_method = "beta"
                elif isinstance(obj.calibrator, IsotonicRegression):
                    obj.calibration_method = "smoothed_isotonic"
                elif hasattr(obj.calibrator, "predict_proba"):
                    obj.calibration_method = "platt"
            elif obj.calibration_method == "isotonic":
                obj.calibration_method = "smoothed_isotonic"
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
    n_missing: float = 0.0,
    d_observed: float = 1.0,
) -> np.ndarray:
    return np.array(
        [
            r_data,
            r_physics,
            r_model,
            r_uncertainty,
            r_stability,
            n_missing,
            d_observed,
        ],
        dtype=float,
    )


# Engineering-cost utilities for decision-band selection (RC3).
# Wrong accept (missed severe rockburst / false routine clear) costs more than
# an unnecessary engineering check; abstaining on an error is mildly rewarded;
# abstaining on a correct prediction has small opportunity cost.
DECISION_BAND_UTILITIES: dict[str, float] = {
    "abstain_on_error": 0.3,
    "abstain_on_correct": -0.05,
    "eng_on_correct": 0.4,
    "eng_on_error": -0.2,
    "caution_on_correct": 0.5,
    "caution_on_error": -0.3,
    "pre_accept_on_correct": 0.7,
    "pre_accept_on_error": -0.5,
    "accept_on_correct": 1.0,
    "accept_on_error": -1.0,
}
DECISION_BAND_UTILITIES_JUSTIFICATION = (
    "Asymmetric engineering cost: false accept (routine clear when wrong) is "
    "penalized at −1; correct accept rewarded +1; intermediate bands scale down "
    "both reward and penalty; abstain on error +0.3 (caught miss) vs abstain on "
    "correct −0.05 (extra review). Alternative: target_accept_rate mode below."
)
MIN_BAND_GAP = 0.05


def ensure_nonempty_bands(
    thresholds: dict[str, float],
    *,
    min_gap: float = MIN_BAND_GAP,
) -> dict[str, float]:
    """Force abstain < eng < caution < accept so every label is reachable (RC3).

    If learned cuts collapsed (e.g. caution==accept→empty caution→accept gap),
    push later cuts up by ``min_gap``. Never merges bands silently.
    """
    a = float(thresholds.get("abstain_r_final", 0.45))
    e = float(thresholds.get("engineering_verification", 0.55))
    c = float(thresholds.get("accept_with_caution", 0.65))
    t = float(thresholds.get("accept_r_final", 0.75))
    e = max(e, a + min_gap)
    c = max(c, e + min_gap)
    t = max(t, c + min_gap)
    # Keep inside (0, 1); if overflow, compress from the top.
    if t > 1.0 - 1e-6:
        t = 1.0 - 1e-6
        c = min(c, t - min_gap)
        e = min(e, c - min_gap)
        a = min(a, e - min_gap)
        a = max(a, 1e-6)
        e = max(e, a + min_gap)
        c = max(c, e + min_gap)
        t = max(t, c + min_gap)
    return {
        "abstain_r_final": float(a),
        "engineering_verification": float(e),
        "accept_with_caution": float(c),
        "accept_r_final": float(t),
        "bands_nonempty": True,
        "min_gap": float(min_gap),
    }


def bands_are_nonempty(thresholds: dict[str, float], *, min_gap: float = 1e-6) -> bool:
    a = float(thresholds["abstain_r_final"])
    e = float(thresholds["engineering_verification"])
    c = float(thresholds["accept_with_caution"])
    t = float(thresholds["accept_r_final"])
    return e >= a + min_gap and c >= e + min_gap and t >= c + min_gap


def select_decision_thresholds(
    r_final: np.ndarray,
    y_correct: np.ndarray,
    *,
    grid: np.ndarray | None = None,
    utilities: dict[str, float] | None = None,
    target_accept_rate: float | None = None,
    min_gap: float = MIN_BAND_GAP,
) -> dict[str, float]:
    """Choose four ordered cutoffs on validation scores (RC3).

    Default: maximize ``DECISION_BAND_UTILITIES`` (engineering-cost).
    Optional ``target_accept_rate``: set accept cutoff to the empirical quantile
    so accept mass ≈ target, then place lower bands with ``min_gap`` (coverage
    targeting + sensitivity alternative to hand utilities).

    Always returns non-empty bands (``ensure_nonempty_bands``).
    """
    r = np.asarray(r_final, dtype=float)
    y = np.asarray(y_correct, dtype=int)
    u = {**DECISION_BAND_UTILITIES, **(utilities or {})}

    if target_accept_rate is not None and len(r):
        q = float(np.clip(1.0 - float(target_accept_rate), 0.05, 0.95))
        accept_t = float(np.quantile(r, q))
        caution_t = accept_t - min_gap
        eng_t = caution_t - min_gap
        abstain_t = eng_t - min_gap
        best = ensure_nonempty_bands(
            {
                "abstain_r_final": abstain_t,
                "engineering_verification": eng_t,
                "accept_with_caution": caution_t,
                "accept_r_final": accept_t,
            },
            min_gap=min_gap,
        )
        best["selection_mode"] = "target_accept_rate"
        best["target_accept_rate"] = float(target_accept_rate)
        best["utilities_justification"] = DECISION_BAND_UTILITIES_JUSTIFICATION
        return best

    grid = grid if grid is not None else np.linspace(0.3, 0.9, 13)
    best = ensure_nonempty_bands(
        {
            "abstain_r_final": 0.45,
            "engineering_verification": 0.55,
            "accept_with_caution": 0.65,
            "accept_r_final": 0.75,
        },
        min_gap=min_gap,
    )
    best_score = -1e18
    for abstain_t in grid:
        for eng_t in grid:
            if eng_t < abstain_t + min_gap:
                continue
            for caution_t in grid:
                if caution_t < eng_t + min_gap:
                    continue
                for accept_t in grid:
                    # Strict inequality → non-empty caution→accept band (RC3).
                    if accept_t < caution_t + min_gap:
                        continue
                    score = 0.0
                    for ri, yi in zip(r, y):
                        if ri < abstain_t:
                            score += u["abstain_on_error"] if yi == 0 else u["abstain_on_correct"]
                        elif ri < eng_t:
                            score += u["eng_on_correct"] if yi == 1 else u["eng_on_error"]
                        elif ri < caution_t:
                            score += (
                                u["caution_on_correct"] if yi == 1 else u["caution_on_error"]
                            )
                        elif ri < accept_t:
                            score += (
                                u["pre_accept_on_correct"]
                                if yi == 1
                                else u["pre_accept_on_error"]
                            )
                        else:
                            score += (
                                u["accept_on_correct"] if yi == 1 else u["accept_on_error"]
                            )
                    if score > best_score:
                        best_score = score
                        best = ensure_nonempty_bands(
                            {
                                "abstain_r_final": float(abstain_t),
                                "engineering_verification": float(eng_t),
                                "accept_with_caution": float(caution_t),
                                "accept_r_final": float(accept_t),
                            },
                            min_gap=min_gap,
                        )
    best["selection_mode"] = "engineering_cost_utility"
    best["utility_score"] = float(best_score)
    best["utilities"] = dict(u)
    best["utilities_justification"] = DECISION_BAND_UTILITIES_JUSTIFICATION
    return best


def decision_band_occupancy(
    r_final: np.ndarray, thresholds: dict[str, float]
) -> dict[str, int]:
    """Count served scores in every operational decision band."""
    fusion = ReliabilityFusion()
    fusion.set_decision_thresholds(thresholds)
    labels = ("abstain", "engineering_verification", "accept_with_caution", "accept")
    counts = {label: 0 for label in labels}
    for score in np.asarray(r_final, dtype=float).ravel():
        counts[fusion.decide(float(score))] += 1
    return counts


def decision_utility_sensitivity(
    r_final: np.ndarray, y_correct: np.ndarray
) -> dict[str, dict[str, Any]]:
    """Re-select bands under plausible error-cost changes and report occupancy."""
    scenarios = {
        "lower_false_accept_cost": {"accept_on_error": -0.5},
        "registered": {},
        "higher_false_accept_cost": {"accept_on_error": -2.0},
    }
    out: dict[str, dict[str, Any]] = {}
    for name, overrides in scenarios.items():
        thresholds = select_decision_thresholds(
            r_final, y_correct, utilities=overrides
        )
        out[name] = {
            "thresholds": thresholds,
            "occupancy": decision_band_occupancy(r_final, thresholds),
        }
    return out


def fit_calibrator_on_crossfit_raw(
    fusion: "ReliabilityFusion",
    R: np.ndarray,
    y_correct: np.ndarray,
    fold_of_row: np.ndarray,
    *,
    feature_mask: np.ndarray | None = None,
    conf_eps: float = 1e-3,
    method: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit stackers per fold, collect OOF raw scores, calibrate on all (~344).

    Returns ``(r_raw_oof, r_final_oof)`` aligned to rows (0 where unscored).
    Mutates ``fusion``: attaches calibrator trained on full cross-fitted raw.
    Production stacker coefficients should be fit separately on all OOF.
    """
    R = np.asarray(R, dtype=float)
    y_correct = np.asarray(y_correct, dtype=int)
    fold_of_row = np.asarray(fold_of_row, dtype=int)
    n = len(y_correct)
    r_raw_oof = np.zeros(n, dtype=float)
    scored = np.zeros(n, dtype=bool)
    folds = sorted(int(f) for f in np.unique(fold_of_row) if f >= 0)
    mask = feature_mask if feature_mask is not None else locked_feature_mask()
    for fold in folds:
        train_mask = (fold_of_row >= 0) & (fold_of_row != fold)
        te_mask = fold_of_row == fold
        if not train_mask.any() or not te_mask.any():
            continue
        idx_fit = np.where(train_mask)[0]
        idx_te = np.where(te_mask)[0]
        fold_fus = ReliabilityFusion()
        fold_fus.conf_eps = float(conf_eps)
        fold_fus.fit(
            R[idx_fit], y_correct[idx_fit], feature_mask=mask, select_features=False
        )
        if not fold_fus.is_fit:
            continue
        for j in idx_te:
            r_raw_oof[j] = fold_fus._raw_score(R[j])
            scored[j] = True
    if not scored.any():
        return r_raw_oof, r_raw_oof.copy()
    fusion.conf_eps = float(conf_eps)
    fusion.fit_calibrator(r_raw_oof[scored], y_correct[scored], method=method)
    r_final_oof = np.zeros(n, dtype=float)
    for j in np.where(scored)[0]:
        r_final_oof[j] = float(
            np.clip(fusion._apply_calibrator(r_raw_oof[j]), fusion.conf_eps, 1.0 - fusion.conf_eps)
        )
    return r_raw_oof, r_final_oof


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
