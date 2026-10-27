"""Evaluation metrics including ordinal and calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
)


def ordinal_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true.astype(float) - y_pred.astype(float))))


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    O = confusion_matrix(y_true, y_pred, labels=list(range(n_classes))).astype(float)
    hist_true = np.bincount(y_true, minlength=n_classes).astype(float)
    hist_pred = np.bincount(y_pred, minlength=n_classes).astype(float)
    E = np.outer(hist_true, hist_pred)
    E = E / E.sum() * O.sum()
    W = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)
    den = (W * E).sum()
    if den == 0:
        return 1.0
    return float(1.0 - (W * O).sum() / den)


def expected_calibration_error(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10
) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_diagram_bins(
    y_correct: np.ndarray,
    confidence: np.ndarray,
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Reliability-diagram bins for a binary correctness / confidence series (RC3/F11)."""
    y = np.asarray(y_correct, dtype=float).ravel()
    conf = np.asarray(confidence, dtype=float).ravel()
    bins = np.linspace(0, 1, n_bins + 1)
    diagram: list[dict[str, Any]] = []
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        diagram.append(
            {
                "bin": [float(bins[i]), float(bins[i + 1])],
                "mean_confidence": float(conf[m].mean()),
                "empirical_correct": float(y[m].mean()),
                "n": int(m.sum()),
            }
        )
    return diagram


def ece_binary(y_correct: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> float:
    """ECE when ``confidence`` is already P(correct) (R_final path)."""
    y = np.asarray(y_correct, dtype=int).ravel()
    conf = np.asarray(confidence, dtype=float).ravel()
    # Two-column probs so expected_calibration_error uses max=conf when conf≥0.5;
    # for general conf use direct binning.
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(float(y[m].mean()) - float(conf[m].mean()))
    return float(ece)


def ece_with_ci(
    y_correct: np.ndarray,
    confidence: np.ndarray,
    groups: np.ndarray | None = None,
    *,
    n_bins: int = 10,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Point ECE + grouped bootstrap CI (RC3 reliability hooks for Phase 5/F11)."""
    y = np.asarray(y_correct, dtype=int).ravel()
    conf = np.asarray(confidence, dtype=float).ravel()
    point = ece_binary(y, conf, n_bins=n_bins)
    g = np.asarray(groups) if groups is not None else np.arange(len(y))
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    samples: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        m = np.isin(g, chosen)
        if m.sum() < 5:
            continue
        samples.append(ece_binary(y[m], conf[m], n_bins=n_bins))
    if len(samples) < 10:
        return {"ece": point, "ci95": None, "n_boot": len(samples)}
    return {
        "ece": point,
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "n_boot": len(samples),
        "reliability_diagram": reliability_diagram_bins(y, conf, n_bins=n_bins),
    }


def severe_misclassification_rate(y_true: np.ndarray, y_pred: np.ndarray, threshold: int = 2) -> float:
    return float(np.mean(np.abs(y_true.astype(int) - y_pred.astype(int)) >= threshold))


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray, n_classes: int = 4) -> float:
    y = y_true.astype(int)
    onehot = np.eye(n_classes)[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "class_recall": recall_score(
            y_true, y_pred, average=None, labels=[0, 1, 2, 3], zero_division=0
        ).tolist(),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
        "qwk": quadratic_weighted_kappa(y_true, y_pred),
        "ordinal_mae": ordinal_mae(y_true, y_pred),
        "severe_misclassification_rate": severe_misclassification_rate(y_true, y_pred),
    }
    if probs is not None:
        out["brier"] = multiclass_brier(y_true, probs)
        out["ece"] = expected_calibration_error(y_true, probs)
    return out


def matched_coverage_metrics(
    systems: dict[str, dict[str, Any]],
    *,
    groups: np.ndarray | None = None,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare each selective system with M1 thresholded to its own coverage.

    All arrays remain aligned to the same pooled prediction population. For each
    non-M1 system, its native accepted subset is compared with the equally sized
    highest-confidence M1 subset. Bootstrap deltas resample duplicate groups and
    repeat both selections inside each replicate.
    """
    if not systems:
        return {"n_matched": 0, "systems": {}}
    names = list(systems.keys())
    y0 = np.asarray(systems[names[0]]["y"], dtype=int)
    n = len(y0)
    baseline_name = "M1_maxprob" if "M1_maxprob" in systems else names[0]
    base = systems[baseline_name]
    base_y = np.asarray(base["y"], dtype=int)
    base_pred = np.asarray(base["pred"], dtype=int)
    base_conf = np.asarray(base.get("confidence"), dtype=float)
    if base_conf.shape != (n,):
        raise ValueError("M1 confidence is required for same-coverage thresholding")
    g = np.asarray(groups) if groups is not None else np.arange(n)

    def _group_sample_indices(rng: np.random.Generator) -> np.ndarray:
        uniq = np.unique(g)
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        return np.concatenate([np.flatnonzero(g == group) for group in chosen])

    def _metrics(y: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        return (
            float((y[mask] == pred[mask]).mean()),
            severe_misclassification_rate(y[mask], pred[mask]),
        )

    comparisons: dict[str, Any] = {}
    for offset, name in enumerate(names):
        if name == baseline_name:
            continue
        y = np.asarray(systems[name]["y"], dtype=int)
        pred = np.asarray(systems[name]["pred"], dtype=int)
        accepted = np.asarray(systems[name]["abstain"], dtype=int) == 0
        k = int(accepted.sum())
        if k == 0:
            comparisons[name] = {"n_total": n, "n_accepted": 0, "coverage": 0.0}
            continue
        base_keep = np.zeros(n, dtype=bool)
        base_keep[np.argsort(-base_conf, kind="stable")[:k]] = True
        sys_acc, sys_sev = _metrics(y, pred, accepted)
        m1_acc, m1_sev = _metrics(base_y, base_pred, base_keep)
        acc_delta, sev_delta = sys_acc - m1_acc, sys_sev - m1_sev
        rng = np.random.default_rng(seed + offset)
        acc_boot: list[float] = []
        sev_boot: list[float] = []
        for _ in range(n_boot):
            idx = _group_sample_indices(rng)
            sys_accept_b = accepted[idx]
            kb = int(sys_accept_b.sum())
            if kb < 3:
                continue
            base_keep_b = np.zeros(len(idx), dtype=bool)
            base_keep_b[np.argsort(-base_conf[idx], kind="stable")[:kb]] = True
            sa, ss = _metrics(y[idx], pred[idx], sys_accept_b)
            ba, bs = _metrics(base_y[idx], base_pred[idx], base_keep_b)
            acc_boot.append(sa - ba)
            sev_boot.append(ss - bs)

        def _ci(values: list[float]) -> list[float] | None:
            return (
                [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
                if len(values) >= 10
                else None
            )

        comparisons[name] = {
            "n_total": n,
            "n_accepted": k,
            "coverage": float(k / n),
            "system": {"accuracy": sys_acc, "severe_error_rate": sys_sev},
            "m1_same_coverage": {"accuracy": m1_acc, "severe_error_rate": m1_sev},
            "delta_system_minus_m1": {
                "accuracy": acc_delta,
                "accuracy_ci95": _ci(acc_boot),
                "severe_error_rate": sev_delta,
                "severe_error_rate_ci95": _ci(sev_boot),
            },
        }
    return {
        "n_total": n,
        "baseline": baseline_name,
        "comparisons": comparisons,
        "note": (
            "Per-system same-coverage comparison: each M3–M7 native accepted "
            "subset versus the highest-confidence M1 subset of equal size on "
            "the same pooled predictions; paired duplicate-grouped bootstrap CIs."
        ),
    }


def _bootstrap_ci95_delta_qwk(
    y_true: np.ndarray,
    pred_multi: np.ndarray,
    pred_ord: np.ndarray,
    groups: np.ndarray | None,
    *,
    n_boot: int = 200,
    seed: int = 42,
) -> list[float] | None:
    """Grouped (or row) bootstrap CI for QWK_ordinal − QWK_multiclass (RM10)."""
    y_true = np.asarray(y_true, dtype=int)
    pred_multi = np.asarray(pred_multi, dtype=int)
    pred_ord = np.asarray(pred_ord, dtype=int)
    n = len(y_true)
    if n == 0 or len(pred_multi) != n or len(pred_ord) != n:
        return None
    if groups is None:
        groups = np.arange(n)
    else:
        groups = np.asarray(groups)
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_boot):
        if len(uniq) >= 2:
            sample_g = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.flatnonzero(np.isin(groups, sample_g))
        else:
            idx = rng.choice(n, size=n, replace=True)
        if len(idx) < 5:
            continue
        try:
            d = quadratic_weighted_kappa(y_true[idx], pred_ord[idx]) - quadratic_weighted_kappa(
                y_true[idx], pred_multi[idx]
            )
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(d):
            boots.append(float(d))
    if not boots:
        return None
    arr = np.asarray(boots, dtype=float)
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def capacity_matched_family_table(
    models: dict[str, dict[str, Any]],
    *,
    pairs: list[tuple[str, str]] | None = None,
    oof_pred_by_model: dict[str, np.ndarray] | None = None,
    y_true: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Multiclass vs ordinal family comparison with point metrics (RM10 / F8).

    Pairs share an imputer suffix (e.g. ``rf__median`` vs ``frank_hall_rf__median``).
    When fold OOF prediction vectors are provided, fill ``ci95_delta_qwk`` via
    grouped bootstrap. Does not change model selection / decode adoption.
    """
    pairs = pairs or [
        ("logistic", "frank_hall_logistic"),
        ("logistic", "ordinal_logistic"),
        ("rf", "frank_hall_rf"),
    ]
    oof_pred_by_model = oof_pred_by_model or {}
    rows: list[dict[str, Any]] = []
    for multi, ordinal in pairs:
        for key, meta in models.items():
            if not key.startswith(f"{multi}__"):
                continue
            imp = key.split("__", 1)[1]
            ord_key = f"{ordinal}__{imp}"
            if ord_key not in models:
                continue
            m_po = (meta.get("pooled_oof") or {})
            o_po = (models[ord_key].get("pooled_oof") or {})
            m_q = m_po.get("qwk")
            o_q = o_po.get("qwk")
            ci95: list[float] | None = None
            if (
                y_true is not None
                and key in oof_pred_by_model
                and ord_key in oof_pred_by_model
            ):
                ci95 = _bootstrap_ci95_delta_qwk(
                    y_true,
                    oof_pred_by_model[key],
                    oof_pred_by_model[ord_key],
                    groups,
                    n_boot=n_boot,
                    seed=seed,
                )
            rows.append(
                {
                    "imputer": imp,
                    "multiclass": key,
                    "ordinal": ord_key,
                    "multiclass_qwk": m_q,
                    "ordinal_qwk": o_q,
                    "delta_ordinal_minus_multi": (
                        None
                        if m_q is None or o_q is None
                        else float(o_q) - float(m_q)
                    ),
                    "multiclass_inner_qwk_mean": meta.get("inner_qwk_mean"),
                    "ordinal_inner_qwk_mean": models[ord_key].get("inner_qwk_mean"),
                    "ci95_delta_qwk": ci95,
                }
            )
    return {
        "pairs": rows,
        "protocol": (
            "Capacity-matched: same imputer; multiclass family vs ordinal/Frank–Hall. "
            "ci95_delta_qwk = grouped bootstrap on OOF QWK_ord − QWK_multi when "
            "fold OOF vectors are supplied. Selection still uses best mean inner "
            "QWK (no silent M1 regression)."
        ),
    }
