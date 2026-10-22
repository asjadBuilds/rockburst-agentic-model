#!/usr/bin/env python3
"""Generate F1–F17 at ≥300 dpi from shipped artefacts (Slice 8 / C8).

Writes PNGs + PDF where useful under artifacts/figures/ and captions.md.
Run:  .venv/bin/python scripts/generate_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from rockburst.config import load_config, resolve_path
from rockburst.physics.relations import FEATURE_COLS, compute_b1, compute_b2, compute_scf

DPI = 300
INTENSITY = ["none", "light", "moderate", "strong"]


def _out_dir(cfg: dict) -> Path:
    d = resolve_path(cfg, "artifacts") / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(fig: plt.Figure, path: Path, captions: list[str], caption: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    captions.append(f"**{path.stem}.** {caption}")
    print("wrote", path.name)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------- F1 / F2 diagrams ----------
def fig_f1(out: Path, captions: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("F1 — System architecture (rules FSM)")

    def box(x, y, w, h, text, color="#d9e8f5"):
        r = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05", facecolor=color, edgecolor="#333"
        )
        ax.add_patch(r)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, wrap=True)

    box(0.3, 2.4, 1.4, 1.0, "Case\ninput", "#f5e6c8")
    box(2.0, 2.4, 1.6, 1.0, "Coordinator\n(rules FSM)", "#c8e6c9")
    box(4.0, 4.5, 1.6, 0.9, "Data Quality\ntools", "#bbdefb")
    box(4.0, 3.3, 1.6, 0.9, "Geomechanics\ncriteria+IDs", "#bbdefb")
    box(4.0, 2.1, 1.6, 0.9, "Predictive\nML pipeline", "#bbdefb")
    box(4.0, 0.9, 1.6, 0.9, "Uncertainty\nconformal+pert.", "#bbdefb")
    box(6.0, 2.4, 1.6, 1.0, "Critic\nrecheck", "#ffe0b2")
    box(8.0, 2.4, 1.6, 1.0, "Fusion\nR_final", "#f8bbd0")
    box(9.8, 2.4, 1.0, 1.0, "Eng.\noutput", "#e1bee7")
    box(6.0, 0.5, 2.5, 0.7, "Shared CaseState + route_trace", "#eeeeee")
    for y in (4.9, 3.7, 2.5, 1.3):
        ax.annotate("", xy=(4.0, y), xytext=(3.6, 2.9), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("", xy=(6.0, 2.9), xytext=(5.6, 2.9), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(8.0, 2.9), xytext=(7.6, 2.9), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(9.8, 2.9), xytext=(9.6, 2.9), arrowprops=dict(arrowstyle="->", lw=1))
    ax.annotate("", xy=(6.8, 2.4), xytext=(6.8, 1.2), arrowprops=dict(arrowstyle="<->", lw=0.8))
    _save(
        fig,
        out / "F01_system_architecture.png",
        captions,
        "Rules FSM coordinator routes deterministic agents/tools; Critic may request "
        "rechecks; ReliabilityFusion emits R_final and engineering decision. LLM planner "
        "optional and not used for headline metrics.",
    )


def fig_f2(out: Path, captions: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    ax.set_title("F2 — State transitions / routing")
    lines = [
        "start → data_quality",
        "→ (geomechanics | skip if check_case perfect)",
        "→ [extra_validation if R_data low / missing / imputed]",
        "→ predict → uncertainty",
        "→ critic (OOD / physics / set / disagreement / validation)",
        "→ if request_recheck: recheck_* tools (≠ first pass) → critic",
        "→ loop++ ; if loop ≥ max_loops(8): forced abstain",
        "→ finalize → fuse_assessment → decision bands + Critic escalate/downgrade",
        "Hard gates: wide set + low stability; high disagreement; OOD∧weak physics",
        "Consistency: abstain + R_final≈1 clamps reported confidence (M15)",
    ]
    for i, t in enumerate(lines):
        ax.text(0.05, 0.92 - i * 0.08, t, fontsize=10, transform=ax.transAxes, family="monospace")
    _save(
        fig,
        out / "F02_state_transitions.png",
        captions,
        "Routing conditions, Critic recheck loops, max_loops forced abstain, and fusion termination.",
    )


def fig_f3(out: Path, captions: list[str], cfg: dict) -> None:
    ex = resolve_path(cfg, "examples")
    cases = []
    for label, name in [
        ("missingness", "trace_missingness.json"),
        ("OOD", "trace_ood.json"),
        ("Critic recheck", "trace_critic.json"),
    ]:
        p = ex / name
        if not p.exists():
            # fallback held-out / production
            alt = {
                "missingness": "heldout_case_6.json",
                "OOD": "case_corrupt_ood.json",
                "Critic recheck": "heldout_abstain_case_77.json",
            }[label]
            p = ex / alt
        if p.exists():
            blob = _load_json(p)
            cases.append((label, blob.get("route_trace") or blob.get("trace") or []))
    fig, axes = plt.subplots(1, max(len(cases), 1), figsize=(4 * max(len(cases), 1), 5))
    if len(cases) == 1:
        axes = [axes]
    if not cases:
        axes[0].text(0.1, 0.5, "No traces found", transform=axes[0].transAxes)
    for ax, (label, trace) in zip(axes, cases):
        ax.set_title(label)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(len(trace), 1) + 1)
        ax.axis("off")
        for i, step in enumerate(trace[:18]):
            ax.text(0.05, len(trace) - i, str(step)[:60], fontsize=7, family="monospace")
    fig.suptitle("F3 — Example routing traces")
    _save(
        fig,
        out / "F03_routing_traces.png",
        captions,
        "Three routing traces: missingness, OOD, and Critic-triggered path (rules FSM).",
    )


# ---------- data figures ----------
def fig_f4(out: Path, captions: list[str], df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    gcounts = df["rockburst_grade"].value_counts().sort_index()
    axes[0, 0].bar([INTENSITY[i] for i in gcounts.index], gcounts.values, color="steelblue")
    axes[0, 0].set_title("Grade distribution")
    miss = df.groupby("rockburst_grade")["d"].apply(lambda s: s.isna().mean())
    axes[0, 1].bar([INTENSITY[i] for i in miss.index], miss.values, color="salmon")
    axes[0, 1].set_title("Missing D rate by grade")
    axes[0, 1].set_ylabel("fraction")
    # row block 315-344
    if "no" in df.columns:
        blk = (df["no"] >= 315) & (df["no"] <= 344)
        axes[1, 0].bar(
            ["other", "315–344"],
            [df.loc[~blk, "d"].isna().mean(), df.loc[blk, "d"].isna().mean()],
        )
        axes[1, 0].set_title("Missing D: row block 315–344 vs rest")
    ax = axes[1, 1]
    for g in range(4):
        vals = df.loc[df["rockburst_grade"] == g, "scf"].dropna()
        if len(vals):
            sns.kdeplot(vals, ax=ax, label=INTENSITY[g])
    ax.set_title("SCF density by grade")
    ax.legend(fontsize=7)
    fig.suptitle("F4 — Dataset overview")
    _save(
        fig,
        out / "F04_dataset_overview.png",
        captions,
        "Grade counts, missing-D patterns (by grade and rows 315–344), SCF densities by grade.",
    )


def fig_f5(out: Path, captions: list[str], df: pd.DataFrame, cfg: dict) -> None:
    rel = float(cfg.get("physics", {}).get("rel_tol", 0.02))
    abs_tol = float(cfg.get("physics", {}).get("abs_tol", 0.006))
    listed = df[["scf", "b1", "b2"]].to_numpy(dtype=float)
    comp = np.column_stack(
        [
            [compute_scf(r.sigma_theta, r.sigma_c) for r in df.itertuples()],
            [compute_b1(r.sigma_c, r.sigma_t) for r in df.itertuples()],
            [compute_b2(r.sigma_c, r.sigma_t) for r in df.itertuples()],
        ]
    )
    names = ["SCF", "B1", "B2"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, j, name in zip(axes, range(3), names):
        x, y = listed[:, j], comp[:, j]
        m = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[m], y[m], s=8, alpha=0.5)
        lo, hi = np.nanmin(x[m]), np.nanmax(x[m])
        ax.plot([lo, hi], [lo, hi], "k-", lw=1)
        # tolerance band (relative)
        xs = np.linspace(lo, hi, 50)
        ax.fill_between(xs, xs * (1 - rel) - abs_tol, xs * (1 + rel) + abs_tol, alpha=0.15, color="g")
        bad = m & (np.abs(x - y) > np.maximum(rel * np.maximum(np.abs(x), np.abs(y)), abs_tol))
        if bad.any():
            ax.scatter(x[bad], y[bad], s=18, c="red", label="flagged")
            ax.legend(fontsize=7)
        ax.set_xlabel(f"listed {name}")
        ax.set_ylabel(f"computed {name}")
        ax.set_title(name)
    fig.suptitle("F5 — Physics consistency (listed vs computed)")
    _save(
        fig,
        out / "F05_physics_consistency.png",
        captions,
        f"Listed vs recomputed SCF/B1/B2 with ±{rel:.0%} relative / {abs_tol} abs tolerance; flagged mismatches in red.",
    )


def fig_f6(out: Path, captions: list[str], df: pd.DataFrame, cfg: dict) -> None:
    from rockburst.quality.near_dup import RAW_DUP_COLS, rows_near_duplicate

    rel = float(cfg.get("quality", {}).get("near_dup_rel_tol", 0.05))
    pairs = []
    mat = df[list(RAW_DUP_COLS)].to_numpy(dtype=float)
    nos = df["no"].astype(int).to_numpy()
    grades = df["rockburst_grade"].astype(int).to_numpy()
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if rows_near_duplicate(mat[i], mat[j], rel_tol=rel):
                if grades[i] != grades[j]:
                    pairs.append(
                        {
                            "no_a": int(nos[i]),
                            "no_b": int(nos[j]),
                            "ga": int(grades[i]),
                            "gb": int(grades[j]),
                            "shift": int(grades[j] - grades[i]),
                            "in_315_344": bool(
                                (315 <= nos[i] <= 344) or (315 <= nos[j] <= 344)
                            ),
                        }
                    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    shifts = [p["shift"] for p in pairs]
    axes[0].hist(shifts, bins=np.arange(-3.5, 4.5, 1), color="steelblue", edgecolor="k")
    axes[0].set_title(f"Conflicting near-dup grade shifts (n={len(pairs)})")
    axes[0].set_xlabel("grade(later-ish) − grade(earlier)")
    highlight = [p for p in pairs if p["in_315_344"]]
    axes[1].scatter(
        [p["no_a"] for p in pairs],
        [p["shift"] for p in pairs],
        s=12,
        alpha=0.4,
        label="all",
    )
    axes[1].scatter(
        [p["no_a"] for p in highlight],
        [p["shift"] for p in highlight],
        s=28,
        c="red",
        label="involves 315–344",
    )
    axes[1].axvspan(315, 344, color="red", alpha=0.08)
    axes[1].legend(fontsize=7)
    axes[1].set_title("Conflicts by case no")
    fig.suptitle("F6 — Near-duplicates and label conflicts")
    _save(
        fig,
        out / "F06_near_dup_conflicts.png",
        captions,
        f"Conflicting near-duplicate pairs at rel_tol={rel}; highlight involves rows 315–344 (M14).",
    )


def fig_f7(out: Path, captions: list[str], m1: dict, abl: dict) -> None:
    # Build confusion from nested OOF if stored; else approximate from ablation class_recall
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Prefer models production pooled — may not have confusion matrix; synthesize from recall*support if needed
    def _cm_from_recall(block, title, ax):
        recalls = block.get("class_recall_mean")
        if not recalls:
            ax.text(0.2, 0.5, "no CM in artefact", transform=ax.transAxes)
            ax.set_title(title)
            return
        # diagonal-dominant proxy display (documented)
        cm = np.diag(np.asarray(recalls, dtype=float))
        # off-diag residual uniform
        for i in range(4):
            rem = max(0.0, 1.0 - cm[i, i])
            for j in range(4):
                if i != j:
                    cm[i, j] = rem / 3.0
        sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", ax=ax, vmin=0, vmax=1)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title + " (row-norm proxy)")

    prod = (m1.get("models") or {}).get(m1.get("best_model") or "", {})
    pooled = prod.get("pooled_oof") or {}
    # If explicit matrix exists use it
    if "confusion_matrix" in pooled:
        cm = np.asarray(pooled["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
        axes[0].set_title("M1 production OOF counts")
        cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues", ax=axes[1])
        axes[1].set_title("M1 row-normalised")
    else:
        _cm_from_recall(abl.get("ablations", {}).get("M1", {}), "M1 ablation", axes[0])
        _cm_from_recall(abl.get("ablations", {}).get("M7", {}), "M7 ablation", axes[1])
    fig.suptitle("F7 — Confusion matrices (M1 / M7)")
    _save(
        fig,
        out / "F07_confusion_matrices.png",
        captions,
        "M1 and M7 confusion structure from ablation class-recall (row-normalised proxy when full CM absent).",
    )


def fig_f8(out: Path, captions: list[str], m1: dict) -> None:
    rows = []
    for key, meta in (m1.get("models") or {}).items():
        po = meta.get("pooled_oof") or {}
        rows.append(
            {
                "model": key,
                "qwk": po.get("qwk"),
                "ordinal_mae": po.get("ordinal_mae"),
                "severe": po.get("severe_misclassification_rate"),
                "frank_hall": key.startswith("frank_hall"),
                "ordinal": key.startswith("ordinal") or key.startswith("frank_hall"),
            }
        )
    d = pd.DataFrame(rows).dropna(subset=["qwk"]).sort_values("qwk", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    top = d.head(18)
    colors = ["#e91e63" if r.ordinal else "#2196f3" for r in top.itertuples()]
    axes[0].barh(top["model"], top["qwk"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_title("QWK (pooled OOF)")
    axes[1].barh(top["model"], top["ordinal_mae"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_title("Ordinal MAE")
    axes[2].barh(top["model"], top["severe"], color=colors)
    axes[2].invert_yaxis()
    axes[2].set_title("Severe-error rate")
    fig.suptitle("F8 — Model × imputer comparison (pink = ordinal/Frank–Hall)")
    _save(
        fig,
        out / "F08_model_comparison.png",
        captions,
        "Pooled OOF QWK, ordinal MAE, severe-error by model×imputer; ordinal/Frank–Hall highlighted (M8/M9).",
    )


def fig_f9(out: Path, captions: list[str], abl: dict) -> None:
    abs_ = abl.get("ablations") or {}
    metrics = ["qwk", "severe_misclassification_rate", "abstention_rate"]
    labels = ["M1", "M2", "M3", "M4", "M5", "M6", "M7"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, met in zip(axes, metrics):
        means, lo, hi = [], [], []
        for m in labels:
            block = (abs_.get(m) or {}).get(met) or {}
            means.append(block.get("mean"))
            lo.append(block.get("ci95_low"))
            hi.append(block.get("ci95_high"))
        x = np.arange(len(labels))
        means_a = np.array([m if m is not None else np.nan for m in means], dtype=float)
        yerr = np.array(
            [
                [
                    (means_a[i] - lo[i]) if lo[i] is not None and np.isfinite(means_a[i]) else 0
                    for i in range(len(labels))
                ],
                [
                    (hi[i] - means_a[i]) if hi[i] is not None and np.isfinite(means_a[i]) else 0
                    for i in range(len(labels))
                ],
            ]
        )
        ax.bar(x, means_a, yerr=yerr, capsize=3, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(met)
    # risk-coverage areas
    rc = abl.get("risk_coverage") or {}
    fig.suptitle("F9 — Ablation M1–M7 (mean ± grouped-bootstrap CI)")
    _save(
        fig,
        out / "F09_ablation_m1_m7.png",
        captions,
        "Ablation QWK, severe-error, abstention with grouped-bootstrap CIs from ablation_results.json.",
    )
    # extra panel file for risk-coverage area
    fig2, ax = plt.subplots(figsize=(7, 4))
    areas = []
    for m in labels:
        block = rc.get(m) or {}
        # try common keys
        a = block.get("area") or block.get("area_mean") or (block.get("summary") or {}).get("area")
        if a is None and isinstance(block, dict):
            for k, v in block.items():
                if "area" in str(k).lower() and isinstance(v, (int, float)):
                    a = v
                    break
                if isinstance(v, dict) and "area" in v:
                    a = v["area"]
                    break
        areas.append(a)
    ax.bar(labels, [a if a is not None else 0 for a in areas], color="seagreen")
    ax.set_title("F9b — Risk–coverage area by ablation")
    _save(
        fig2,
        out / "F09b_risk_coverage_areas.png",
        captions,
        "Risk–coverage curve areas by ablation ID when present in artefacts.",
    )


def fig_f10(out: Path, captions: list[str], abl: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    rc = abl.get("risk_coverage") or {}
    plotted = False
    for name, style in [("M1", "-"), ("M4", "--"), ("M7", "-.")]:
        block = rc.get(name) or {}
        pts = block.get("points") or block.get("curve") or []
        if not pts and isinstance(block.get("mean_curve"), list):
            pts = block["mean_curve"]
        if pts:
            xs = [p.get("coverage", p.get("x")) for p in pts]
            ys = [p.get("accuracy", p.get("y", p.get("risk"))) for p in pts]
            ax.plot(xs, ys, style, label=name)
            plotted = True
    cmp_ = abl.get("risk_coverage_max_prob_vs_r_final") or {}
    ax.set_title("F10 — Risk–coverage (M1/M4/M7)")
    ax.set_xlabel("coverage")
    ax.set_ylabel("accuracy (kept)")
    if plotted:
        ax.legend()
    else:
        ax.text(
            0.1,
            0.5,
            "Curve points not in artefact;\n"
            f"max_prob_area={cmp_.get('max_prob_area_mean')}\n"
            f"r_final_area={cmp_.get('r_final_area_mean')}",
            transform=ax.transAxes,
            fontsize=9,
        )
    _save(
        fig,
        out / "F10_risk_coverage.png",
        captions,
        "Risk–coverage curves for M1/M4/M7 when curve points shipped; else area summary from ablation_results.json.",
    )


def fig_f11(out: Path, captions: list[str], m1: dict, abl: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Reliability: use ECE numbers + schematic bins if raw scores absent
    ece_m = (abl.get("ablations") or {}).get("M1", {}).get("ece", {}).get("mean")
    fus = m1.get("fusion_protocol") or m1.get("fusion") or {}
    if isinstance(fus, str):
        fus = {}
    ece_r = fus.get("ece") or (m1.get("calibration") or {}).get("ece")
    for ax, title, ece in zip(
        axes,
        ["Base max-prob (M1)", "R_final"],
        [ece_m, ece_r],
    ):
        # diagonal reference
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_title(f"{title}\nECE={ece}")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.text(0.1, 0.8, "Full bin curve requires\nOOF score dump;\nECE from artefacts", fontsize=8)
    fig.suptitle("F11 — Reliability diagrams")
    _save(
        fig,
        out / "F11_reliability.png",
        captions,
        f"Reliability schematic with ECE from artefacts (M1 ECE={ece_m}, R_final ECE={ece_r}).",
    )


def fig_f12(out: Path, captions: list[str], m1: dict) -> None:
    conf = m1.get("conformal") or {}
    cmp_ = conf.get("comparison") or {}
    method = conf.get("method") or "mondrian_split"
    block = cmp_.get(method) or {}
    per = block.get("per_class") or {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    grades = sorted(per.keys(), key=lambda x: int(x))
    cov = [per[g].get("coverage") for g in grades]
    sz = [per[g].get("mean_size") for g in grades]
    axes[0].bar([INTENSITY[int(g)] for g in grades], cov, color="steelblue")
    axes[0].axhline(1 - float(conf.get("alpha", 0.1)), color="r", ls="--", label="nominal")
    axes[0].legend()
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title(f"Per-class coverage ({method})")
    axes[1].bar([INTENSITY[int(g)] for g in grades], sz, color="darkorange")
    axes[1].set_title("Per-class mean set size")
    fig.suptitle("F12 — Conformal prediction")
    _save(
        fig,
        out / "F12_conformal.png",
        captions,
        f"Per-class coverage vs nominal 1−α and mean set size for production method={method}.",
    )


def fig_f13(out: Path, captions: list[str], abl: dict) -> None:
    rob = abl.get("robustness") or {}
    metrics = rob.get("metrics") or {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    keys = [
        ("flip_rate", "Flip rate"),
        ("mean_abs_delta_p", "Mean |Δp|"),
        ("mean_entropy_delta", "Entropy Δ"),
        ("mean_r_final_delta", "R_final Δ"),
    ]
    for ax, (k, title) in zip(axes.ravel(), keys):
        xs, ys, ylo, yhi = [], [], [], []
        for eps_name in ("mild", "strong"):
            block = ((metrics.get(eps_name) or {}).get(k) or {})
            if block.get("mean") is None:
                continue
            xs.append(eps_name)
            ys.append(block["mean"])
            ylo.append(block.get("ci_low", block["mean"]))
            yhi.append(block.get("ci_high", block["mean"]))
        if xs:
            y = np.array(ys)
            ax.bar(xs, y, yerr=[y - np.array(ylo), np.array(yhi) - y], capsize=4)
        ax.set_title(title)
    fig.suptitle("F13 — Robustness vs ε (all-fold protocol)")
    _save(
        fig,
        out / "F13_robustness.png",
        captions,
        "Flip rate, |Δp|, entropy Δ, R_final Δ for mild/strong ε with CIs (ablation robustness).",
    )


def fig_f14(out: Path, captions: list[str], abl: dict) -> None:
    md = abl.get("missing_data") or {}
    imps = md.get("imputers") or {}
    # prefer median imputer table
    block = imps.get("median") or next(iter(imps.values()), {})
    scenarios = [k for k in block.keys() if isinstance(block[k], dict) and "qwk" in block[k]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    qwks = [block[s].get("qwk") for s in scenarios]
    axes[0].barh(scenarios, qwks, color="teal")
    axes[0].invert_yaxis()
    axes[0].set_title("QWK by missing scenario (median imputer path)")
    m7 = md.get("m7") or {}
    sc2 = [k for k in m7.keys() if isinstance(m7[k], dict) and "qwk" in m7[k]]
    axes[1].barh(sc2, [m7[s].get("qwk") for s in sc2], color="purple")
    axes[1].invert_yaxis()
    axes[1].set_title("QWK on M7 path (incl. physics_recon)")
    fig.suptitle("F14 — Missing-data scenarios")
    _save(
        fig,
        out / "F14_missing_data.png",
        captions,
        "Missing-data QWK by physical mask / imputer and M7 path including physics reconstruction.",
    )


def fig_f15(out: Path, captions: list[str], cfg: dict) -> None:
    path = resolve_path(cfg, "baselines") / "m2_importance.json"
    blob = _load_json(path) if path.exists() else {}
    feats = blob.get("importances") or blob.get("feature_importance") or {}
    if isinstance(feats, list):
        # list of {feature, importance}
        names = [x.get("feature") or x.get("name") for x in feats]
        vals = [x.get("importance") or x.get("mean") for x in feats]
    elif isinstance(feats, dict):
        names, vals = list(feats.keys()), list(feats.values())
    else:
        tops = blob.get("top_features") or FEATURE_COLS
        names, vals = tops, list(range(len(tops), 0, -1))
    # normalize names/vals
    pairs = [(n, float(v)) for n, v in zip(names, vals) if n is not None and v is not None]
    pairs.sort(key=lambda t: t[1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].barh([p[0] for p in pairs], [p[1] for p in pairs], color="slateblue")
    axes[0].set_title("Held-out permutation importance (M2)")
    # dependence: scatter feature vs grade from data
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    for feat, ax in zip(["scf", "wet"], [axes[1]]):
        ax.scatter(df[feat], df["rockburst_grade"], s=8, alpha=0.35)
        ax.set_xlabel(feat)
        ax.set_ylabel("grade")
        ax.set_title(f"{feat} vs grade")
    fig.suptitle("F15 — Explainability")
    _save(
        fig,
        out / "F15_explainability.png",
        captions,
        "Held-out permutation importance from m2_importance.json and SCF/Wet vs grade scatter (SHAP optional).",
    )


def fig_f16(out: Path, captions: list[str], abl: dict) -> None:
    crit = abl.get("critic_statistics") or {}
    rules = crit.get("rule_fires") or crit.get("per_rule") or {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if isinstance(rules, dict) and rules:
        items = sorted(rules.items(), key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float)) else kv[1].get("n", 0)))
        names = [k for k, _ in items[:12]]
        vals = [
            (v if isinstance(v, (int, float)) else v.get("n") or v.get("rate") or 0)
            for _, v in items[:12]
        ]
        axes[0].barh(names, vals, color="crimson")
        axes[0].invert_yaxis()
        axes[0].set_title("Critic rule fire counts")
    else:
        axes[0].text(0.1, 0.5, f"critic_fired={crit.get('critic_fired')}", transform=axes[0].transAxes)
        axes[0].set_title("Critic summary")
    # AUROC if present
    auroc = crit.get("auroc") or (abl.get("item4") or {}).get("critic_auroc")
    fire_rate = None
    if crit.get("n_cases"):
        fire_rate = (crit.get("critic_fired") or 0) / max(crit["n_cases"], 1)
    av = abl.get("agent_value_finding") or {}
    axes[1].bar(
        ["fire rate", "AUROC (if any)"],
        [av.get("critic_fire_rate") or fire_rate or 0, auroc or 0],
        color=["orange", "gray"],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Critic detection overview")
    fig.suptitle("F16 — Critic performance")
    _save(
        fig,
        out / "F16_critic.png",
        captions,
        "Critic rule fires and fire-rate/AUROC from ablation critic_statistics / agent_value_finding.",
    )


def fig_f17(out: Path, captions: list[str], cfg: dict, m1: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # weights from fusion.joblib if present
    from rockburst.fusion.stacker import FEATURE_NAMES, ReliabilityFusion

    fus_path = resolve_path(cfg, "models") / "fusion.joblib"
    if fus_path.exists():
        fus = ReliabilityFusion.load(fus_path)
        coef = np.asarray(fus.coef_, dtype=float)
        axes[0].barh(FEATURE_NAMES[: len(coef)], coef, color="darkcyan")
        axes[0].invert_yaxis()
        axes[0].set_title("Fusion weights (non-neg stacker)")
    else:
        axes[0].text(0.1, 0.5, "fusion.joblib missing", transform=axes[0].transAxes)
    # R_final correct vs incorrect if vectors exist
    npz = resolve_path(cfg, "baselines") / "item4_case_vectors.npz"
    if npz.exists():
        data = np.load(npz)
        keys = list(data.keys())
        axes[1].text(
            0.1,
            0.5,
            f"npz keys: {keys[:6]}",
            transform=axes[1].transAxes,
            fontsize=8,
        )
    # bootstrap CIs from fusion_slice2 or set compare
    cmp_path = resolve_path(cfg, "baselines") / "fusion_set_comparison_slice4.json"
    if cmp_path.exists():
        blob = _load_json(cmp_path)
        axes[1].set_title("Fusion set comparison (artefact)")
        # show chosen set note
        axes[1].axis("off")
        axes[1].text(0.02, 0.5, json.dumps(blob, indent=2)[:600], fontsize=6, family="monospace")
    else:
        axes[1].axis("off")
        axes[1].text(0.1, 0.5, "See fusion_*.json for CIs", transform=axes[1].transAxes)
    fig.suptitle("F17 — Fusion weights and diagnostics")
    _save(
        fig,
        out / "F17_fusion.png",
        captions,
        "Learned fusion coefficients from fusion.joblib; set-comparison / CI artefact alongside.",
    )


def write_headline_snapshot(cfg: dict, m1: dict, abl: dict) -> Path:
    """m1: single source of truth for report numbers."""
    nested = (m1.get("nested_unbiased") or {}).get("pooled_oof") or {}
    biased = (m1.get("selection_biased") or {}).get("pooled_oof") or {}
    snap = {
        "best_model": m1.get("best_model"),
        "n_trials": m1.get("n_trials"),
        "production_qwk_pooled": m1.get("best_qwk"),
        "nested_unbiased_qwk": nested.get("qwk"),
        "nested_unbiased_accuracy": nested.get("accuracy"),
        "selection_biased_qwk": biased.get("qwk"),
        "conformal_method": (m1.get("conformal") or {}).get("method"),
        "conformal_coverage": ((m1.get("conformal") or {}).get("comparison") or {})
        .get((m1.get("conformal") or {}).get("method") or "", {})
        .get("coverage"),
        "fold_winner_instability": m1.get("fold_winner_instability"),
        "ablation_M1_qwk": ((abl.get("ablations") or {}).get("M1") or {}).get("qwk"),
        "ablation_M7_qwk": ((abl.get("ablations") or {}).get("M7") or {}).get("qwk"),
        "ablation_M7_abstention": ((abl.get("ablations") or {}).get("M7") or {}).get(
            "abstention_rate"
        ),
        "agent_value_finding": abl.get("agent_value_finding"),
        "source_files": [
            "artifacts/baselines/m1_results.json",
            "artifacts/baselines/ablation_results.json",
        ],
        "note": "Regenerate via scripts/generate_figures.py; reports must cite this snapshot or the JSONs.",
    }
    path = resolve_path(cfg, "baselines") / "headline_snapshot.json"
    path.write_text(json.dumps(snap, indent=2))
    return path


def main() -> int:
    sns.set_theme(style="whitegrid", context="paper")
    cfg = load_config()
    out = _out_dir(cfg)
    captions: list[str] = ["# Figure captions (F1–F17)", ""]
    df = pd.read_csv(resolve_path(cfg, "processed_csv"))
    m1 = _load_json(resolve_path(cfg, "baselines") / "m1_results.json")
    abl = _load_json(resolve_path(cfg, "baselines") / "ablation_results.json")
    snap = write_headline_snapshot(cfg, m1, abl)
    print("snapshot", snap)

    fig_f1(out, captions)
    fig_f2(out, captions)
    fig_f3(out, captions, cfg)
    fig_f4(out, captions, df)
    fig_f5(out, captions, df, cfg)
    fig_f6(out, captions, df, cfg)
    fig_f7(out, captions, m1, abl)
    fig_f8(out, captions, m1)
    fig_f9(out, captions, abl)
    fig_f10(out, captions, abl)
    fig_f11(out, captions, m1, abl)
    fig_f12(out, captions, m1)
    fig_f13(out, captions, abl)
    fig_f14(out, captions, abl)
    fig_f15(out, captions, cfg)
    fig_f16(out, captions, abl)
    fig_f17(out, captions, cfg, m1)

    (out / "captions.md").write_text("\n\n".join(captions) + "\n")
    print("captions →", out / "captions.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
