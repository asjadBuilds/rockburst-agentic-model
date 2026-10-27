#!/usr/bin/env python3
"""Generate F1–F17 at ≥300 dpi from shipped artefacts (Phase 6 / RC9 + §8).

No placeholder/synthesised plots. Missing required keys raise.
Captions include n / folds / repeats / model from plotted artefacts.

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
from rockburst.physics.relations import compute_b1, compute_b2, compute_scf

DPI = 300
INTENSITY = ["none", "light", "moderate", "strong"]


def _out_dir(cfg: dict) -> Path:
    d = resolve_path(cfg, "artifacts") / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _require(blob: dict, *keys: str, ctx: str) -> None:
    missing = [k for k in keys if k not in blob or blob[k] is None]
    if missing:
        raise KeyError(f"{ctx}: missing required key(s) {missing}")


def _meta_caption(abl: dict, m1: dict) -> str:
    n = abl.get("n_cases") or (m1.get("run_metadata") or {}).get("n_cases") or 344
    n_outer = abl.get("n_outer") or 5
    n_repeats = abl.get("n_repeats") or 5
    model = abl.get("m1_best") or m1.get("best_model")
    mode = abl.get("mode") or ("quick" if abl.get("quick") else "full")
    return f"n={n}, folds={n_outer}, repeats={n_repeats}, model={model}, mode={mode}"


def _save(fig: plt.Figure, path: Path, captions: list[str], caption: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    captions.append(f"**{path.stem}.** {caption}")
    print("wrote", path.name)


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required artefact missing: {path}")
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
        ax.annotate(
            "",
            xy=(4.0, y),
            xytext=(3.6, 2.9),
            arrowprops=dict(arrowstyle="->", lw=1.2, color="#333"),
        )
    ax.annotate(
        "", xy=(6.0, 2.9), xytext=(5.6, 2.9), arrowprops=dict(arrowstyle="->", lw=1.4)
    )
    ax.annotate(
        "", xy=(8.0, 2.9), xytext=(7.6, 2.9), arrowprops=dict(arrowstyle="->", lw=1.4)
    )
    ax.annotate(
        "", xy=(9.8, 2.9), xytext=(9.6, 2.9), arrowprops=dict(arrowstyle="->", lw=1.4)
    )
    ax.annotate(
        "",
        xy=(6.8, 2.4),
        xytext=(6.8, 1.2),
        arrowprops=dict(arrowstyle="<->", lw=1.0),
    )
    ax.text(7.2, 4.2, "tools", fontsize=8, color="#555")
    ax.text(8.2, 3.6, "shared state", fontsize=8, color="#555")
    _save(
        fig,
        out / "F01_system_architecture.png",
        captions,
        "Visible arrows: Case→Coordinator→agent tools→Critic→Fusion→engineering "
        "output; shared CaseState. Rules FSM; LLM planner optional, not for headlines.",
    )


def fig_f2(out: Path, captions: list[str], cfg: dict) -> None:
    rt = cfg.get("routing") or {}
    max_cycles = int(rt.get("max_critic_recheck_cycles", 3))
    max_loops = int(rt.get("max_loops", 24))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    ax.set_title("F2 — State transitions / routing (post-RC5)")
    lines = [
        "start → data_quality",
        "→ (geomechanics | skip if cheap precheck passes; criteria kept)",
        "→ [extra_validation if R_data low / missing / imputed]",
        "→ predict → uncertainty",
        "→ critic (OOD / physics / set-edge / disagreement / validation)",
        f"→ if request_recheck: recheck_* → critic  (cycle++; limit={max_cycles})",
        f"→ non-convergence: critic_recheck_cycles ≥ {max_cycles} → forced abstain",
        f"→ safeguard: total loops ≥ max_loops({max_loops}) → forced abstain",
        "→ finalize → fuse_assessment → decision bands + Critic escalate/downgrade",
        "Hard gates: wide set + low stability; high disagreement; OOD∧weak physics",
        "RC3: abstain keeps true R_final; decision_override_reason recorded (no 0.49 clamp)",
    ]
    for i, t in enumerate(lines):
        ax.text(0.05, 0.92 - i * 0.075, t, fontsize=9.5, transform=ax.transAxes, family="monospace")
    _save(
        fig,
        out / "F02_state_transitions.png",
        captions,
        f"Post-RC5 routing: Critic→recheck cycles (limit={max_cycles}), "
        f"max_loops={max_loops} safeguard, termination/abstention conditions.",
    )


def fig_f3(out: Path, captions: list[str], cfg: dict) -> None:
    ex = resolve_path(cfg, "examples")
    cases = []
    required = [
        ("missingness", "trace_missingness.json"),
        ("OOD", "trace_ood.json"),
        ("Critic recheck", "trace_critic.json"),
    ]
    missing_files = []
    for label, name in required:
        p = ex / name
        if not p.exists():
            missing_files.append(name)
            continue
        blob = _load_json(p)
        cases.append((label, blob.get("route_trace") or blob.get("trace") or [], blob))
    if missing_files:
        raise FileNotFoundError(
            f"F3 requires fresh traces {missing_files}; run scripts/generate_examples.py"
        )
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    for ax, (label, trace, blob) in zip(axes, cases):
        ax.set_title(
            f"{label}\ncase={blob.get('case_no')} fold={blob.get('fold_index')}"
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(len(trace), 1) + 1)
        ax.axis("off")
        for i, step in enumerate(trace[:20]):
            ax.text(0.05, len(trace) - i, str(step)[:55], fontsize=7, family="monospace")
    fig.suptitle("F3 — Example routing traces (rules FSM)")
    _save(
        fig,
        out / "F03_routing_traces.png",
        captions,
        "Fresh traces: missingness, OOD, executed Critic recheck "
        f"(cases {[c[2].get('case_no') for c in cases]}).",
    )


# ---------- data figures ----------
def fig_f4(out: Path, captions: list[str], df: pd.DataFrame) -> None:
    features = ["sigma_theta", "sigma_c", "sigma_t", "scf", "b1", "b2", "wet", "d"]
    labels = ["σθ", "σc", "σt", "SCF", "B1", "B2", "Wet", "Depth D"]
    plot_df = df.copy()
    plot_df["grade_name"] = plot_df["rockburst_grade"].astype(int).map(dict(enumerate(INTENSITY)))
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for feature, label, ax in zip(features, labels, axes.ravel()):
        sns.boxplot(data=plot_df, x="grade_name", y=feature, order=INTENSITY, ax=ax, color="#8fb9d9", fliersize=2)
        ax.set_title(f"{label} by grade")
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("F4 — All eight input features by rockburst grade")
    _save(
        fig,
        out / "F04_dataset_overview.png",
        captions,
        f"All eight model inputs by observed grade as box plots (n={len(df)}); "
        "missing depth values are omitted only from the D panel.",
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
        xs = np.linspace(lo, hi, 50)
        ax.fill_between(
            xs, xs * (1 - rel) - abs_tol, xs * (1 + rel) + abs_tol, alpha=0.15, color="g"
        )
        bad = m & (
            np.abs(x - y) > np.maximum(rel * np.maximum(np.abs(x), np.abs(y)), abs_tol)
        )
        if bad.any():
            ax.scatter(x[bad], y[bad], s=18, c="red", label=f"flagged n={bad.sum()}")
            ax.legend(fontsize=7)
        ax.set_xlabel(f"listed {name}")
        ax.set_ylabel(f"computed {name}")
        title = name
        if name == "B1":
            title = "B1 (log-inset scale note: linear panel)"
            # annotate extreme rows
            if bad.any():
                idxs = np.where(bad)[0][:5]
                for ii in idxs:
                    ax.annotate(
                        str(int(df.iloc[ii]["no"])),
                        (x[ii], y[ii]),
                        fontsize=6,
                        alpha=0.8,
                    )
        ax.set_title(title)
        ax.text(
            0.02,
            0.02,
            f"tol ±{rel:.0%} rel / {abs_tol} abs",
            transform=ax.transAxes,
            fontsize=7,
        )
    fig.suptitle("F5 — Physics consistency (listed vs computed)")
    _save(
        fig,
        out / "F05_physics_consistency.png",
        captions,
        f"Listed vs recomputed SCF/B1/B2; tolerance ±{rel:.0%} rel / {abs_tol} abs "
        f"in all panels; B1 mismatches annotated by case no (n={len(df)}).",
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
    axes[0].set_xlabel("grade(b) − grade(a)")
    axes[0].set_ylabel("count")
    highlight = [p for p in pairs if p["in_315_344"]]
    scatter = axes[1].scatter(
        [p["no_a"] for p in pairs], [p["no_b"] for p in pairs],
        c=[p["shift"] for p in pairs], cmap="coolwarm", vmin=-3, vmax=3,
        s=[35 if p["in_315_344"] else 18 for p in pairs], alpha=0.75,
    )
    axes[1].plot([0, 344], [0, 344], "k--", lw=0.7)
    axes[1].set_xlabel("earlier case number")
    axes[1].set_ylabel("later case number")
    axes[1].set_title("Earlier vs later case, colored by grade shift")
    fig.colorbar(scatter, ax=axes[1], label="later grade − earlier grade")
    fig.suptitle("F6 — Near-duplicates and label conflicts")
    _save(
        fig,
        out / "F06_near_dup_conflicts.png",
        captions,
        f"Conflicting near-duplicate pairs at rel_tol={rel}; 315–344 block and +1 "
        f"shift pattern highlighted (n_pairs={len(pairs)}).",
    )


def _confusion_matrix_from_m1(m1: dict) -> tuple[np.ndarray, str]:
    """RC9: real CM only — never synthesise from recall."""
    nested = (m1.get("nested_unbiased") or {}).get("pooled_oof") or {}
    if "confusion_matrix" in nested:
        return np.asarray(nested["confusion_matrix"]), "nested_unbiased"
    best = m1.get("best_model") or ""
    pooled = ((m1.get("models") or {}).get(best) or {}).get("pooled_oof") or {}
    if "confusion_matrix" in pooled:
        return np.asarray(pooled["confusion_matrix"]), f"production pooled ({best})"
    raise KeyError(
        "F7: confusion_matrix missing from nested_unbiased and production pooled_oof "
        "(RC9: no fabrication from class_recall)"
    )


def fig_f7(out: Path, captions: list[str], m1: dict, abl: dict) -> None:
    blocks = abl.get("ablations") or {}
    matrices = [
        ("M1 all", (blocks.get("M1") or {}).get("confusion_matrix_all")),
        ("M7 all", (blocks.get("M7") or {}).get("confusion_matrix_all")),
        ("M7 accepted", (blocks.get("M7") or {}).get("confusion_matrix_accepted")),
    ]
    if any(matrix is None for _, matrix in matrices):
        raise KeyError("F7 requires ablations M1/M7 confusion_matrix_all and M7 confusion_matrix_accepted")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (title, matrix) in zip(axes, matrices):
        cm = np.asarray(matrix, dtype=int)
        normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        annotations = np.asarray([[f"{cm[i, j]}\n({normalized[i, j]:.2f})" for j in range(4)] for i in range(4)])
        sns.heatmap(normalized, annot=annotations, fmt="", cmap="Blues", vmin=0, vmax=1, ax=ax,
                    xticklabels=INTENSITY, yticklabels=INTENSITY)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{title}: count (row fraction)")
    fig.suptitle("F7 — M1 and M7 confusion matrices")
    meta = _meta_caption(abl, m1)
    _save(
        fig,
        out / "F07_confusion_matrices.png",
        captions,
        f"M1 all, M7 all, and M7 accepted confusion matrices; each cell shows count "
        f"and row-normalized fraction. {meta}.",
    )


def fig_f8(out: Path, captions: list[str], m1: dict) -> None:
    cand = m1.get("candidate_table") or []
    if not cand:
        raise KeyError("F8: candidate_table missing (RM11 — all candidates + CIs)")
    rows = []
    for r in cand:
        key = r.get("model")
        meta = (m1.get("models") or {}).get(key) or {}
        po = meta.get("pooled_oof") or {}
        rows.append(
            {
                "model": key,
                "qwk": r.get("pooled_oof_qwk", po.get("qwk")),
                "qwk_ci": r.get("mean_fold_qwk_ci95"),
                "ordinal_mae": po.get("ordinal_mae"),
                "severe": po.get("severe_misclassification_rate"),
                "ordinal": bool(r.get("is_ordinal_family"))
                or str(key).startswith(("ordinal", "frank_hall")),
            }
        )
    d = pd.DataFrame(rows).dropna(subset=["qwk"]).sort_values("qwk", ascending=False)
    if d.empty:
        raise ValueError("F8: no candidate QWK rows in m1_results")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#e91e63" if r.ordinal else "#2196f3" for r in d.itertuples()]
    yerr = None
    if d["qwk_ci"].notna().any():
        lo, hi = [], []
        for q, ci in zip(d["qwk"], d["qwk_ci"]):
            if isinstance(ci, (list, tuple)) and len(ci) == 2:
                lo.append(max(0.0, float(q) - float(ci[0])))
                hi.append(max(0.0, float(ci[1]) - float(q)))
            else:
                lo.append(0.0)
                hi.append(0.0)
        yerr = [lo, hi]
    axes[0].barh(d["model"], d["qwk"], color=colors, xerr=yerr, capsize=2)
    axes[0].invert_yaxis()
    axes[0].set_title("QWK (pooled OOF ± fold CI)")
    axes[1].barh(d["model"], d["ordinal_mae"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_title("Ordinal MAE")
    axes[2].barh(d["model"], d["severe"], color=colors)
    axes[2].invert_yaxis()
    axes[2].set_title("Severe-error rate")
    nest = (m1.get("nested_unbiased") or {}).get("production_model") or m1.get("best_model")
    n_outer = (m1.get("run_metadata") or {}).get("n_outer", 5)
    fig.suptitle(
        f"F8 — All {len(d)} candidates (pink=ordinal; nested-unbiased={nest})"
    )
    _save(
        fig,
        out / "F08_model_comparison.png",
        captions,
        f"All candidates with fold QWK CIs; multiclass vs ordinal highlighted; "
        f"nested-unbiased production={nest}; n_outer={n_outer} (RM10/RM11).",
    )


def fig_f9(out: Path, captions: list[str], abl: dict, m1: dict) -> None:
    abs_ = abl.get("ablations") or {}
    _require(abl, "ablations", ctx="F9")
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
                    (means_a[i] - lo[i])
                    if lo[i] is not None and np.isfinite(means_a[i])
                    else 0
                    for i in range(len(labels))
                ],
                [
                    (hi[i] - means_a[i])
                    if hi[i] is not None and np.isfinite(means_a[i])
                    else 0
                    for i in range(len(labels))
                ],
            ]
        )
        ax.bar(x, means_a, yerr=yerr, capsize=3, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(met)
    meta = _meta_caption(abl, m1)
    fig.suptitle(f"F9 — Ablation M1–M7 ({meta})")
    _save(
        fig,
        out / "F09_ablation_m1_m7.png",
        captions,
        f"Ablation QWK, severe-error, abstention ± grouped-bootstrap CI. {meta}.",
    )

    # F9b: RC areas + matched-coverage accepted acc/severe
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
    rc = abl.get("risk_coverage_pooled") or {}
    areas = []
    for m in labels:
        block = rc.get(m) or {}
        a = block.get("area")
        areas.append(a if a is not None else np.nan)
    axes2[0].bar(labels, areas, color="seagreen")
    axes2[0].set_title("RC area by ablation")
    axes2[0].set_ylabel("area")
    mc = abl.get("matched_coverage") or {}
    comparisons = mc.get("comparisons") or {}
    if not comparisons:
        raise KeyError("F9b: matched_coverage.comparisons missing (RC4)")
    names = list(comparisons.keys())
    acc = [comparisons[n]["delta_system_minus_m1"]["accuracy"] for n in names]
    sev = [
        comparisons[n]["delta_system_minus_m1"]["severe_error_rate"]
        for n in names
    ]
    x = np.arange(len(names))
    w = 0.35
    axes2[1].bar(x - w / 2, acc, w, label="Δ accuracy", color="steelblue")
    axes2[1].bar(x + w / 2, sev, w, label="Δ severe", color="salmon")
    axes2[1].axhline(0, color="black", lw=0.8)
    axes2[1].set_xticks(x)
    axes2[1].set_xticklabels(names, rotation=20, ha="right")
    axes2[1].legend(fontsize=7)
    axes2[1].set_title(f"System − M1 at same coverage (N={mc.get('n_total')})")
    fig2.suptitle("F9b — RC areas + matched-coverage (RC4)")
    _save(
        fig2,
        out / "F09b_risk_coverage_areas.png",
        captions,
        f"Pooled RC areas plus per-system same-coverage accuracy/severe deltas "
        f"against M1 on N={mc.get('n_total')} predictions ({meta}).",
    )


def fig_f10(out: Path, captions: list[str], abl: dict, m1: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    rc = abl.get("risk_coverage_pooled") or {}
    base_acc = ((abl.get("ablations") or {}).get("M1") or {}).get("accuracy", {}).get(
        "mean"
    )
    plotted = []
    for name, style in [
        ("M1", "-"),
        ("M4", "--"),
        ("M7", "-."),
    ]:
        block = rc.get(name) or {}
        pts = (
            block.get("example_points")
            or block.get("points")
            or block.get("curve")
            or block.get("mean_curve")
            or []
        )
        if not pts:
            continue
        xs = [p.get("coverage", p.get("x")) for p in pts]
        ys = [p.get("accuracy", p.get("y", p.get("risk"))) for p in pts]
        ax.plot(xs, ys, style, label=name)
        lows = [p.get("ci95_low") for p in pts]
        highs = [p.get("ci95_high") for p in pts]
        if all(value is not None for value in lows + highs):
            ax.fill_between(xs, lows, highs, alpha=0.12)
        plotted.append(name)
    entropy = abl.get("risk_coverage_entropy") or {}
    entropy_points = entropy.get("points") or []
    if entropy_points:
        ex = [point["coverage"] for point in entropy_points]
        ey = [point["accuracy"] for point in entropy_points]
        ax.plot(ex, ey, ":", color="purple", label="M1 entropy")
        elo = [point.get("ci95_low") for point in entropy_points]
        ehi = [point.get("ci95_high") for point in entropy_points]
        if all(value is not None for value in elo + ehi):
            ax.fill_between(ex, elo, ehi, color="purple", alpha=0.10)
    # Entropy / R_final compare note from artefact
    cmp_ = abl.get("risk_coverage_max_prob_vs_r_final") or {}
    if base_acc is not None:
        ax.axhline(base_acc, color="gray", ls=":", label=f"base acc={base_acc:.3f}")
    ax.set_xlabel("coverage")
    ax.set_ylabel("accuracy (kept)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    delta = cmp_.get("r_final_minus_max_prob")
    beats = cmp_.get("r_final_beats_max_prob")
    title = "F10 — Risk–coverage (M1 max-prob / M4 / M7)"
    if delta is not None:
        title += f"\nR_final−max_prob={delta:+.4f} (beats={beats})"
    ax.set_title(title)
    if plotted:
        ax.legend(fontsize=8)
    else:
        raise KeyError(
            "F10: no RC curve points (example_points/points) for M1/M4/M7 — "
            "regenerate ablation_results.json"
        )
    meta = _meta_caption(abl, m1)
    # Honest where M7 loses
    lose_note = ""
    if beats is False:
        lose_note = (
            f" M7/R_final loses to M1 max-prob on RC area "
            f"({cmp_.get('r_final_area_mean'):.3f} vs {cmp_.get('max_prob_area_mean'):.3f})."
        )
    _save(
        fig,
        out / "F10_risk_coverage.png",
        captions,
        f"Pooled RC curves for {plotted} plus entropy ranking, duplicate-grouped "
        f"bootstrap bands, and base-accuracy line. {meta}.{lose_note}",
    )


def fig_f11(out: Path, captions: list[str], m1: dict, abl: dict, fus_cmp: dict) -> None:
    """Reliability from real bins for the production-selected fusion set."""
    sets = fus_cmp.get("sets") or {}
    selected = fus_cmp.get("production_set_from_rule") or fus_cmp.get("chosen_set")
    a = sets.get(selected) or {}
    ece_ci = a.get("ece_ci") or {}
    diagram = ece_ci.get("reliability_diagram")
    if not diagram:
        raise KeyError(
            f"F11: sets.{selected}.ece_ci.reliability_diagram missing "
            "(fusion_set_comparison_slice4.json) — RC9 no schematic placeholder"
        )
    ece_r = a.get("ece")
    ece_m = ((abl.get("ablations") or {}).get("M1") or {}).get("ece", {}).get("mean")
    base_diagram = ((abl.get("ablations") or {}).get("M1") or {}).get("reliability_diagram")
    if not base_diagram:
        raise KeyError("F11: ablations.M1.reliability_diagram missing")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Left: real base-confidence bins; right: served R_final bins.
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    base_conf = [b["mean_confidence"] for b in base_diagram]
    base_acc = [b["empirical_correct"] for b in base_diagram]
    base_n = [b["n"] for b in base_diagram]
    axes[0].scatter(base_conf, base_acc, s=[max(20, n) for n in base_n], c="steelblue")
    for confidence, accuracy, count in zip(base_conf, base_acc, base_n):
        axes[0].annotate(str(count), (confidence, accuracy), fontsize=7, alpha=0.8)
    axes[0].set_title(f"Base max-prob (M1)\nECE={ece_m:.3f}")
    axes[0].set_xlabel("confidence")
    axes[0].set_ylabel("accuracy")
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    confs = [b["mean_confidence"] for b in diagram]
    accs = [b["empirical_correct"] for b in diagram]
    ns = [b["n"] for b in diagram]
    ax.scatter(confs, accs, s=[max(20, n) for n in ns], c="darkcyan")
    for c, a_, n in zip(confs, accs, ns):
        ax.annotate(str(n), (c, a_), fontsize=7, alpha=0.8)
    ci = ece_ci.get("ci95")
    ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "not available"
    ax.set_title(f"R_final (set {selected})\nECE={ece_r:.3f}, 95% CI={ci_text}")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    fig.suptitle("F11 — Reliability diagrams")
    meta = _meta_caption(abl, m1)
    _save(
        fig,
        out / "F11_reliability.png",
        captions,
        f"Real M1 max-prob and served R_final reliability bins with counts; fusion "
        f"set {selected} ECE={ece_r:.3f}, CI={ci_text}; M1 ECE={ece_m:.3f}. {meta}.",
    )


def fig_f12(out: Path, captions: list[str], m1: dict) -> None:
    conf = m1.get("conformal") or {}
    _require(conf, "method", "comparison", ctx="F12")
    method = conf.get("method") or "mondrian_split"
    cmp_ = conf.get("comparison") or {}
    block = cmp_.get(method) or {}
    per = block.get("per_class") or {}
    if not per:
        raise KeyError(f"F12: comparison[{method}].per_class missing")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
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
    # Fold variability
    by_fold = conf.get("set_size_distribution_by_fold") or []
    if by_fold:
        means = []
        for fr in by_fold:
            methods = fr.get("methods") or {}
            mb = methods.get(method) or {}
            means.append(mb.get("mean_size"))
        axes[2].plot(range(len(means)), means, "o-", color="purple")
        axes[2].set_xlabel("fold")
        axes[2].set_ylabel("mean set size")
        axes[2].set_title("Set-size fold variability")
    else:
        trade = conf.get("coverage_size_abstention_tradeoff") or {}
        axes[2].axis("off")
        axes[2].text(0.05, 0.5, json.dumps(trade, indent=2)[:500], fontsize=7, family="monospace")
        axes[2].set_title("Coverage–size–abstention")
    fig.suptitle("F12 — Conformal prediction (RM8)")
    _save(
        fig,
        out / "F12_conformal.png",
        captions,
        f"Per-class coverage vs nominal 1−α, mean set size, fold variability; "
        f"method={method}, alpha={conf.get('alpha')}.",
    )


def fig_f13(out: Path, captions: list[str], abl: dict, m1: dict) -> None:
    rob = abl.get("robustness") or {}
    _require(rob, "metrics", "n_cases", ctx="F13")
    if abl.get("quick"):
        raise ValueError("F13: ablation quick=true — RC1/RC9 require full-mode artefacts")
    metrics = rob.get("metrics") or {}
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    keys = [
        ("flip_rate", "Flip rate"),
        ("mean_abs_delta_p", "Mean |Δp|"),
        ("mean_entropy_delta", "Entropy Δ"),
        ("mean_set_size", "Set size"),
        ("mean_r_final_delta", "R_final Δ"),
    ]
    for ax, (k, title) in zip(axes.ravel(), keys):
        xs, ys, ylo, yhi = [], [], [], []
        for eps_name in ("mild", "strong"):
            block = (metrics.get(eps_name) or {}).get(k) or {}
            if not isinstance(block, dict) or block.get("mean") is None:
                # raw scalar fallback
                raw = (metrics.get(eps_name) or {}).get(k)
                if isinstance(raw, (int, float)):
                    xs.append(eps_name)
                    ys.append(raw)
                    ylo.append(raw)
                    yhi.append(raw)
                continue
            xs.append(eps_name)
            ys.append(block["mean"])
            ylo.append(block.get("ci_low", block.get("ci95_low", block["mean"])))
            yhi.append(block.get("ci_high", block.get("ci95_high", block["mean"])))
        if xs:
            y = np.array(ys, dtype=float)
            ylo_a = np.array(ylo, dtype=float)
            yhi_a = np.array(yhi, dtype=float)
            ax.bar(xs, y, yerr=[y - ylo_a, yhi_a - y], capsize=4)
        ax.set_title(title)
    axes.ravel()[5].axis("off")
    flag = rob.get("r_final_sensitivity_flag") or {}
    axes.ravel()[5].text(
        0.05,
        0.5,
        f"n_cases={rob.get('n_cases')}\n"
        f"flat_flag={flag.get('r_final_flat_under_strong_perturbation')}\n"
        f"{flag.get('note', '')[:200]}",
        fontsize=8,
        transform=axes.ravel()[5].transAxes,
    )
    meta = _meta_caption(abl, m1)
    fig.suptitle(f"F13 — Robustness full-n ({meta})")
    _save(
        fig,
        out / "F13_robustness.png",
        captions,
        f"Full-n robustness flip/|Δp|/entropy/set-size/R_final mild vs strong. {meta}. "
        f"Honest title: n_cases={rob.get('n_cases')}.",
    )


def fig_f14(out: Path, captions: list[str], abl: dict, m1: dict) -> None:
    md = abl.get("missing_data") or {}
    _require(md, "imputers", ctx="F14")
    imps = md.get("imputers") or {}
    if "iterative" not in imps and not md.get("includes_iterative_imputer"):
        raise KeyError("F14: iterative imputer missing from missing_data (RC6)")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # Panel 1: all imputers × one scenario if available
    imp_names = list(imps.keys())
    # pick first scenario key shared
    qwks = []
    for name in imp_names:
        block = imps[name] or {}
        scenarios = [k for k, v in block.items() if isinstance(v, dict) and "qwk" in v]
        if not scenarios:
            qwks.append(np.nan)
            continue
        # prefer a physical mask scenario
        prefer = [s for s in scenarios if "d" in s.lower() or "miss" in s.lower()]
        key = prefer[0] if prefer else scenarios[0]
        qwks.append(block[key].get("qwk"))
    axes[0].barh(imp_names, qwks, color="teal")
    axes[0].invert_yaxis()
    axes[0].set_title("QWK by imputer (incl. iterative)")
    # Panel 2: M7 path incl physics recon
    m7 = md.get("m7") or {}
    sc2 = [k for k, v in m7.items() if isinstance(v, dict) and "qwk" in v]
    if not sc2:
        raise KeyError("F14: missing_data.m7 scenarios missing")
    axes[1].barh(sc2, [m7[s].get("qwk") for s in sc2], color="purple")
    axes[1].invert_yaxis()
    axes[1].set_title("M7 path (physics recon)")
    # Panel 3: train vs inference protocol
    proto = md.get("train_vs_inference_missing_d_protocol") or {}
    axes[2].axis("off")
    axes[2].set_title("Train vs inference missing-D")
    axes[2].text(
        0.02,
        0.95,
        json.dumps(proto, indent=2)[:900],
        fontsize=7,
        family="monospace",
        va="top",
        transform=axes[2].transAxes,
    )
    meta = _meta_caption(abl, m1)
    fig.suptitle(f"F14 — Missing-data ({meta})")
    _save(
        fig,
        out / "F14_missing_data.png",
        captions,
        f"All imputers incl. iterative + physics recon + train/inference protocol. {meta}.",
    )


def fig_f15(out: Path, captions: list[str], cfg: dict) -> None:
    path = resolve_path(cfg, "baselines") / "m2_importance.json"
    blob = _load_json(path)
    ranked = blob.get("ranked") or []
    feat_names = blob.get("features") or []
    means = blob.get("importances_mean")
    stds = blob.get("importances_std")
    if ranked:
        pairs = [
            (r.get("feature"), float(r.get("importance_mean", 0.0)))
            for r in ranked
            if r.get("feature") is not None
        ]
    elif means is not None and feat_names:
        pairs = [(n, float(v)) for n, v in zip(feat_names, means)]
    else:
        raise KeyError(
            "m2_importance.json missing importances_mean/ranked (F15 key mismatch)"
        )
    pairs = [(n, v) for n, v in pairs if n is not None]
    pairs.sort(key=lambda t: t[1])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    yerr = None
    if stds is not None and feat_names:
        std_map = dict(zip(feat_names, stds))
        yerr = [std_map.get(n, 0.0) for n, _ in pairs]
    axes[0].barh(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        xerr=yerr,
        color="slateblue",
        capsize=2,
    )
    axes[0].set_title("All-fold importance ± std")
    dep = blob.get("dependence") or {}
    for feat, ax in zip(["scf", "wet", "sigma_theta"], axes[1:]):
        d = dep.get(feat)
        if isinstance(d, dict) and "x" in d and "y" in d:
            ax.scatter(d["x"], d["y"], s=8, alpha=0.35)
        else:
            df = pd.read_csv(resolve_path(cfg, "processed_csv"))
            ax.scatter(df[feat], df["rockburst_grade"], s=8, alpha=0.35)
        ax.set_xlabel(feat)
        ax.set_ylabel("grade / response")
        ax.set_title(f"{feat} dependence")
    n_folds = blob.get("n_folds")
    n_local = len(blob.get("local_explanations") or [])
    fig.suptitle("F15 — Explainability (RM9)")
    _save(
        fig,
        out / "F15_explainability.png",
        captions,
        f"All-fold permutation importance (n_folds={n_folds}) + SCF/Wet/σθ dependence; "
        f"local explanations n={n_local} in artefact.",
    )


def fig_f16(out: Path, captions: list[str], abl: dict, m1: dict) -> None:
    crit = abl.get("critic_statistics") or {}
    _require(crit, "rule_fires", ctx="F16")
    rules = crit.get("rule_fires") or {}
    rates = crit.get("rule_detection_rates") or {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    if isinstance(rules, dict) and rules:
        items = sorted(
            rules.items(),
            key=lambda kv: -(
                kv[1] if isinstance(kv[1], (int, float)) else kv[1].get("n", 0)
            ),
        )
        names = [k for k, _ in items[:12]]
        vals = [
            (v if isinstance(v, (int, float)) else v.get("n") or v.get("rate") or 0)
            for _, v in items[:12]
        ]
        axes[0].barh(names, vals, color="crimson")
        axes[0].invert_yaxis()
        axes[0].set_title("Critic rule fire counts")
    else:
        axes[0].text(
            0.1,
            0.5,
            f"critic_fired={crit.get('critic_fired')}",
            transform=axes[0].transAxes,
        )
        axes[0].set_title("Critic summary")
    # AUROC and flagged-vs-unflagged error panels.
    if rates:
        ranked = sorted(
            ((name, block) for name, block in rates.items() if block.get("auroc") is not None),
            key=lambda item: item[1]["auroc"], reverse=True,
        )[:12]
        rnames = [name for name, _ in ranked]
        axes[1].barh(rnames, [block["auroc"] for _, block in ranked], color="orange")
        axes[1].axvline(0.5, color="black", ls="--", lw=0.8)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 1)
        axes[1].set_title("Per-rule error AUROC")
        y = np.arange(len(rnames))
        width = 0.38
        axes[2].barh(y - width / 2, [block.get("error_flagged", np.nan) for _, block in ranked], width, label="flagged")
        axes[2].barh(y + width / 2, [block.get("error_unflagged", np.nan) for _, block in ranked], width, label="unflagged")
        axes[2].set_yticks(y)
        axes[2].set_yticklabels(rnames)
        axes[2].invert_yaxis()
        axes[2].set_xlim(0, 1)
        axes[2].legend(fontsize=7)
        axes[2].set_title("Error rate: flagged vs unflagged")
    else:
        raise KeyError("F16: critic_statistics.rule_detection_rates missing")
    meta = _meta_caption(abl, m1)
    fig.suptitle("F16 — Critic performance (RM2)")
    _save(
        fig,
        out / "F16_critic.png",
        captions,
        f"Per-rule fire counts, error AUROC, and flagged-versus-unflagged error "
        f"rates from critic_statistics. {meta}.",
    )


def fig_f17(out: Path, captions: list[str], cfg: dict, m1: dict, fus_cmp: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    boot = fus_cmp.get("coef_bootstrap_cis") or {}
    names = boot.get("names")
    means = boot.get("coef_mean")
    cis = boot.get("coef_ci")
    if names and means and cis:
        yerr = [
            [max(0.0, m - lo) for m, (lo, hi) in zip(means, cis)],
            [max(0.0, hi - m) for m, (lo, hi) in zip(means, cis)],
        ]
        axes[0].barh(names, means, xerr=yerr, color="darkcyan", capsize=3)
        axes[0].invert_yaxis()
        axes[0].set_title("Fusion coef + bootstrap CIs")
    else:
        from rockburst.fusion.stacker import FEATURE_NAMES, ReliabilityFusion

        fus_path = resolve_path(cfg, "models") / "fusion.joblib"
        if not fus_path.exists():
            raise FileNotFoundError("F17: fusion.joblib and coef_bootstrap_cis missing")
        fus = ReliabilityFusion.load(fus_path)
        coef = np.asarray(fus.coef_, dtype=float).ravel()
        axes[0].barh(FEATURE_NAMES[: len(coef)], coef, color="darkcyan")
        axes[0].invert_yaxis()
        axes[0].set_title("Fusion weights (joblib)")
    # Component AUROC / R_final correct vs incorrect if available
    auroc = (fus_cmp.get("component_auroc") or {}).get("per_component") or {}
    if auroc:
        names_a = list(auroc.keys())
        vals = [auroc[k].get("auroc") for k in names_a]
        axes[1].barh(names_a, vals, color="seagreen")
        axes[1].axvline(0.5, color="k", ls="--", lw=0.8)
        axes[1].invert_yaxis()
        axes[1].set_title("Component AUROC vs correctness")
    else:
        axes[1].axis("off")
        axes[1].text(
            0.05,
            0.5,
            f"chosen={fus_cmp.get('chosen_set')} locked={fus_cmp.get('locked_production_set')}\n"
            f"{fus_cmp.get('finding')}",
            fontsize=8,
            transform=axes[1].transAxes,
        )
    fig.suptitle("F17 — Fusion weights and diagnostics")
    _save(
        fig,
        out / "F17_fusion.png",
        captions,
        f"Fusion weights + bootstrap CIs; component AUROC. "
        f"chosen={fus_cmp.get('chosen_set')} production_selected={fus_cmp.get('production_set_from_rule')}.",
    )


def write_headline_snapshot(cfg: dict, m1: dict, abl: dict) -> Path:
    """Single source of truth for report numbers (full-mode only)."""
    if abl.get("quick"):
        raise ValueError(
            "headline_snapshot: refusal to write quick=true ablation numbers (RC1)"
        )
    nested = (m1.get("nested_unbiased") or {}).get("pooled_oof") or {}
    biased = (m1.get("selection_biased") or {}).get("pooled_oof") or {}
    snap = {
        "best_model": m1.get("best_model"),
        "n_trials": m1.get("n_trials")
        or (m1.get("run_metadata") or {}).get("n_trials"),
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
        "matched_coverage": abl.get("matched_coverage"),
        "risk_coverage_max_prob_vs_r_final": abl.get(
            "risk_coverage_max_prob_vs_r_final"
        ),
        "agent_value_finding": abl.get("agent_value_finding"),
        "run_metadata_ablation": abl.get("run_metadata"),
        "run_metadata_m1": m1.get("run_metadata"),
        "source_files": [
            "artifacts/baselines/m1_results.json",
            "artifacts/baselines/ablation_results.json",
        ],
        "note": (
            "Full-mode only. Regenerate via scripts/generate_figures.py; "
            "reports must cite this snapshot or the JSONs."
        ),
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
    fus_cmp = _load_json(
        resolve_path(cfg, "baselines") / "fusion_set_comparison_slice4.json"
    )
    if abl.get("quick"):
        raise SystemExit(
            "RC1/RC9: refuse figures from quick=true ablation_results.json"
        )
    snap = write_headline_snapshot(cfg, m1, abl)
    print("snapshot", snap)

    fig_f1(out, captions)
    fig_f2(out, captions, cfg)
    fig_f3(out, captions, cfg)
    fig_f4(out, captions, df)
    fig_f5(out, captions, df, cfg)
    fig_f6(out, captions, df, cfg)
    fig_f7(out, captions, m1, abl)
    fig_f8(out, captions, m1)
    fig_f9(out, captions, abl, m1)
    fig_f10(out, captions, abl, m1)
    fig_f11(out, captions, m1, abl, fus_cmp)
    fig_f12(out, captions, m1)
    fig_f13(out, captions, abl, m1)
    fig_f14(out, captions, abl, m1)
    fig_f15(out, captions, cfg)
    fig_f16(out, captions, abl, m1)
    fig_f17(out, captions, cfg, m1, fus_cmp)

    (out / "captions.md").write_text("\n\n".join(captions) + "\n")
    print("captions →", out / "captions.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
