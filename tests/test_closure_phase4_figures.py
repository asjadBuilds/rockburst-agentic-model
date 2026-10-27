from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from rockburst.eval.ablation import _risk_coverage_with_grouped_band


ROOT = Path(__file__).resolve().parents[1]


def _figure_module():
    path = ROOT / "scripts" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("generate_figures_closure", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grouped_risk_coverage_band_preserves_point_curve():
    confidence = np.linspace(0.1, 0.9, 20)
    correct = np.asarray(([0, 1] * 10), dtype=int)
    groups = np.repeat(np.arange(10), 2)
    result = _risk_coverage_with_grouped_band(
        confidence, correct, groups, n_boot=20, seed=42
    )
    assert result["band_method"] == "duplicate-grouped bootstrap"
    assert result["n_boot"] == 20
    assert len(result["points"]) == 10
    assert all(point["ci95_low"] <= point["accuracy"] + 0.5 for point in result["points"])
    assert all(0 <= point["ci95_low"] <= point["ci95_high"] <= 1 for point in result["points"])


def test_f7_renders_m1_m7_all_and_accepted(tmp_path):
    figures = _figure_module()
    matrix = [[5, 1, 0, 0], [1, 4, 1, 0], [0, 1, 4, 1], [0, 0, 1, 5]]
    abl = {
        "n_cases": 24,
        "n_outer": 5,
        "n_repeats": 5,
        "m1_best": "test__median",
        "mode": "full",
        "ablations": {
            "M1": {"confusion_matrix_all": matrix},
            "M7": {
                "confusion_matrix_all": matrix,
                "confusion_matrix_accepted": matrix,
            },
        },
    }
    captions: list[str] = []
    figures.fig_f7(tmp_path, captions, {}, abl)
    assert (tmp_path / "F07_confusion_matrices.png").is_file()
    assert "M7 accepted" in captions[0]


def test_figure_source_contains_no_placeholder_and_requires_real_evidence():
    source = (ROOT / "scripts" / "generate_figures.py").read_text(encoding="utf-8")
    assert "Base bin curve not shipped" not in source
    assert 'raise KeyError("F11: ablations.M1.reliability_diagram missing")' in source
    assert "confusion_matrix_accepted" in source
    assert "risk_coverage_entropy" in source
    assert "Error rate: flagged vs unflagged" in source
