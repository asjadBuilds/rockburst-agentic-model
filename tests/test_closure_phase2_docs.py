from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_example_references_match_regenerated_selection():
    selection = json.loads(_read("artifacts/examples/c5_selection_summary.json"))
    report = _read("reports/technical_report.md")
    response = _read("reports/feedback_response.md")
    assert selection["failure_no"] == 88
    assert selection["abstain_no"] == 129
    assert selection["failure_abs_error"] == 2
    for text in (report, response):
        assert "case_no=194" not in text
        assert "heldout_failure_case_194" not in text
        assert "case **88**" in text or "`case_no=88`" in text


def test_methodology_names_the_actual_production_conformal_method():
    m1 = json.loads(_read("artifacts/baselines/m1_results.json"))
    methodology = _read("reports/methodology.md")
    assert m1["conformal"]["method"] == "marginal_split"
    assert "production method is **marginal split**" in methodology
    assert "Production prefers **Mondrian**" not in methodology


def test_rules_fsm_description_and_optional_planner_default_agree():
    runner = _read("src/rockburst/graph/runner.py")
    cfg = yaml.safe_load(_read("configs/default.yaml"))
    assert runner.startswith(
        '"""Deterministic rules-FSM coordinator with an optional Gemini routing planner."""'
    )
    fallback = re.search(
        r'get\("planner_model", "([^"]+)"\)', runner
    )
    assert fallback is not None
    assert fallback.group(1) == cfg["gemini"]["planner_model"] == "gemini-3.5-flash"


def test_near_duplicate_comment_and_value_are_consistent():
    config_text = _read("configs/default.yaml")
    cfg = yaml.safe_load(config_text)
    assert cfg["quality"]["near_dup_rel_tol"] == 0.05
    assert "Production 5%" in config_text
    assert "# 1% catches" not in config_text


def test_round3_response_covers_every_finding_and_records_final_figure_inspection():
    response = _read("reports/feedback_response.md")
    for prefix, count in (("TC", 9), ("TM", 10), ("Tm", 8)):
        for index in range(1, count + 1):
            assert f"**{prefix}{index}**" in response
    for figure in ("F4", "F6", "F7", "F10", "F11", "F16"):
        assert re.search(rf"\| \*\*{figure}\*\* \| Regenerated and inspected", response)
    assert "**Open**" not in response


def test_every_required_figure_is_embedded_and_has_png_pdf_and_caption():
    report = _read("reports/technical_report.md")
    captions = _read("artifacts/figures/captions.md")
    stems = [
        "F01_system_architecture",
        "F02_state_transitions",
        "F03_routing_traces",
        "F04_dataset_overview",
        "F05_physics_consistency",
        "F06_near_dup_conflicts",
        "F07_confusion_matrices",
        "F08_model_comparison",
        "F09_ablation_m1_m7",
        "F09b_risk_coverage_areas",
        "F10_risk_coverage",
        "F11_reliability",
        "F12_conformal",
        "F13_robustness",
        "F14_missing_data",
        "F15_explainability",
        "F16_critic",
        "F17_fusion",
    ]
    for stem in stems:
        assert f"../artifacts/figures/{stem}.png" in report
        assert (ROOT / "artifacts" / "figures" / f"{stem}.png").is_file()
        assert (ROOT / "artifacts" / "figures" / f"{stem}.pdf").is_file()
        assert f"**{stem}.**" in captions
