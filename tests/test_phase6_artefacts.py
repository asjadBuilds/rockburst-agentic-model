"""Phase 6: RC5/RC7–RC9 example/figure guards (cheap; no full generate_*)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rockburst.eval.metrics import capacity_matched_family_table


ROOT = Path(__file__).resolve().parents[1]


def _load_gen():
    import importlib.util

    path = ROOT / "scripts" / "generate_examples.py"
    spec = importlib.util.spec_from_file_location("gen_ex", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_rc7_failure_must_not_be_abstain_helper():
    mod = _load_gen()
    bad = {
        "case_no": 1,
        "true_grade": 3,
        "true_intensity": "strong",
        "predicted_grade": 0,
        "predicted_intensity": "none",
        "abstained": True,
        "recommended_action": "abstain",
        "final_system_confidence": 0.9,
    }
    with pytest.raises(ValueError, match="accepted-wrong"):
        mod._record_failure(ROOT / "artifacts" / "examples", bad, err=3, fold_max_prob=0.8)


def test_rc7_try_record_rejects_same_case_as_abstain():
    """RC7: failure case_no must differ from abstain case_no."""
    mod = _load_gen()
    result = {
        "case_no": 77,
        "true_grade": 3,
        "true_intensity": "strong",
        "predicted_grade": 0,
        "predicted_intensity": "none",
        "abstained": False,
        "recommended_action": "accept",
        "final_system_confidence": 0.9,
    }
    assert (
        mod._try_record_accepted_failure(
            ROOT / "artifacts" / "examples",
            result,
            err=3,
            fold_max_prob=0.9,
            abstain_no=77,
            min_err=2,
        )
        is None
    )


def test_rc7_generator_asserts_distinct_cases():
    src = (ROOT / "scripts" / "generate_examples.py").read_text()
    assert "cases_distinct" in src
    assert "failure_no ≠ abstain_no" in src or "failure_no != abstain_no" in src or (
        "abstain_no" in src and "failure_no" in src and "cases_distinct" in src
    )
    assert "_prune_stale_heldout" in src
    assert "min_err=1" in src  # fallback |ŷ−y|≥1


def test_rc8_corrupt_ood_must_record_perturbation_not_copy():
    """Source guard: generate_corrupt_ood_example must not shutil.copy case_146."""
    src = (ROOT / "scripts" / "generate_examples.py").read_text()
    assert "generate_corrupt_ood_example" in src
    assert "perturbation" in src
    assert "shutil.copy(ex / f\"case_{no}.json\", ex / \"case_corrupt_ood.json\")" not in src
    assert "shutil.copy(ex / f'case_{no}.json', ex / 'case_corrupt_ood.json')" not in src
    assert "_strip_abs_paths" in src


def test_rc8_strip_abs_paths_helper():
    mod = _load_gen()
    abs_trace = str(ROOT / "artifacts" / "traces" / "case_1.json")
    cleaned = mod._strip_abs_paths({"trace_path": abs_trace, "ok": 1})
    assert cleaned["ok"] == 1
    assert "/home/" not in str(cleaned["trace_path"])
    assert "artifacts/traces/case_1.json" in str(cleaned["trace_path"]).replace("\\", "/")


def test_rc8_assess_case_trace_path_relative_source():
    src = (ROOT / "src" / "rockburst" / "graph" / "runner.py").read_text()
    assert "relative_to(ROOT" in src
    assert 'result["trace_path"] = str(out_path.resolve().relative_to(ROOT.resolve()))' in src


def test_rc8_example_schema_fields():
    mod = _load_gen()
    assert mod.example_schema_ok({}) == list(mod.REQUIRED_EXAMPLE_FIELDS)
    ok = {f: 1 for f in mod.REQUIRED_EXAMPLE_FIELDS}
    assert mod.example_schema_ok(ok) == []


def test_rc9_f7_rejects_recall_fabrication():
    import importlib.util

    path = ROOT / "scripts" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("gen_fig", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    with pytest.raises(KeyError, match="confusion_matrix"):
        mod._confusion_matrix_from_m1({"models": {}, "nested_unbiased": {}})
    cm, src = mod._confusion_matrix_from_m1(
        {
            "nested_unbiased": {
                "pooled_oof": {"confusion_matrix": [[1, 0], [0, 1]]}
            }
        }
    )
    assert cm.shape == (2, 2)
    assert "nested" in src


def test_rc9_headline_refuses_quick_ablation():
    import importlib.util

    path = ROOT / "scripts" / "generate_figures.py"
    spec = importlib.util.spec_from_file_location("gen_fig", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    with pytest.raises(ValueError, match="quick"):
        mod.write_headline_snapshot({}, {"best_model": "x"}, {"quick": True})


def test_rc8_archive_script_lists_item4():
    src = (ROOT / "scripts" / "archive_stale_artefacts.py").read_text()
    assert "item4_metrics.json" in src
    assert "slice6_m13_m14.json" in src


def test_rm10_capacity_matched_fills_ci_from_oof():
    """RM10: ci95_delta_qwk filled when fold OOF vectors are supplied."""
    rng = np.random.default_rng(0)
    n = 40
    y = rng.integers(0, 4, size=n)
    pred_m = y.copy()
    pred_m[:8] = (pred_m[:8] + 1) % 4
    pred_o = y.copy()
    pred_o[:5] = (pred_o[:5] + 1) % 4
    groups = np.repeat(np.arange(10), 4)
    models = {
        "rf__median": {
            "pooled_oof": {"qwk": 0.65},
            "inner_qwk_mean": 0.62,
        },
        "frank_hall_rf__median": {
            "pooled_oof": {"qwk": 0.66},
            "inner_qwk_mean": 0.61,
        },
    }
    tab = capacity_matched_family_table(
        models,
        oof_pred_by_model={
            "rf__median": pred_m,
            "frank_hall_rf__median": pred_o,
        },
        y_true=y,
        groups=groups,
        n_boot=30,
        seed=0,
    )
    row = next(r for r in tab["pairs"] if r["multiclass"] == "rf__median")
    assert row["ci95_delta_qwk"] is not None
    assert len(row["ci95_delta_qwk"]) == 2
    # Without OOF, CI stays null (schema still present).
    tab2 = capacity_matched_family_table(models)
    assert tab2["pairs"][0]["ci95_delta_qwk"] is None


def _phase2_examples_regenerated() -> bool:
    """True once USER has run the Phase-2 generate_examples.py (summary marker)."""
    summary = ROOT / "artifacts" / "examples" / "c5_selection_summary.json"
    if not summary.exists():
        return False
    try:
        blob = json.loads(summary.read_text())
    except json.JSONDecodeError:
        return False
    return "cases_distinct" in blob


def test_shipped_failure_case_is_accepted_wrong_not_abstain():
    """RC7: after regen, failure_case must be accepted-wrong (no abstain soft-skip)."""
    if not _phase2_examples_regenerated():
        pytest.skip("pending USER scripts/generate_examples.py (Phase-2 marker)")
    p = ROOT / "artifacts" / "examples" / "failure_case.json"
    if not p.exists():
        pytest.fail("failure_case.json missing after Phase-2 regen")
    blob = json.loads(p.read_text())
    fa = blob.get("failure_analysis") or {}
    if fa.get("kind") == "none_found" or (
        blob.get("note") and "No held-out" in str(blob.get("note"))
    ):
        pytest.skip("generator found no accepted-wrong in last scan")
    # No soft-skip when kind is missing/wrong or system abstained — hard fail.
    assert fa.get("kind") == "accepted_wrong", (
        "RC7: failure_case must have failure_analysis.kind=accepted_wrong "
        f"(got kind={fa.get('kind')!r}); abstain-as-failure is invalid"
    )
    assert fa.get("system_abstained") is False
    assert blob.get("abstained") is not True
    assert blob.get("recommended_action") != "abstain"
    abstain_p = ROOT / "artifacts" / "examples" / "abstain_case.json"
    if abstain_p.exists():
        abstain = json.loads(abstain_p.read_text())
        if abstain.get("case_no") is not None and blob.get("case_no") is not None:
            assert int(abstain["case_no"]) != int(blob["case_no"]), (
                "RC7: failure_case case_no must differ from abstain_case"
            )


def test_shipped_forced_abstain_uses_production_cycles_not_max_loops_2():
    """RC5: after regen, forced_abstain must use Critic-cycle protocol, not max_loops=2."""
    if not _phase2_examples_regenerated():
        pytest.skip("pending USER scripts/generate_examples.py (Phase-2 marker)")
    p = ROOT / "artifacts" / "examples" / "forced_abstain_case.json"
    if not p.exists():
        pytest.fail("forced_abstain_case.json missing after Phase-2 regen")
    blob = json.loads(p.read_text())
    proto = str(blob.get("example_protocol") or "")
    assert "max_loops=2" not in proto.replace(" ", ""), (
        "RC5: forced abstain must not use artificial max_loops=2 protocol"
    )
    assert "max_loops=2" not in json.dumps(blob)
    assert blob.get("termination_reason") is not None, (
        "RC5: forced_abstain_case must record termination_reason"
    )
    assert blob.get("production_max_critic_recheck_cycles") is not None, (
        "RC5: must record production_max_critic_recheck_cycles"
    )
    assert int(blob.get("production_max_loops") or 0) >= 8


def test_shipped_corrupt_ood_schema_and_no_abs_paths():
    """RC8: after regen, corrupt OOD records perturbation, ≠ case_146, no /home/."""
    if not _phase2_examples_regenerated():
        pytest.skip("pending USER scripts/generate_examples.py (Phase-2 marker)")
    p = ROOT / "artifacts" / "examples" / "case_corrupt_ood.json"
    if not p.exists():
        pytest.fail("case_corrupt_ood.json missing after Phase-2 regen")
    text = p.read_text()
    assert "/home/" not in text, "RC8: examples must not embed absolute /home/ paths"
    blob = json.loads(text)
    assert "perturbation" in blob, "RC8: must record perturbation"
    twin = ROOT / "artifacts" / "examples" / "case_146.json"
    if twin.exists():
        assert p.read_bytes() != twin.read_bytes(), (
            "RC8: case_corrupt_ood must not be a byte-copy of case_146"
        )
    missing = [
        f
        for f in ("case_no", "true_grade", "example_protocol", "delivered_planner")
        if f not in blob
    ]
    assert not missing, f"RC8 schema missing fields: {missing}"


def test_shipped_examples_have_no_abs_home_paths():
    """RC8: after regen, no shipped example JSON may contain /home/ absolute paths."""
    if not _phase2_examples_regenerated():
        pytest.skip("pending USER scripts/generate_examples.py (Phase-2 marker)")
    ex = ROOT / "artifacts" / "examples"
    offenders: list[str] = []
    for p in sorted(ex.glob("*.json")):
        if "/home/" in p.read_text():
            offenders.append(p.name)
    assert not offenders, f"RC8 absolute /home/ paths in: {offenders}"
