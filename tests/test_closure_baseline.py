from __future__ import annotations

import copy
import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_closure_baseline.py"
_SPEC = importlib.util.spec_from_file_location("verify_closure_baseline", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
compare_values = _MODULE.compare_values
verify = _MODULE.verify


def test_regeneration_mode_checks_protected_metrics_and_code():
    assert verify(check_artifact_hashes=False) == []


def test_closure_comparison_rejects_changed_headline_metric():
    expected = {"metrics": {"qwk": 0.669951155562134}, "fusion_set": "C"}
    changed = copy.deepcopy(expected)
    changed["metrics"]["qwk"] = 0.60
    differences = compare_values(expected, changed, tolerance=1e-12)
    assert differences
    assert "metrics.qwk" in differences[0]


def test_closure_comparison_rejects_changed_method_identifier():
    differences = compare_values(
        {"fusion_set": "C", "conformal": "marginal_split"},
        {"fusion_set": "A", "conformal": "marginal_split"},
        tolerance=1e-12,
    )
    assert differences == ["fusion_set: expected 'C', got 'A'"]


def test_closure_comparison_allows_only_serialization_scale_float_drift():
    assert compare_values(
        {"qwk": 0.669951155562134},
        {"qwk": 0.6699511555621341},
        tolerance=1e-12,
    ) == []
