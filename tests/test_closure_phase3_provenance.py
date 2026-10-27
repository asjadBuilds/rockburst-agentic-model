from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from rockburst.eval.run_meta import build_run_metadata, finalize_run_metadata
from rockburst.ml.train import _repo_relative_artifact


ROOT = Path(__file__).resolve().parents[1]


def _load_scanner():
    path = ROOT / "scripts" / "scan_client_content.py"
    spec = importlib.util.spec_from_file_location("scan_client_content", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_metadata_uses_relative_input_keys_and_stable_config_hash(tmp_path):
    repo_a = tmp_path / "repo-a"
    repo_b = tmp_path / "repo-b"
    repo_a.mkdir()
    repo_b.mkdir()
    input_a = repo_a / "data" / "processed.csv"
    input_b = repo_b / "data" / "processed.csv"
    input_a.parent.mkdir()
    input_b.parent.mkdir()
    input_a.write_text("x\n1\n", encoding="utf-8")
    input_b.write_text("x\n1\n", encoding="utf-8")
    cfg_a = {"_root": str(repo_a), "seed": 42, "paths": {"processed": "data/processed.csv"}}
    cfg_b = {"_root": str(repo_b), "seed": 42, "paths": {"processed": "data/processed.csv"}}
    meta_a = build_run_metadata(
        mode="full", command="test", config=cfg_a, input_paths=[input_a]
    )
    meta_b = build_run_metadata(
        mode="full", command="test", config=cfg_b, input_paths=[input_b]
    )
    assert meta_a["input_sha256"] == meta_b["input_sha256"]
    assert list(meta_a["input_sha256"]) == ["data/processed.csv"]
    assert meta_a["configuration_sha256"] == meta_b["configuration_sha256"]
    assert isinstance(meta_a["git_dirty"], (bool, type(None)))
    assert "git_hash" in meta_a


def test_training_artifact_label_is_repository_relative(tmp_path):
    model = tmp_path / "artifacts" / "models" / "fusion.joblib"
    model.parent.mkdir(parents=True)
    model.touch()
    assert _repo_relative_artifact(model, {"_root": str(tmp_path)}) == (
        "artifacts/models/fusion.joblib"
    )


def test_completion_metadata_preserves_start_git_state(monkeypatch):
    started = {
        "date_utc": "2026-01-01T00:00:00Z",
        "git_hash": "clean-checkpoint",
        "git_dirty": False,
    }
    monkeypatch.setattr("rockburst.eval.run_meta._git_hash", lambda: "later")
    monkeypatch.setattr("rockburst.eval.run_meta._git_dirty", lambda: True)
    completed = finalize_run_metadata(started, mode="full", command="test")
    assert completed["git_hash"] == "clean-checkpoint"
    assert completed["git_dirty"] is False
    assert completed["started_date_utc"] == "2026-01-01T00:00:00Z"
    assert completed["completed_date_utc"]
    assert completed["git_state_scope"] == "captured before generated artefacts were written"


def test_client_scanner_catches_baseline_notebook_and_report_paths(tmp_path):
    scanner = _load_scanner()
    fixtures = {
        "artifacts/baselines/result.json": '{"input": "/home/alice/project/data.csv"}',
        "notebooks/demo.ipynb": '{"output": "C:\\\\Users\\\\alice\\\\project"}',
        "reports/report.md": "local path: /Users/alice/project/output.json",
    }
    for relative, content in fixtures.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    findings = scanner.scan_client_content(tmp_path)
    assert any(item.startswith("artifacts/baselines/result.json:") for item in findings)
    assert any(item.startswith("notebooks/demo.ipynb:") for item in findings)
    assert any(item.startswith("reports/report.md:") for item in findings)


def test_client_scanner_accepts_repository_relative_content(tmp_path):
    scanner = _load_scanner()
    for relative in (
        "artifacts/baselines/result.json",
        "notebooks/demo.ipynb",
        "reports/report.md",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"path": "artifacts/models/model.joblib"}', encoding="utf-8")
    assert scanner.scan_client_content(tmp_path) == []


def test_notebook_source_and_saved_outputs_have_no_local_home_path():
    notebook_path = ROOT / "notebooks" / "end_to_end_demo.ipynb"
    text = notebook_path.read_text(encoding="utf-8")
    notebook = json.loads(text)
    assert "/home/" not in text
    assert "\\\\Users\\\\" not in text
    assert all(output.get("output_type") != "error" for cell in notebook["cells"] for output in cell.get("outputs", []))


def test_extract_audit_source_is_filename_not_host_path():
    source = (ROOT / "src" / "rockburst" / "data" / "extract.py").read_text(
        encoding="utf-8"
    )
    assert '"source": pdf_path.name' in source
    assert '"source": str(pdf_path)' not in source


def test_package_uses_the_allowlisted_client_content_scanner():
    package = (ROOT / "scripts" / "package_delivery.sh").read_text(encoding="utf-8")
    assert "scripts/scan_client_content.py" in package
    assert "grep -R -n -I '/home/'" not in package
