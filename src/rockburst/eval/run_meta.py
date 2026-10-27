"""Rm8: run metadata embedded in every produced artefact JSON."""

from __future__ import annotations

import subprocess
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any


_PKG_NAMES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "joblib",
    "optuna",
    "catboost",
    "xgboost",
    "lightgbm",
)


def _git_hash() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return out.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return bool(out.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def _sha256(path: str | Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _repository_root(config: dict[str, Any] | None = None) -> Path:
    configured = (config or {}).get("_root")
    return Path(configured).resolve() if configured else Path(__file__).resolve().parents[3]


def _repository_relative_label(path: str | Path, root: Path) -> str:
    """Return a portable provenance label without serializing a host path."""
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return f"external/{resolved.name}"


def _portable_config(config: dict[str, Any]) -> dict[str, Any]:
    """Exclude runtime-only absolute fields from the stable configuration hash."""
    return {key: value for key, value in config.items() if not str(key).startswith("_")}


def build_run_metadata(
    *,
    mode: str,
    command: str | None = None,
    extra: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    input_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Record how an artefact was produced (Rm8 / Phase 5).

    ``mode`` is typically ``\"full\"`` or ``\"quick\"``.
    """
    from rockburst import __version__ as rockburst_version

    meta: dict[str, Any] = {
        "mode": mode,
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_hash": _git_hash(),
        "git_dirty": _git_dirty(),
        "rockburst_version": rockburst_version,
        "package_versions": {
            name: _pkg_version(name) for name in _PKG_NAMES if _pkg_version(name)
        },
    }
    if command:
        meta["command"] = command
    if config is not None:
        encoded = json.dumps(
            _portable_config(config), sort_keys=True, default=str
        ).encode("utf-8")
        meta["configuration_sha256"] = hashlib.sha256(encoded).hexdigest()
    if input_paths:
        root = _repository_root(config)
        meta["input_sha256"] = {
            _repository_relative_label(path, root): _sha256(path)
            for path in input_paths
        }
    if extra:
        meta.update(extra)
    return meta


def finalize_run_metadata(
    started: dict[str, Any],
    *,
    mode: str,
    command: str,
    extra: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    input_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Refresh completion metadata while preserving start-of-run provenance."""
    completed = build_run_metadata(
        mode=mode,
        command=command,
        extra=extra,
        config=config,
        input_paths=input_paths,
    )
    completed["started_date_utc"] = started.get("date_utc")
    completed["completed_date_utc"] = completed.pop("date_utc")
    completed["git_hash"] = started.get("git_hash")
    completed["git_dirty"] = started.get("git_dirty")
    completed["git_state_scope"] = "captured before generated artefacts were written"
    return completed
