#!/usr/bin/env python3
"""Verify that the approved Phase 1-5 implementation and results did not drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "artifacts" / "baselines" / "round3_closure_baseline.json"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _get(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        current = current[int(part)] if isinstance(current, list) else current[part]
    return current


def compare_values(expected: Any, actual: Any, *, tolerance: float, path: str = "") -> list[str]:
    """Return human-readable differences, allowing only tiny float serialization drift."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path or '<root>'}: expected mapping, got {type(actual).__name__}"]
        differences: list[str] = []
        for key, expected_value in expected.items():
            child = f"{path}.{key}" if path else str(key)
            if key not in actual:
                differences.append(f"{child}: missing")
            else:
                differences.extend(
                    compare_values(expected_value, actual[key], tolerance=tolerance, path=child)
                )
        return differences
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [f"{path}: expected list length {len(expected)}, got {actual!r}"]
        differences = []
        for index, expected_value in enumerate(expected):
            differences.extend(
                compare_values(
                    expected_value,
                    actual[index],
                    tolerance=tolerance,
                    path=f"{path}.{index}",
                )
            )
        return differences
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        if math.isclose(expected, float(actual), rel_tol=tolerance, abs_tol=tolerance):
            return []
    elif expected == actual:
        return []
    return [f"{path}: expected {expected!r}, got {actual!r}"]


def verify(manifest_path: Path = DEFAULT_MANIFEST, root: Path = ROOT) -> list[str]:
    manifest = _load(manifest_path)
    differences: list[str] = []

    for relative, expected_hash in manifest["protected_files_sha256"].items():
        path = root / relative
        if not path.is_file():
            differences.append(f"protected file missing: {relative}")
        elif _sha256(path) != expected_hash:
            differences.append(f"protected file changed: {relative}")

    for relative, expected_hash in manifest["approved_artefacts_sha256"].items():
        path = root / relative
        if not path.is_file():
            differences.append(f"approved artefact missing: {relative}")
        elif _sha256(path) != expected_hash:
            differences.append(f"approved artefact changed: {relative}")

    cache: dict[str, dict[str, Any]] = {}
    actual_values: dict[str, Any] = {}
    for name, specification in manifest["protected_values"].items():
        relative = specification["file"]
        cache.setdefault(relative, _load(root / relative))
        actual_values[name] = _get(cache[relative], specification["path"])
    differences.extend(
        compare_values(
            manifest["expected_values"],
            actual_values,
            tolerance=float(manifest["float_tolerance"]),
        )
    )
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    differences = verify(args.manifest.resolve(), args.root.resolve())
    if differences:
        print("Round-3 closure baseline FAILED:")
        for difference in differences:
            print(f"- {difference}")
        return 1
    print("Round-3 closure baseline OK: protected implementation, artefacts, and metrics unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
