#!/usr/bin/env python3
"""Reject machine-local absolute paths in client-facing delivery content."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


CLIENT_PATTERNS = (
    "artifacts/baselines/*.json",
    "artifacts/examples/*.json",
    "artifacts/examples/*.md",
    "artifacts/figures/captions.md",
    "data/raw/*.json",
    "notebooks/*.ipynb",
    "reports/*.md",
    "COVER_NOTE.md",
    "DELIVERABLES.md",
    "README.md",
    "README_DELIVERY.txt",
)

LOCAL_PATH_PATTERNS = (
    re.compile(r"/(?:home|Users)/[^\s\"'<>]+"),
    # One slash in plain text, two when embedded as an escaped JSON string.
    re.compile(r"[A-Za-z]:\\+Users\\+[^\s\"'<>]+", re.IGNORECASE),
)


def client_files(root: Path) -> list[Path]:
    selected: set[Path] = set()
    for pattern in CLIENT_PATTERNS:
        selected.update(path for path in root.glob(pattern) if path.is_file())
    return sorted(selected)


def scan_client_content(root: Path) -> list[str]:
    findings: list[str] = []
    for path in client_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for pattern in LOCAL_PATH_PATTERNS:
                match = pattern.search(line)
                if match:
                    relative = path.relative_to(root).as_posix()
                    findings.append(f"{relative}:{line_number}:{match.group(0)}")
                    break
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    findings = scan_client_content(root)
    if findings:
        print("Machine-local absolute paths found in client-facing content:")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"Client-content path scan OK ({len(client_files(root))} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
