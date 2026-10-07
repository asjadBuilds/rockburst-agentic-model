#!/usr/bin/env python
"""Thin wrappers around rockburst CLI for Makefile-less workflows."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rockburst.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
