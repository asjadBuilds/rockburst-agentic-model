"""Extract Table S1 from the supplementary PDF into a clean CSV."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

COLS = [
    "no",
    "sigma_theta",
    "sigma_c",
    "sigma_t",
    "scf",
    "b1",
    "b2",
    "wet",
    "d",
    "rockburst_grade",
]

# Page-break fused row pairs called out in the task breakdown.
FUSED_ROW_PAIRS = [(38, 39), (81, 82), (124, 125), (210, 211), (253, 254), (296, 297), (339, 340)]


def _to_float(tok: str) -> float:
    t = tok.strip().replace(",", "")
    if t.lower() in {"nan", "na", "null", "-", ""}:
        return float("nan")
    return float(t)


def _parse_line(line: str) -> list[float] | None:
    """Parse a data line into [no, 8 features, grade] (10 values)."""
    parts = line.strip().split()
    if len(parts) < 10:
        return None
    if not parts[0].isdigit():
        return None
    try:
        no = int(parts[0])
        if no < 1 or no > 344:
            return None
        vals = [_to_float(x) for x in parts[1:9]]
        grade = int(float(parts[9]))
        if grade not in {0, 1, 2, 3}:
            return None
        return [float(no), *vals, float(grade)]
    except (ValueError, IndexError):
        return None


def extract_table_from_pdf(pdf_path: Path | str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse Table S1 using pdfplumber words grouped by line, with text fallback."""
    import pdfplumber

    pdf_path = Path(pdf_path)
    rows: dict[int, list[float]] = {}
    audit: dict[str, Any] = {
        "source": str(pdf_path),
        "method": "pdfplumber+text",
        "fused_row_pairs_checked": [list(p) for p in FUSED_ROW_PAIRS],
        "warnings": [],
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Stop once we leave Table S1 (PCA tables start around S2)
            text = page.extract_text() or ""
            if "Table S2" in text and rows:
                # Still allow remaining S1 rows that might share the page; only skip pure S2 pages
                if "Rockburst grade" not in text and "σθ" not in text and "sigma" not in text.lower():
                    if re.search(r"Table S[234]", text) and not re.search(
                        r"^\s*\d+\s+[\d\.]", text, re.M
                    ):
                        continue
            for line in text.splitlines():
                parsed = _parse_line(line)
                if parsed is None:
                    continue
                no = int(parsed[0])
                rows[no] = parsed

    missing = [i for i in range(1, 345) if i not in rows]
    if missing:
        # Fallback: pdftotext-style full dump via pdfplumber aggregate
        audit["warnings"].append(f"missing_after_pdfplumber={missing[:20]}")
        full_text = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                full_text.append(page.extract_text() or "")
        for line in "\n".join(full_text).splitlines():
            parsed = _parse_line(line)
            if parsed is None:
                continue
            no = int(parsed[0])
            if no not in rows:
                rows[no] = parsed

    missing = [i for i in range(1, 345) if i not in rows]
    if missing:
        raise ValueError(f"Failed to extract rows: {missing}")

    data = [rows[i] for i in range(1, 345)]
    df = pd.DataFrame(data, columns=COLS)
    df["no"] = df["no"].astype(int)
    df["rockburst_grade"] = df["rockburst_grade"].astype(int)

    # Sanity on fused pairs
    fused_ok = []
    for a, b in FUSED_ROW_PAIRS:
        fused_ok.append({"pair": [a, b], "present": a in rows and b in rows})
    audit["fused_row_status"] = fused_ok
    audit["n_rows"] = len(df)
    audit["d_missing"] = int(df["d"].isna().sum())
    audit["grade_counts"] = df["rockburst_grade"].value_counts().sort_index().to_dict()
    return df, audit


def write_raw_csv(
    df: pd.DataFrame, csv_path: Path | str, audit: dict[str, Any], audit_path: Path | str
) -> None:
    csv_path = Path(csv_path)
    audit_path = Path(audit_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2))
