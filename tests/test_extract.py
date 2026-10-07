import json
from pathlib import Path

import pandas as pd
import pytest

from rockburst.config import ROOT, load_config, resolve_path
from rockburst.data import extract_table_from_pdf, process_raw


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    cfg = load_config()
    pdf = ROOT / cfg["paths"]["pdf"]
    if not pdf.exists():
        pytest.skip("PDF not present")
    raw = resolve_path(cfg, "raw_csv")
    if raw.exists():
        df = pd.read_csv(raw)
    else:
        df, _ = extract_table_from_pdf(pdf)
    processed_df, audit = process_raw(df, cfg)
    return processed_df, audit


def test_extract_344_rows(processed):
    df, audit = processed
    assert len(df) == 344
    assert set(df["rockburst_grade"].unique()) <= {0, 1, 2, 3}
    assert df["no"].min() == 1 and df["no"].max() == 344
    assert audit["d_missing"] > 50


def test_dup_groups_assigned(processed):
    df, audit = processed
    assert "dup_group_id" in df.columns
    assert audit["n_dup_groups"] < 344
    assert audit["n_groups_with_duplicates"] >= 3


KNOWN_NEAR_DUP_PAIRS = [(6, 315), (14, 316), (8, 322), (1, 327)]


def test_known_near_dup_pairs_same_group(processed):
    df, audit = processed
    by_no = df.set_index("no")
    for a, b in KNOWN_NEAR_DUP_PAIRS:
        assert by_no.loc[a, "dup_group_id"] == by_no.loc[b, "dup_group_id"], (a, b)
    assert audit["n_conflicting_label_groups"] >= 1
    assert any(
        {p["no_a"], p["no_b"]} == {6, 315} for p in audit["conflicting_label_pairs"]
    )


def test_dup_grouping_deterministic(processed):
    cfg = load_config()
    raw = resolve_path(cfg, "raw_csv")
    df = pd.read_csv(raw)
    a, _ = process_raw(df, cfg)
    b, _ = process_raw(df, cfg)
    pd.testing.assert_series_equal(a["dup_group_id"], b["dup_group_id"])
