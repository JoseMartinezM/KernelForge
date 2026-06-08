from __future__ import annotations

from pathlib import Path

import pytest

from kernelforge.benchmark.tritonbench import (
    TB_SEPARATOR,
    load_t_datasets,
    load_t_simple_entries,
    match_simple_alpaca_entries,
    prepare_t_entry,
)


def test_load_t_datasets_returns_paired_metadata(tritonbench_root: Path):
    simple_alpaca, t_json = load_t_datasets(tritonbench_root)

    assert len(simple_alpaca) == len(t_json)
    assert len(simple_alpaca) == 166
    assert all("instruction" in row for row in simple_alpaca)
    assert all("file" in row for row in t_json)


def test_match_simple_alpaca_entries_has_no_errors(tritonbench_root: Path):
    simple_alpaca, t_json = load_t_datasets(tritonbench_root)
    entries, errors = match_simple_alpaca_entries(simple_alpaca, t_json)

    assert not errors
    assert len(entries) == 166
    assert {entry["file"] for entry in entries} == {entry["file"] for entry in t_json}


def test_load_t_simple_entries_prepares_reference_and_test_code(tritonbench_root: Path):
    entries, errors, _, _ = load_t_simple_entries(tritonbench_root)

    assert not errors
    assert len(entries) == 166

    tanh = next(entry for entry in entries if entry["file"] == "tanh.py")
    assert tanh["source_path"].replace("\\", "/").endswith("TritonBench_T_v1/tanh.py")
    assert "def tanh" in tanh["ref_code"] or "@triton.jit" in tanh["ref_code"]
    assert "def test_" in tanh["test_code"]
    assert TB_SEPARATOR not in tanh["ref_code"]
    assert TB_SEPARATOR not in tanh["test_code"]


def test_prepare_t_entry_splits_vendor_file_at_separator(tritonbench_root: Path):
    _, t_json = load_t_datasets(tritonbench_root)
    tanh_meta = next(row for row in t_json if row["file"] == "tanh.py")

    prepared = prepare_t_entry(tanh_meta, tritonbench_root)
    source = Path(prepared["source_path"]).read_text(encoding="utf-8")
    ref_part, test_part = source.split(TB_SEPARATOR, maxsplit=1)

    assert ref_part.strip() == prepared["ref_code"]
    assert test_part.strip() == prepared["test_code"]


@pytest.mark.parametrize(
    ("funcname", "expected_file"),
    [
        ("tanh", "tanh.py"),
        ("div", "div.py"),
    ],
)
def test_match_resolves_wrapper_funcname_to_single_jsonl_row(
    tritonbench_root: Path,
    funcname: str,
    expected_file: str,
):
    simple_alpaca, t_json = load_t_datasets(tritonbench_root)
    target = next(
        row
        for row in simple_alpaca
        if f"Wrapper Entry Information: {funcname}(" in row["instruction"]
    )
    entries, errors = match_simple_alpaca_entries([target], t_json)

    assert not errors
    assert len(entries) == 1
    assert entries[0]["file"] == expected_file
