from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRITONBENCH_ROOT = PROJECT_ROOT / "vendor" / "TritonBench"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
LLM_MODELS_PATH = PROJECT_ROOT / "src" / "kernelforge" / "benchmark" / "llm_models.json"


@pytest.fixture(scope="session")
def tritonbench_root() -> Path:
    marker = TRITONBENCH_ROOT / "data" / "TritonBench_T_v1.jsonl"
    if not marker.is_file():
        pytest.skip("vendor/TritonBench is not available")
    return TRITONBENCH_ROOT


@pytest.fixture
def sample_inference_ledger() -> Path:
    path = FIXTURES_DIR / "sample_inference.jsonl"
    assert path.is_file(), f"missing fixture ledger: {path}"
    return path


@pytest.fixture
def llm_config() -> dict[str, Any]:
    return json.loads(LLM_MODELS_PATH.read_text(encoding="utf-8"))
