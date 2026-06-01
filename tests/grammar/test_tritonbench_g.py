from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRITONBENCH_G_ROOT = PROJECT_ROOT / "vendor" / "TritonBench" / "data" / "TritonBench_G_v1"
TRITONBENCH_G_INDEX = PROJECT_ROOT / "vendor" / "TritonBench" / "data" / "TritonBench_G_v1.json"
TEST_DELIMITER = "#" * 146

# Raise this in the same change that expands grammar coverage. Keep the floor in
# code so ratchet changes are visible in review instead of hidden in shell state.
TRITONBENCH_G_ACCEPTANCE_FLOOR = 0


def kernel_section(path: Path) -> str:
    code, delimiter, _tests = path.read_text(encoding="utf-8").partition(TEST_DELIMITER)
    assert delimiter, f"{path} does not contain the TritonBench test delimiter"
    return code.rstrip() + "\n"


@pytest.mark.corpus
def test_tritonbench_g_ratchet(triton_llguidance):
    data = json.loads(TRITONBENCH_G_INDEX.read_text(encoding="utf-8"))

    accepted = [
        row["file"]
        for row in data
        if triton_llguidance.match(kernel_section(TRITONBENCH_G_ROOT / row["file"])).accepted
    ]

    assert len(accepted) >= TRITONBENCH_G_ACCEPTANCE_FLOOR, (
        f"LLGuidance grammar accepted {len(accepted)} TritonBench-G snippets, "
        f"below ratchet floor {TRITONBENCH_G_ACCEPTANCE_FLOOR}"
    )
