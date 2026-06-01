from __future__ import annotations

from pathlib import Path

import pytest


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "lite"


def cases(kind: str) -> list[Path]:
    paths = sorted((FIXTURE_ROOT / kind).glob("*.py"))
    assert paths, f"No {kind!r} lite grammar fixtures found"
    return paths


@pytest.mark.parametrize("path", cases("valid"), ids=lambda path: path.stem)
def test_accepts_valid(triton_llguidance, path: Path):
    result = triton_llguidance.match(path.read_text(encoding="utf-8"))
    assert result.accepted, result.error


@pytest.mark.parametrize("path", cases("invalid"), ids=lambda path: path.stem)
def test_rejects_invalid(triton_llguidance, path: Path):
    result = triton_llguidance.match(path.read_text(encoding="utf-8"))
    assert not result.accepted
