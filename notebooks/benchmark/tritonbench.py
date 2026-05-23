from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

DEFAULT_TRITONBENCH_ROOT = Path("vendor/TritonBench")
TB_SEPARATOR = "#" * 146


def load_t_datasets(
    tritonbench_root: str | Path = DEFAULT_TRITONBENCH_ROOT,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = Path(tritonbench_root)
    with open(root / "data" / "TritonBench_T_simp_alpac_v1.json", encoding="utf-8") as f:
        simple_alpaca = json.load(f)

    with open(root / "data" / "TritonBench_T_v1.jsonl", encoding="utf-8") as f:
        t_json = json.load(f)

    return simple_alpaca, t_json


def match_simple_alpaca_entries(
    simple_alpaca: list[dict[str, Any]],
    t_json: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = []
    errors = []

    for entry in simple_alpaca:
        match = re.search(r"Wrapper Entry Information: (.+?)\(", entry["instruction"])
        if match is None:
            errors.append({"entry": entry, "error": "no wrapper entry info"})
            continue

        funcname = match.group(1)
        matches = [
            candidate
            for candidate in t_json
            if re.search(rf"^{funcname}\(", candidate["func_inputs"]) is not None
        ]
        if len(matches) != 1:
            errors.append(
                {
                    "entry": entry,
                    "error": f"multiple matches or no matches in jsonl dataset for {funcname}",
                    "matches": matches,
                }
            )
            continue

        entries.append(matches[0])

    return entries, errors


def prepare_t_entry(
    entry: dict[str, Any],
    tritonbench_root: str | Path = DEFAULT_TRITONBENCH_ROOT,
) -> dict[str, Any]:
    prepared = dict(entry)
    source_path = Path(tritonbench_root) / "data" / "TritonBench_T_v1" / entry["file"]
    with open(source_path, encoding="utf-8") as f:
        ref_code, test_code = f.read().split(TB_SEPARATOR, maxsplit=1)

    prepared["source_path"] = str(source_path)
    prepared["ref_code"] = ref_code.strip()
    prepared["test_code"] = test_code.strip()
    assert "def test_" in prepared["test_code"]
    return prepared


def prepare_t_entries(
    entries: list[dict[str, Any]],
    tritonbench_root: str | Path = DEFAULT_TRITONBENCH_ROOT,
) -> list[dict[str, Any]]:
    return [prepare_t_entry(entry, tritonbench_root) for entry in entries]


def load_t_simple_entries(
    tritonbench_root: str | Path = DEFAULT_TRITONBENCH_ROOT,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    simple_alpaca, t_json = load_t_datasets(tritonbench_root)
    entries, errors = match_simple_alpaca_entries(simple_alpaca, t_json)
    return prepare_t_entries(entries, tritonbench_root), errors, t_json, simple_alpaca


def make_prompt(entry: dict[str, Any]) -> str:
    def clean(value: Any) -> str:
        text = str(value or "").strip()
        return "" if text in {"N/A", "None"} else text

    math = clean(entry.get("math"))
    other = clean(entry.get("other"))

    math_block = f"\n## Mathematical Formulation\n{math}" if math else ""
    other_block = f"\n## Additional Information\n{other}" if other else ""

    return f"""\
## Functional Description
{clean(entry.get("description"))}
## Function Signature & Arguments
{clean(entry.get("func_inputs"))}{math_block}
## Reference PyTorch Implementation
```python
{clean(entry.get("torch_code"))}
```{other_block}"""


def cleanup_generated_code(message: str) -> str:
    code = str(message or "").strip()
    if "```python" in code:
        code = code.split("```python", maxsplit=1)[-1]
    if "```" in code:
        code = code.split("```", maxsplit=1)[0]
    code = code.replace("<|im_end|>", "").replace("<|EOT|>", "").strip()

    try:
        ast.parse(code)
    except SyntaxError:
        return code
    return code


def _append_result_dump(script: str) -> str:
    return f"""{script.rstrip()}

import os as _tb_os
import torch as _tb_torch
_tb_torch.save(test_results, _tb_os.environ["TRITONBENCH_RESULT_PATH"])
"""


def _make_test_script(impl_code: str, test_code: str) -> str:
    return _append_result_dump(
        f"""{impl_code.rstrip()}

{TB_SEPARATOR}

{test_code.rstrip()}
"""
    )


def _coerce_timeout_stream(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _run_test_script(
    impl_code: str,
    test_code: str,
    result_path: str | Path,
    timeout: int,
) -> dict[str, Any]:
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    script = _make_test_script(impl_code, test_code)

    with tempfile.TemporaryDirectory(prefix="tritonbench_") as workdir:
        script_path = Path(workdir) / "case.py"
        script_path.write_text(script, encoding="utf-8")

        env = os.environ.copy()
        env["TRITONBENCH_RESULT_PATH"] = str(result_path)
        try:
            run = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "returncode": None,
                "stdout": _coerce_timeout_stream(exc.stdout),
                "stderr": _coerce_timeout_stream(exc.stderr)
                or f"timed out after {timeout}s",
            }

    return {
        "ok": run.returncode == 0,
        "returncode": run.returncode,
        "stdout": run.stdout,
        "stderr": run.stderr,
    }


def _load_torch_result(result_path: str | Path) -> Any:
    import torch

    return torch.load(result_path, map_location="cpu", weights_only=False)


def _compare_values(actual: Any, expected: Any, where: str) -> list[str]:
    import torch

    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if actual.shape != expected.shape:
            return [f"{where}: shape {tuple(actual.shape)} != {tuple(expected.shape)}"]
        if actual.dtype != expected.dtype:
            return [f"{where}: dtype {actual.dtype} != {expected.dtype}"]
        if actual.dtype.is_floating_point or actual.dtype.is_complex:
            if torch.allclose(actual, expected, rtol=1e-3, atol=1e-3, equal_nan=True):
                return []
        elif torch.equal(actual, expected):
            return []
        return [f"{where}: tensor values differ"]

    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        return [f"{where}: type {type(actual).__name__} != {type(expected).__name__}"]

    if isinstance(actual, dict) and isinstance(expected, dict):
        mismatches = []
        if actual.keys() != expected.keys():
            return [f"{where}: keys {actual.keys()} != {expected.keys()}"]
        for key in actual:
            mismatches.extend(
                _compare_values(actual[key], expected[key], f"{where}.{key}")
            )
        return mismatches

    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            return [f"{where}: len {len(actual)} != {len(expected)}"]
        mismatches = []
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            mismatches.extend(
                _compare_values(actual_item, expected_item, f"{where}[{index}]")
            )
        return mismatches

    return [] if actual == expected else [f"{where}: {actual!r} != {expected!r}"]


def compare_results(
    pred_result_path: str | Path, ref_result_path: str | Path
) -> list[str]:
    pred_result = _load_torch_result(pred_result_path)
    ref_result = _load_torch_result(ref_result_path)
    return _compare_values(pred_result, ref_result, "test_results")


def evaluate_entry(
    entry: dict[str, Any],
    pred_code: str,
    timeout: int = 180,
) -> dict[str, Any]:
    pred_code = cleanup_generated_code(pred_code)

    with tempfile.TemporaryDirectory(
        prefix=f"tritonbench_{Path(entry['file']).stem}_"
    ) as workdir:
        workdir_path = Path(workdir)
        pred_result_path = workdir_path / "pred.pt"
        ref_result_path = workdir_path / "ref.pt"

        pred = _run_test_script(
            pred_code,
            entry["test_code"],
            pred_result_path,
            timeout=timeout,
        )
        ref = _run_test_script(
            entry["ref_code"],
            entry["test_code"],
            ref_result_path,
            timeout=timeout,
        )

        call_passed = pred["ok"]
        if not ref["ok"]:
            return {
                "file": entry["file"],
                "call@1": call_passed,
                "exe@1": None,
                "ref": ref,
                "pred": pred,
                "mismatches": ["reference execution failed"],
            }

        if not call_passed:
            return {
                "file": entry["file"],
                "call@1": False,
                "exe@1": False,
                "ref": ref,
                "pred": pred,
                "mismatches": [],
            }

        mismatches = compare_results(pred_result_path, ref_result_path)

    return {
        "file": entry["file"],
        "call@1": True,
        "exe@1": not mismatches,
        "ref": ref,
        "pred": pred,
        "mismatches": mismatches,
    }
