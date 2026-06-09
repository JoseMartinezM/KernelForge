from __future__ import annotations

import ast
import copy
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


def _benchmark_prelude(warmup: int, rep: int) -> str:
    return f"""
import json as _tb_json
import os as _tb_os
import torch as _tb_torch
import triton as _tb_triton

_tb_benchmark_results = {{}}

def _tb_jsonable(value):
    if isinstance(value, (list, tuple)):
        return [_tb_jsonable(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return value

def _tb_record_benchmark(name, op):
    try:
        _tb_torch.cuda.synchronize()
        op()
        _tb_torch.cuda.synchronize()
        result = _tb_triton.testing.do_bench(
            op,
            warmup={warmup},
            rep={rep},
            quantiles=[0.5, 0.2, 0.8],
        )
        values = result if isinstance(result, (list, tuple)) else [result]
        _tb_benchmark_results[str(name)] = {{
            "ms": float(values[0]),
            "p20_ms": float(values[1]) if len(values) > 1 else None,
            "p80_ms": float(values[2]) if len(values) > 2 else None,
        }}
    except Exception as exc:
        _tb_benchmark_results[str(name)] = {{"error": str(exc)}}

"""


def _benchmark_footer() -> str:
    return """
with open(_tb_os.environ["TRITONBENCH_BENCHMARK_PATH"], "w", encoding="utf-8") as _tb_f:
    _tb_json.dump(_tb_benchmark_results, _tb_f, indent=2)
"""


class _BenchmarkCallInstrumenter(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        if not node.name.startswith("test_"):
            return node

        instrumented_body: list[ast.stmt] = []
        for statement in node.body:
            instrumented_body.append(statement)
            if benchmark_statement := _make_benchmark_statement(statement):
                instrumented_body.append(benchmark_statement)
        node.body = instrumented_body
        return node


def _make_benchmark_statement(statement: ast.stmt) -> ast.stmt | None:
    if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Call):
        return None

    result_target = next(
        (
            target
            for target in statement.targets
            if isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Name)
            and target.value.id == "results"
        ),
        None,
    )
    if result_target is None:
        return None

    key = copy.deepcopy(result_target.slice)
    call = copy.deepcopy(statement.value)
    benchmark_statement = ast.Expr(
        value=ast.Call(
            func=ast.Name(id="_tb_record_benchmark", ctx=ast.Load()),
            args=[
                key,
                ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=call,
                ),
            ],
            keywords=[],
        )
    )
    return ast.copy_location(benchmark_statement, statement)


def _instrument_test_code_for_benchmark(test_code: str) -> str:
    tree = ast.parse(test_code)
    tree = _BenchmarkCallInstrumenter().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _make_benchmark_script(
    impl_code: str,
    test_code: str,
    *,
    warmup: int,
    rep: int,
) -> str:
    return f"""{impl_code.rstrip()}

{TB_SEPARATOR}

{_benchmark_prelude(warmup, rep)}

{_instrument_test_code_for_benchmark(test_code).rstrip()}

{_benchmark_footer()}
"""


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


def _run_benchmark_script(
    impl_code: str,
    test_code: str,
    benchmark_path: str | Path,
    timeout: int,
    warmup: int,
    rep: int,
) -> dict[str, Any]:
    benchmark_path = Path(benchmark_path)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        script = _make_benchmark_script(
            impl_code,
            test_code,
            warmup=warmup,
            rep=rep,
        )
    except SyntaxError as exc:
        return {
            "ok": False,
            "returncode": None,
            "stdout": "",
            "stderr": f"benchmark instrumentation failed: {exc}",
        }

    with tempfile.TemporaryDirectory(prefix="tritonbench_bench_") as workdir:
        script_path = Path(workdir) / "case.py"
        script_path.write_text(script, encoding="utf-8")

        env = os.environ.copy()
        env["TRITONBENCH_BENCHMARK_PATH"] = str(benchmark_path)
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


def _load_json_result(result_path: str | Path) -> Any:
    with open(result_path, encoding="utf-8") as f:
        return json.load(f)


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


def _summarize_benchmarks(
    pred_cases: dict[str, Any],
    ref_cases: dict[str, Any],
) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    total_pred_ms = 0.0
    total_ref_ms = 0.0
    measured = 0

    for case_name in sorted(set(pred_cases) | set(ref_cases)):
        pred = pred_cases.get(case_name) or {}
        ref = ref_cases.get(case_name) or {}
        pred_ms = pred.get("ms") if isinstance(pred, dict) else None
        ref_ms = ref.get("ms") if isinstance(ref, dict) else None

        case = {"pred": pred, "ref": ref, "pred_ms": pred_ms, "ref_ms": ref_ms}
        if isinstance(pred_ms, int | float) and isinstance(ref_ms, int | float):
            case["speedup"] = ref_ms / pred_ms if pred_ms else None
            total_pred_ms += float(pred_ms)
            total_ref_ms += float(ref_ms)
            measured += 1
        else:
            case["speedup"] = None
        cases[case_name] = case

    pred_ms = total_pred_ms if measured else None
    ref_ms = total_ref_ms if measured else None
    return {
        "pred_ms": pred_ms,
        "ref_ms": ref_ms,
        "speedup": (ref_ms / pred_ms) if pred_ms else None,
        "measured_cases": measured,
        "cases": cases,
    }


def _benchmark_entry(
    entry: dict[str, Any],
    pred_code: str,
    workdir_path: Path,
    timeout: int,
    warmup: int,
    rep: int,
) -> dict[str, Any]:
    pred_benchmark_path = workdir_path / "pred_benchmark.json"
    ref_benchmark_path = workdir_path / "ref_benchmark.json"

    pred = _run_benchmark_script(
        pred_code,
        entry["test_code"],
        pred_benchmark_path,
        timeout=timeout,
        warmup=warmup,
        rep=rep,
    )
    ref = _run_benchmark_script(
        entry["ref_code"],
        entry["test_code"],
        ref_benchmark_path,
        timeout=timeout,
        warmup=warmup,
        rep=rep,
    )

    result: dict[str, Any] = {"pred": pred, "ref": ref}
    if pred["ok"] and ref["ok"]:
        pred_cases = _load_json_result(pred_benchmark_path)
        ref_cases = _load_json_result(ref_benchmark_path)
        result.update(_summarize_benchmarks(pred_cases, ref_cases))
    else:
        result.update(
            {
                "pred_ms": None,
                "ref_ms": None,
                "speedup": None,
                "measured_cases": 0,
                "cases": {},
            }
        )
    return result


def evaluate_entry(
    entry: dict[str, Any],
    pred_code: str,
    timeout: int = 180,
    benchmark: bool = False,
    benchmark_warmup: int = 25,
    benchmark_rep: int = 100,
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

        benchmark_result = None
        if benchmark and not mismatches:
            benchmark_result = _benchmark_entry(
                entry,
                pred_code,
                workdir_path,
                timeout=timeout,
                warmup=benchmark_warmup,
                rep=benchmark_rep,
            )

    result = {
        "file": entry["file"],
        "call@1": True,
        "exe@1": not mismatches,
        "ref": ref,
        "pred": pred,
        "mismatches": mismatches,
    }
    if benchmark_result is not None:
        result["benchmark"] = benchmark_result
        result["pred_ms"] = benchmark_result["pred_ms"]
        result["ref_ms"] = benchmark_result["ref_ms"]
        result["speedup"] = benchmark_result["speedup"]
        result["execution_time"] = benchmark_result["pred_ms"]
    return result
