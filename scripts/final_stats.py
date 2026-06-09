"""Aggregate final KernelForge run statistics for presentation reporting.

Usage:
    uv run python scripts/final_stats.py

The script is intentionally self-contained and dependency-free so it can be run
while the environment is in flux. It reads the known TritonBench and KAGBench
ledgers, stitches interrupted KAGBench shards, computes headline rates with
Wilson confidence intervals, and runs paired exact McNemar/binomial comparisons
where the artifacts support them.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

KAGBENCH_TASK_COUNT = 48
TRITONBENCH_TASK_COUNT = 166
LIBSTDCXX_IMPORT_FRAGMENT = "libstdc++.so.6"
DEFAULT_KAGBENCH_SOURCE_LEDGER = Path("benchmarks/KAGBench/kagbench.jsonl")

DEFAULT_KAGBENCH_CONSTRAINED_DIRS = [
    Path("runs/agent/full-local-llamacpp-4000-teacher3000-3cand-3repair-all-kagbench-20260609T000156Z"),
    Path(
        "runs/agent/"
        "full-local-llamacpp-4000-teacher3000-3cand-3repair-all-kagbench-tail-20260609T020237Z"
    ),
]

DEFAULT_KAGBENCH_NOGRAMMAR_INVALID_DIRS = [
    Path(
        "runs/agent/"
        "full-local-llamacpp-4000-teacher3000-3cand-3repair-all-kagbench-nogrammar-fresh-20260609T031403Z"
    ),
    Path(
        "runs/agent/"
        "full-local-llamacpp-3000-teacher3000-3cand-1repaircand-3repair-nogrammar-missing-20260609T043325Z"
    ),
    Path(
        "runs/agent/"
        "full-local-llamacpp-3000-teacher3000-3cand-1repaircand-3repair-nogrammar-remaining-20260609T050851Z"
    ),
]

DEFAULT_KAGBENCH_NOGRAMMAR_CLEAN_REPLACEMENT_DIRS = [
    Path("runs/agent/full-local-llamacpp-nogrammar-fast-salvage-nix-20260609T060007Z"),
    Path(
        "runs/agent/"
        "full-local-llamacpp-nogrammar-repair-cached-fails-nix-20260609T060455Z"
    ),
    Path("runs/agent/full-local-llamacpp-nogrammar-breadth-remaining-nix-20260609T062030Z"),
    Path("runs/agent/full-local-llamacpp-nogrammar-first24-replay-nix-20260609T063606Z"),
]

DEFAULT_KAGBENCH_NOGRAMMAR_DIRS = DEFAULT_KAGBENCH_NOGRAMMAR_CLEAN_REPLACEMENT_DIRS

TRITONBENCH_LLGUIDANCE_LEDGER = Path("runs/tritonbench/gemma4-e4b-llguidance-full.jsonl")
TRITONBENCH_LLGUIDANCE_EVAL = Path("runs/tritonbench/gemma4-e4b-llguidance-full-eval.jsonl")
TRITONBENCH_LLGUIDANCE_MANIFEST = Path(
    "runs/tritonbench/gemma4-e4b-llguidance-full.jsonl.manifest.json"
)
TRITONBENCH_MODAL_GEMMA_LEDGER = Path(
    "runs/tritonbench/modal-gemma4-e4b-t-simple-max8000-temp0.jsonl"
)
TRITONBENCH_NOTEBOOK_EVAL = Path("notebooks/results/eval_results.json")
TRITONBENCH_NOTEBOOK_GENERATION_LEDGERS = [
    Path("notebooks/data/deepseek-v4-pro-t-simple-max4000-temp0.jsonl"),
    Path("notebooks/data/gpt54-t-simple-max8500-lowreason-temp0.jsonl"),
    Path("notebooks/data/modal-gemma4-e4b-t-simple-max8000-temp0.jsonl"),
]
TRITONBENCH_SPEED_CACHE = Path("runs/tritonbench/final-speed-benchmarks.jsonl")
KAGBENCH_SPEED_CACHE = Path("runs/agent/final-kagbench-speed-benchmarks.jsonl")

TRITONBENCH_LABEL_ORDER = [
    "DeepSeek V4 Pro",
    "GPT 5.4",
    "Modal Gemma 4 E4B vLLM",
    "Gemma 4 E4B llama.cpp + LLGuidance",
]


@dataclass(frozen=True)
class RateSummary:
    successes: int
    total: int
    rate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class PairedComparison:
    name: str
    metric: str
    treatment_label: str
    baseline_label: str
    common_n: int
    treatment_successes: int
    baseline_successes: int
    improved: int
    regressed: int
    both_success: int
    both_fail: int
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    q_value: float | None = None


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row in {path}:{line_number}")
            rows.append(row)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def numeric_or_none(value: Any) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> RateSummary:
    if total <= 0:
        return RateSummary(successes=successes, total=total, rate=0.0, ci_low=0.0, ci_high=0.0)
    phat = successes / total
    denom = 1 + z**2 / total
    centre = phat + z**2 / (2 * total)
    radius = z * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
    return RateSummary(
        successes=successes,
        total=total,
        rate=phat,
        ci_low=max(0.0, (centre - radius) / denom),
        ci_high=min(1.0, (centre + radius) / denom),
    )


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def signed_pct(value: float) -> str:
    return f"{100 * value:+.1f} pp"


def format_rate(summary: RateSummary) -> str:
    return (
        f"{summary.successes}/{summary.total} ({pct(summary.rate)}; "
        f"95% CI {pct(summary.ci_low)}–{pct(summary.ci_high)})"
    )


def format_p(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return ordered[int(pos)]
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_numeric(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "sum": sum(values),
        "mean": mean(values),
        "median": quantile(values, 0.5),
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def metric_bool(row: dict[str, Any], key: str) -> bool:
    value = row.get(key)
    if value is None and key == "call@1":
        value = row.get("call_at_1")
    if value is None and key == "exe@1":
        value = row.get("exe_at_1")
    return bool(value)


def metric_raw(row: dict[str, Any], key: str) -> Any:
    if key in row:
        return row[key]
    if key == "call@1":
        return row.get("call_at_1")
    if key == "exe@1":
        return row.get("exe_at_1")
    return None


def ref_returncode(row: dict[str, Any]) -> Any:
    if "ref_returncode" in row:
        return row.get("ref_returncode")
    ref = row.get("ref")
    if isinstance(ref, dict):
        return ref.get("returncode")
    return None


def pred_returncode(row: dict[str, Any]) -> Any:
    if "pred_returncode" in row:
        return row.get("pred_returncode")
    pred = row.get("pred")
    if isinstance(pred, dict):
        return pred.get("returncode")
    return None


def benchmark_field(row: dict[str, Any], key: str) -> float | None:
    direct = numeric_or_none(row.get(key))
    if direct is not None:
        return direct
    benchmark = row.get("benchmark")
    if isinstance(benchmark, dict):
        return numeric_or_none(benchmark.get(key))
    return None


def normalize_tritonbench_eval(row: dict[str, Any], label: str) -> dict[str, Any]:
    entry_file = row.get("entry_file") or row.get("file")
    exe_raw = metric_raw(row, "exe@1")
    ref_code = ref_returncode(row)
    pred_code = pred_returncode(row)
    return {
        "label": label,
        "model": row.get("model"),
        "model_label": row.get("model_label"),
        "entry_file": entry_file,
        "entry_index": row.get("entry_index"),
        "call": metric_bool(row, "call@1"),
        "exe": bool(exe_raw),
        "exe_raw": exe_raw,
        "ref_returncode": ref_code,
        "pred_returncode": pred_code,
        "ref_failed": ref_code not in (None, 0),
        "mismatches": row.get("mismatches") or [],
        "speedup": benchmark_field(row, "speedup"),
        "pred_ms": benchmark_field(row, "pred_ms"),
        "ref_ms": benchmark_field(row, "ref_ms"),
    }


def load_tritonbench_eval_sets(root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    warnings: list[str] = []
    eval_sets: dict[str, list[dict[str, Any]]] = defaultdict(list)

    notebook_path = resolve_path(root, TRITONBENCH_NOTEBOOK_EVAL)
    if notebook_path.exists():
        data = read_json(notebook_path)
        for row in data.get("per_kernel", []):
            label = row.get("model_label") or row.get("model") or "unknown"
            eval_sets[label].append(normalize_tritonbench_eval(row, label))
    else:
        warnings.append(f"Missing TritonBench notebook eval file: {notebook_path}")

    llguidance_path = resolve_path(root, TRITONBENCH_LLGUIDANCE_EVAL)
    if llguidance_path.exists():
        label = "Gemma 4 E4B llama.cpp + LLGuidance"
        eval_sets[label].extend(
            normalize_tritonbench_eval(row, label) for row in read_jsonl(llguidance_path)
        )
    else:
        warnings.append(f"Missing LLGuidance eval file: {llguidance_path}")

    return dict(eval_sets), warnings


def summarize_tritonbench_eval(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    unique_files = len({row.get("entry_file") for row in rows if row.get("entry_file")})
    call_successes = sum(1 for row in rows if row["call"])
    exe_successes = sum(1 for row in rows if row["exe"])
    speedups = [row["speedup"] for row in rows if row.get("speedup") is not None]
    pred_ms = [row["pred_ms"] for row in rows if row.get("pred_ms") is not None]
    ref_ms = [row["ref_ms"] for row in rows if row.get("ref_ms") is not None]
    return {
        "total": total,
        "unique_files": unique_files,
        "call": wilson(call_successes, total),
        "exe": wilson(exe_successes, total),
        "exe_unknown": sum(1 for row in rows if row.get("exe_raw") is None),
        "ref_failed": sum(1 for row in rows if row.get("ref_failed")),
        "pred_failed": sum(1 for row in rows if row.get("pred_returncode") not in (None, 0)),
        "speedup": summarize_numeric(speedups),
        "pred_ms": summarize_numeric(pred_ms),
        "ref_ms": summarize_numeric(ref_ms),
    }


def extract_usage(row: dict[str, Any]) -> dict[str, Any]:
    usage = row.get("usage")
    if isinstance(usage, dict):
        return usage
    response = row.get("response")
    if isinstance(response, dict) and isinstance(response.get("usage"), dict):
        return response["usage"]
    return {}


def finish_reason(row: dict[str, Any]) -> str | None:
    if row.get("finish_reason") is not None:
        return str(row.get("finish_reason"))
    response = row.get("response")
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    reason = choice.get("finish_reason")
    return str(reason) if reason is not None else None


def summarize_generation_ledger(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "missing": True}
    rows = read_jsonl(path)
    latencies = [float(row["latency_s"]) for row in rows if isinstance(row.get("latency_s"), int | float)]
    completion_tokens = []
    total_tokens = []
    for row in rows:
        usage = extract_usage(row)
        if isinstance(usage.get("completion_tokens"), int | float):
            completion_tokens.append(float(usage["completion_tokens"]))
        if isinstance(usage.get("total_tokens"), int | float):
            total_tokens.append(float(usage["total_tokens"]))
    return {
        "path": str(path),
        "missing": False,
        "rows": len(rows),
        "unique_files": len({row.get("entry_file") for row in rows if row.get("entry_file")}),
        "status": dict(Counter(row.get("status") for row in rows)),
        "model": dict(Counter(row.get("model") for row in rows)),
        "provider": dict(Counter(row.get("provider") for row in rows)),
        "finish_reason": dict(Counter(finish_reason(row) for row in rows)),
        "latency_s": summarize_numeric(latencies),
        "completion_tokens": summarize_numeric(completion_tokens),
        "total_tokens": summarize_numeric(total_tokens),
    }


def tritonbench_generation_label(path: Path, row: dict[str, Any]) -> str:
    if path.name == TRITONBENCH_LLGUIDANCE_LEDGER.name:
        return "Gemma 4 E4B llama.cpp + LLGuidance"
    return str(row.get("model_label") or row.get("model") or "unknown")


def load_tritonbench_generation_code(
    root: Path,
    warnings: list[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    code_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    ledger_paths = [*TRITONBENCH_NOTEBOOK_GENERATION_LEDGERS, TRITONBENCH_LLGUIDANCE_LEDGER]
    for relative_path in ledger_paths:
        path = resolve_path(root, relative_path)
        if not path.exists():
            warnings.append(f"Missing TritonBench generation ledger for speed benchmarks: {path}")
            continue
        for row in read_jsonl(path):
            entry_file = row.get("entry_file")
            content = row.get("content")
            if row.get("status") != "success" or not entry_file or not content:
                continue
            label = tritonbench_generation_label(relative_path, row)
            code_by_key[(label, str(entry_file))] = {
                "content": str(content),
                "path": str(path),
                "request_hash": row.get("request_hash"),
            }
    return code_by_key


def speed_cache_key(
    label: str,
    entry_file: str,
    content: str,
    *,
    warmup: int,
    rep: int,
    timeout: int,
) -> str:
    content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    return json.dumps(
        {
            "label": label,
            "entry_file": entry_file,
            "content_sha256": content_hash,
            "warmup": warmup,
            "rep": rep,
            "timeout": timeout,
        },
        sort_keys=True,
    )


def load_speed_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = row.get("cache_key")
        if isinstance(key, str):
            cache[key] = row
    return cache


def apply_speed_result(eval_row: dict[str, Any], result: dict[str, Any]) -> None:
    eval_row["speedup"] = numeric_or_none(result.get("speedup"))
    eval_row["pred_ms"] = numeric_or_none(result.get("pred_ms"))
    eval_row["ref_ms"] = numeric_or_none(result.get("ref_ms"))
    eval_row["speed_benchmark_status"] = result.get("status")
    if result.get("error"):
        eval_row["speed_benchmark_error"] = result.get("error")


def benchmark_tritonbench_speedups(
    root: Path,
    eval_sets: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    warnings: list[str],
) -> dict[str, Any]:
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from kernelforge.benchmark.tritonbench import evaluate_entry, load_t_simple_entries

    entries, errors, _, _ = load_t_simple_entries(resolve_path(root, Path("vendor/TritonBench")))
    if errors:
        warnings.append(f"TritonBench speed benchmark dataset load errors: {errors}")
    entries_by_file = {entry["file"]: entry for entry in entries}
    generation_code = load_tritonbench_generation_code(root, warnings)
    cache_path = resolve_path(root, args.speed_benchmark_cache)
    cache = load_speed_cache(cache_path)

    report = {
        "enabled": True,
        "cache_path": str(cache_path),
        "warmup": args.speed_benchmark_warmup,
        "rep": args.speed_benchmark_rep,
        "timeout": args.speed_benchmark_timeout,
        "cuda_available": None,
        "eligible": 0,
        "cached": 0,
        "measured": 0,
        "missing_code": 0,
        "missing_entry": 0,
        "failed": 0,
        "limited": False,
    }

    can_run_benchmarks = True
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - final stats should still render without GPU deps.
        report["cuda_available"] = False
        warnings.append(f"Skipped TritonBench speed benchmarks because PyTorch import failed: {exc}")
        can_run_benchmarks = False
    else:
        report["cuda_available"] = bool(torch.cuda.is_available())

    if report["cuda_available"] is False:
        report["cuda_available"] = False
        warnings.append(
            "Only cached TritonBench speed benchmarks can be used because torch.cuda is not available."
        )
        can_run_benchmarks = False

    remaining = args.speed_benchmark_limit
    for label in sorted(eval_sets):
        for eval_row in eval_sets[label]:
            if not eval_row.get("exe"):
                continue
            entry_file = eval_row.get("entry_file")
            if not entry_file:
                continue
            report["eligible"] += 1
            code_row = generation_code.get((label, str(entry_file)))
            if code_row is None:
                report["missing_code"] += 1
                continue
            entry = entries_by_file.get(str(entry_file))
            if entry is None:
                report["missing_entry"] += 1
                continue

            content = code_row["content"]
            key = speed_cache_key(
                label,
                str(entry_file),
                content,
                warmup=args.speed_benchmark_warmup,
                rep=args.speed_benchmark_rep,
                timeout=args.speed_benchmark_timeout,
            )
            if key in cache:
                apply_speed_result(eval_row, cache[key])
                report["cached"] += 1
                continue
            if not can_run_benchmarks:
                continue
            if remaining is not None and remaining <= 0:
                report["limited"] = True
                continue

            print(
                f"benchmarking {label} {entry_file} "
                f"(warmup={args.speed_benchmark_warmup}, rep={args.speed_benchmark_rep})",
                file=sys.stderr,
                flush=True,
            )
            try:
                result = evaluate_entry(
                    entry,
                    pred_code=content,
                    timeout=args.speed_benchmark_timeout,
                    benchmark=True,
                    benchmark_warmup=args.speed_benchmark_warmup,
                    benchmark_rep=args.speed_benchmark_rep,
                )
                benchmark = result.get("benchmark") if isinstance(result, dict) else None
                cache_row = {
                    "cache_key": key,
                    "status": "measured" if isinstance(benchmark, dict) else "failed",
                    "label": label,
                    "entry_file": entry_file,
                    "request_hash": code_row.get("request_hash"),
                    "source_ledger": code_row.get("path"),
                    "call@1": result.get("call@1"),
                    "exe@1": result.get("exe@1"),
                    "pred_ms": benchmark_field(result, "pred_ms"),
                    "ref_ms": benchmark_field(result, "ref_ms"),
                    "speedup": benchmark_field(result, "speedup"),
                    "benchmark": benchmark,
                    "mismatches": result.get("mismatches") or [],
                }
            except Exception as exc:  # noqa: BLE001 - cache failures per kernel and keep report going.
                cache_row = {
                    "cache_key": key,
                    "status": "failed",
                    "label": label,
                    "entry_file": entry_file,
                    "request_hash": code_row.get("request_hash"),
                    "source_ledger": code_row.get("path"),
                    "error": str(exc),
                    "pred_ms": None,
                    "ref_ms": None,
                    "speedup": None,
                }

            append_jsonl(cache_path, cache_row)
            cache[key] = cache_row
            apply_speed_result(eval_row, cache_row)
            if cache_row.get("speedup") is not None:
                report["measured"] += 1
            else:
                report["failed"] += 1
            if remaining is not None:
                remaining -= 1

    if report["missing_code"]:
        warnings.append(
            f"Skipped {report['missing_code']} TritonBench speed benchmarks because generated code "
            "was not found in the known generation ledgers."
        )
    if report["failed"]:
        warnings.append(f"{report['failed']} TritonBench speed benchmark attempts failed.")
    return report


def _kagbench_benchmark_prelude(warmup: int, rep: int) -> str:
    return f"""
import torch as _kf_torch
import triton as _kf_triton

_kf_benchmarks = {{"candidate": [], "reference": []}}

def _kf_record_benchmark(kind, name, op):
    try:
        _kf_torch.cuda.synchronize()
        op()
        _kf_torch.cuda.synchronize()
        result = _kf_triton.testing.do_bench(
            op,
            warmup={warmup},
            rep={rep},
            quantiles=[0.5, 0.2, 0.8],
        )
        values = result if isinstance(result, (list, tuple)) else [result]
        _kf_benchmarks[kind].append({{
            "name": str(name),
            "ms": float(values[0]),
            "p20_ms": float(values[1]) if len(values) > 1 else None,
            "p80_ms": float(values[2]) if len(values) > 2 else None,
        }})
    except Exception as exc:
        _kf_benchmarks[kind].append({{"name": str(name), "error": str(exc)}})

"""


def _kagbench_call_kind(call: ast.Call) -> str | None:
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        if func.value.id in {"candidate", "candidate_fn"}:
            return "candidate"
        if func.value.id in {"reference", "pytorch_reference"}:
            return "reference"
    if isinstance(func, ast.Name):
        if func.id in {"fn", "candidate_fn"}:
            return "candidate"
        if func.id.startswith("reference_"):
            return "reference"
    return None


class _KAGBenchCallInstrumenter(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        instrumented_body: list[ast.stmt] = []
        for statement in node.body:
            instrumented_body.append(statement)
            if benchmark_statement := _make_kagbench_benchmark_statement(statement):
                instrumented_body.append(benchmark_statement)
        node.body = instrumented_body
        return node


def _make_kagbench_benchmark_statement(statement: ast.stmt) -> ast.stmt | None:
    call: ast.Call | None = None
    name = "call"
    if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
        call = statement.value
        if statement.targets and isinstance(statement.targets[0], ast.Name):
            name = statement.targets[0].id
    elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
        call = statement.value
    if call is None:
        return None

    kind = _kagbench_call_kind(call)
    if kind is None:
        return None
    benchmark_statement = ast.Expr(
        value=ast.Call(
            func=ast.Name(id="_kf_record_benchmark", ctx=ast.Load()),
            args=[
                ast.Constant(value=kind),
                ast.Constant(value=name),
                ast.Lambda(
                    args=ast.arguments(
                        posonlyargs=[],
                        args=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        defaults=[],
                    ),
                    body=copy_ast(call),
                ),
            ],
            keywords=[],
        )
    )
    return ast.copy_location(benchmark_statement, statement)


def copy_ast(node: ast.AST) -> ast.AST:
    return ast.fix_missing_locations(ast.parse(ast.unparse(node), mode="eval").body)


def instrument_kagbench_tests_for_benchmark(test_code: str, *, warmup: int, rep: int) -> str:
    tree = ast.parse(test_code)
    tree = _KAGBenchCallInstrumenter().visit(tree)
    ast.fix_missing_locations(tree)
    return _kagbench_benchmark_prelude(warmup, rep) + "\n" + ast.unparse(tree)


def _kagbench_benchmark_runner(phase: str) -> str:
    test_function_name = "unit_tests" if phase == "hidden" else "public_tests"
    return f"""
import importlib
import json
import traceback

RESULT_PREFIX = "__KAGBENCH_SPEED_RESULT__"

try:
    import torch
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    candidate = importlib.import_module("candidate")
    tests_module = importlib.import_module("tests_module")
    getattr(tests_module, {test_function_name!r})(candidate)
    print(RESULT_PREFIX + json.dumps({{"ok": True, "benchmarks": tests_module._kf_benchmarks}}), flush=True)
except BaseException as exc:
    print(RESULT_PREFIX + json.dumps({{
        "ok": False,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
        "benchmarks": getattr(importlib.import_module("tests_module"), "_kf_benchmarks", {{}}) if "tests_module" in globals() else {{}},
    }}), flush=True)
"""


def _parse_kagbench_speed_result(stdout: str) -> dict[str, Any] | None:
    prefix = "__KAGBENCH_SPEED_RESULT__"
    for line in reversed(stdout.splitlines()):
        if line.startswith(prefix):
            parsed = json.loads(line[len(prefix) :])
            return parsed if isinstance(parsed, dict) else None
    return None


def summarize_kagbench_benchmarks(benchmarks: dict[str, Any]) -> dict[str, Any]:
    candidate_cases = [
        item for item in benchmarks.get("candidate", []) if isinstance(item, dict)
    ]
    reference_cases = [
        item for item in benchmarks.get("reference", []) if isinstance(item, dict)
    ]
    candidate_ms = [numeric_or_none(item.get("ms")) for item in candidate_cases]
    reference_ms = [numeric_or_none(item.get("ms")) for item in reference_cases]
    candidate_values = [value for value in candidate_ms if value is not None]
    reference_values = [value for value in reference_ms if value is not None]
    pred_ms = sum(candidate_values) if candidate_values else None
    ref_ms = sum(reference_values) if reference_values else None
    return {
        "pred_ms": pred_ms,
        "ref_ms": ref_ms,
        "speedup": (ref_ms / pred_ms) if pred_ms and ref_ms else None,
        "measured_candidate_calls": len(candidate_values),
        "measured_reference_calls": len(reference_values),
        "candidate_calls": candidate_cases,
        "reference_calls": reference_cases,
    }


def run_kagbench_speed_benchmark(
    *,
    candidate_code: str,
    pytorch_reference: str,
    test_code: str,
    phase: str,
    timeout: int,
    warmup: int,
    rep: int,
) -> dict[str, Any]:
    instrumented_tests = instrument_kagbench_tests_for_benchmark(test_code, warmup=warmup, rep=rep)
    with tempfile.TemporaryDirectory(prefix="kernelforge_kagbench_speed_") as workdir:
        workdir_path = Path(workdir)
        (workdir_path / "candidate.py").write_text(candidate_code, encoding="utf-8")
        (workdir_path / "pytorch_reference.py").write_text(pytorch_reference, encoding="utf-8")
        (workdir_path / "tests_module.py").write_text(instrumented_tests, encoding="utf-8")
        (workdir_path / "run_benchmark.py").write_text(_kagbench_benchmark_runner(phase), encoding="utf-8")

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(workdir_path)
            if not existing_pythonpath
            else f"{workdir_path}{os.pathsep}{existing_pythonpath}"
        )
        try:
            run = subprocess.run(
                [sys.executable, str(workdir_path / "run_benchmark.py")],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "status": "failed",
                "error": f"timed out after {timeout}s",
                "stdout": exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                "stderr": exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                "pred_ms": None,
                "ref_ms": None,
                "speedup": None,
            }

    parsed = _parse_kagbench_speed_result(run.stdout)
    if parsed is None:
        return {
            "status": "failed",
            "error": f"subprocess exited {run.returncode} without speed result marker",
            "stdout": run.stdout,
            "stderr": run.stderr,
            "pred_ms": None,
            "ref_ms": None,
            "speedup": None,
        }
    summary = summarize_kagbench_benchmarks(parsed.get("benchmarks") or {})
    status = "measured" if parsed.get("ok") and summary.get("speedup") is not None else "failed"
    return {
        "status": status,
        "error": None if status == "measured" else parsed.get("error_message") or "no paired benchmark calls measured",
        "stdout": run.stdout,
        "stderr": run.stderr,
        **summary,
    }


def kagbench_speed_cache_key(
    label: str,
    task_id: str,
    candidate_code: str,
    *,
    phase: str,
    warmup: int,
    rep: int,
    timeout: int,
) -> str:
    content_hash = hashlib.sha256(candidate_code.encode("utf-8", errors="replace")).hexdigest()
    return json.dumps(
        {
            "label": label,
            "task_id": task_id,
            "content_sha256": content_hash,
            "phase": phase,
            "warmup": warmup,
            "rep": rep,
            "timeout": timeout,
        },
        sort_keys=True,
    )


def benchmark_kagbench_speedups(
    root: Path,
    runs: list[dict[str, Any]],
    args: argparse.Namespace,
    warnings: list[str],
) -> dict[str, Any]:
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from kernelforge.agent.tasks import load_kagbench

    tasks_by_id = {task.task_id: task for task in load_kagbench(resolve_path(root, DEFAULT_KAGBENCH_SOURCE_LEDGER))}
    cache_path = resolve_path(root, args.kagbench_speed_cache)
    cache = load_speed_cache(cache_path)
    report = {
        "enabled": True,
        "cache_path": str(cache_path),
        "warmup": args.speed_benchmark_warmup,
        "rep": args.speed_benchmark_rep,
        "timeout": args.speed_benchmark_timeout,
        "phase": "hidden",
        "eligible": 0,
        "cached": 0,
        "measured": 0,
        "missing_code": 0,
        "failed": 0,
        "limited": False,
    }
    remaining = args.kagbench_speed_limit
    for run in runs:
        speedups: list[float] = []
        pred_ms_values: list[float] = []
        ref_ms_values: list[float] = []
        candidate_by_source_id = run.get("candidate_by_source_id") or {}
        for summary in run.get("summaries", []):
            if not hidden_passed(summary):
                continue
            task_id = summary.get("task_id")
            candidate_id = summary.get("final_candidate_id")
            source_dir = summary.get("__source_dir")
            if not isinstance(task_id, str) or not isinstance(candidate_id, str) or not isinstance(source_dir, str):
                continue
            report["eligible"] += 1
            candidate_row = candidate_by_source_id.get((source_dir, candidate_id))
            if not isinstance(candidate_row, dict) or not isinstance(candidate_row.get("content"), str):
                report["missing_code"] += 1
                continue
            task = tasks_by_id.get(task_id)
            if task is None:
                report["missing_code"] += 1
                continue

            candidate_code = candidate_row["content"]
            key = kagbench_speed_cache_key(
                run["label"],
                task_id,
                candidate_code,
                phase="hidden",
                warmup=args.speed_benchmark_warmup,
                rep=args.speed_benchmark_rep,
                timeout=args.speed_benchmark_timeout,
            )
            cache_hit = key in cache
            if cache_hit:
                cache_row = cache[key]
                report["cached"] += 1
            else:
                if remaining is not None and remaining <= 0:
                    report["limited"] = True
                    continue
                print(
                    f"benchmarking {run['label']} {task_id} hidden "
                    f"(warmup={args.speed_benchmark_warmup}, rep={args.speed_benchmark_rep})",
                    file=sys.stderr,
                    flush=True,
                )
                result = run_kagbench_speed_benchmark(
                    candidate_code=candidate_code,
                    pytorch_reference=task.pytorch_reference,
                    test_code=task.unit_tests,
                    phase="hidden",
                    timeout=args.speed_benchmark_timeout,
                    warmup=args.speed_benchmark_warmup,
                    rep=args.speed_benchmark_rep,
                )
                cache_row = {
                    "cache_key": key,
                    "label": run["label"],
                    "task_id": task_id,
                    "entry_file": summary.get("entry_file"),
                    "candidate_id": candidate_id,
                    "source_dir": source_dir,
                    **result,
                }
                append_jsonl(cache_path, cache_row)
                cache[key] = cache_row
                if remaining is not None:
                    remaining -= 1

            speedup = numeric_or_none(cache_row.get("speedup"))
            pred_ms = numeric_or_none(cache_row.get("pred_ms"))
            ref_ms = numeric_or_none(cache_row.get("ref_ms"))
            summary["speedup"] = speedup
            summary["pred_ms"] = pred_ms
            summary["ref_ms"] = ref_ms
            summary["speed_benchmark_status"] = cache_row.get("status")
            if speedup is not None:
                speedups.append(speedup)
                if not cache_hit:
                    report["measured"] += 1
            elif cache_row.get("status") == "failed":
                if not cache_hit:
                    report["failed"] += 1
            if pred_ms is not None:
                pred_ms_values.append(pred_ms)
            if ref_ms is not None:
                ref_ms_values.append(ref_ms)

        run["speedup"] = summarize_numeric(speedups)
        run["pred_ms"] = summarize_numeric(pred_ms_values)
        run["ref_ms"] = summarize_numeric(ref_ms_values)
        run.pop("candidate_by_source_id", None)

    if report["missing_code"]:
        warnings.append(
            f"Skipped {report['missing_code']} KAGBench speed benchmarks because final candidate code was not found."
        )
    return report


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    smaller = min(improved, regressed)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2 * tail)


def bootstrap_diff_ci(
    diffs: list[int], *, samples: int, seed: int, alpha: float = 0.05
) -> tuple[float, float]:
    if not diffs:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(diffs)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    low_index = max(0, min(samples - 1, int((alpha / 2) * samples)))
    high_index = max(0, min(samples - 1, int((1 - alpha / 2) * samples) - 1))
    return estimates[low_index], estimates[high_index]


def paired_comparison(
    *,
    name: str,
    metric: str,
    treatment_label: str,
    treatment_rows: list[dict[str, Any]],
    baseline_label: str,
    baseline_rows: list[dict[str, Any]],
    bootstrap_samples: int,
    seed: int,
) -> PairedComparison:
    treatment_by_file = {row["entry_file"]: row for row in treatment_rows if row.get("entry_file")}
    baseline_by_file = {row["entry_file"]: row for row in baseline_rows if row.get("entry_file")}
    common_files = sorted(set(treatment_by_file) & set(baseline_by_file))

    treatment_successes = 0
    baseline_successes = 0
    improved = 0
    regressed = 0
    both_success = 0
    both_fail = 0
    diffs: list[int] = []
    for entry_file in common_files:
        treatment_value = bool(treatment_by_file[entry_file][metric])
        baseline_value = bool(baseline_by_file[entry_file][metric])
        treatment_successes += int(treatment_value)
        baseline_successes += int(baseline_value)
        diff = int(treatment_value) - int(baseline_value)
        diffs.append(diff)
        if treatment_value and baseline_value:
            both_success += 1
        elif treatment_value and not baseline_value:
            improved += 1
        elif not treatment_value and baseline_value:
            regressed += 1
        else:
            both_fail += 1

    common_n = len(common_files)
    diff_rate = sum(diffs) / common_n if common_n else 0.0
    ci_low, ci_high = bootstrap_diff_ci(diffs, samples=bootstrap_samples, seed=seed)
    return PairedComparison(
        name=name,
        metric=metric,
        treatment_label=treatment_label,
        baseline_label=baseline_label,
        common_n=common_n,
        treatment_successes=treatment_successes,
        baseline_successes=baseline_successes,
        improved=improved,
        regressed=regressed,
        both_success=both_success,
        both_fail=both_fail,
        diff=diff_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=exact_mcnemar_p(improved, regressed),
    )


def benjamini_hochberg(comparisons: list[PairedComparison]) -> list[PairedComparison]:
    indexed = sorted(enumerate(comparisons), key=lambda item: item[1].p_value)
    m = len(indexed)
    adjusted = [1.0] * m
    running_min = 1.0
    for rank_from_end, (original_index, comparison) in enumerate(reversed(indexed), start=1):
        rank = m - rank_from_end + 1
        q_value = min(running_min, comparison.p_value * m / rank)
        running_min = q_value
        adjusted[original_index] = min(1.0, q_value)

    return [
        PairedComparison(
            **{
                **comparison.__dict__,
                "q_value": adjusted[index],
            }
        )
        for index, comparison in enumerate(comparisons)
    ]


def manifest_task_ids(manifest: dict[str, Any]) -> list[str]:
    tasks = manifest.get("tasks") or manifest.get("task_ids") or []
    result = []
    for task in tasks:
        if isinstance(task, dict) and isinstance(task.get("task_id"), str):
            result.append(task["task_id"])
        elif isinstance(task, str):
            result.append(task)
    return result


def load_kagbench_source_task_ids(path: Path) -> list[str]:
    task_ids = []
    if not path.exists():
        return task_ids
    for row in read_jsonl(path):
        task_id = row.get("id") or row.get("task_id")
        if isinstance(task_id, str):
            task_ids.append(task_id)
    return task_ids


def row_created_at_key(row: dict[str, Any]) -> str:
    value = row.get("created_at")
    return value if isinstance(value, str) else ""


def public_passed(summary: dict[str, Any]) -> bool:
    if isinstance(summary.get("public_passed"), bool):
        return summary["public_passed"]
    return summary.get("status") in {"public_pass", "hidden_pass", "hidden_fail"}


def hidden_passed(summary: dict[str, Any]) -> bool:
    if isinstance(summary.get("hidden_passed"), bool):
        return summary["hidden_passed"]
    return summary.get("status") == "hidden_pass"


def load_kagbench_stitch(root: Path, label: str, dirs: list[Path]) -> dict[str, Any]:
    resolved_dirs = [resolve_path(root, path) for path in dirs]
    warnings: list[str] = []
    manifests = []
    reference_tasks: list[str] = []
    selected: dict[str, dict[str, Any]] = {}
    duplicates: list[dict[str, str]] = []
    summaries_by_source_task: dict[tuple[str, str], dict[str, Any]] = {}
    eval_rows_by_source_task: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    generation_rows: list[dict[str, Any]] = []

    for source_order, source_dir in enumerate(resolved_dirs):
        if not source_dir.exists():
            warnings.append(f"Missing KAGBench shard: {source_dir}")
            continue

        manifest_path = source_dir / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            manifests.append(manifest)
            task_ids = manifest_task_ids(manifest)
            if len(task_ids) > len(reference_tasks):
                reference_tasks = task_ids

        summary_path = source_dir / "summary.jsonl"
        if summary_path.exists():
            for row in read_jsonl(summary_path):
                task_id = row.get("task_id")
                if not isinstance(task_id, str):
                    continue
                enriched = dict(row)
                enriched["__source_dir"] = str(source_dir.relative_to(root))
                enriched["__source_order"] = source_order
                enriched["__created_at_key"] = row_created_at_key(enriched)
                key = (enriched["__source_dir"], task_id)
                summaries_by_source_task[key] = enriched
                if task_id in selected:
                    previous = selected[task_id]
                    duplicates.append(
                        {
                            "task_id": task_id,
                            "old_source": previous["__source_dir"],
                            "new_source": enriched["__source_dir"],
                        }
                    )
                    previous_time = previous.get("__created_at_key", "")
                    current_time = enriched.get("__created_at_key", "")
                    previous_order = int(previous.get("__source_order", -1))
                    if previous_time and current_time and current_time < previous_time:
                        continue
                    if previous_time and not current_time:
                        continue
                    if current_time == previous_time and source_order < previous_order:
                        continue
                selected[task_id] = enriched

        evaluation_path = source_dir / "evaluation.jsonl"
        if evaluation_path.exists():
            source_name = str(source_dir.relative_to(root))
            for row in read_jsonl(evaluation_path):
                task_id = row.get("task_id")
                if isinstance(task_id, str):
                    eval_rows_by_source_task[(source_name, task_id)].append(row)

        generation_path = source_dir / "generation.jsonl"
        if generation_path.exists():
            source_name = str(source_dir.relative_to(root))
            for row in read_jsonl(generation_path):
                enriched = dict(row)
                enriched["__source_dir"] = source_name
                generation_rows.append(enriched)

    if len(reference_tasks) < KAGBENCH_TASK_COUNT:
        source_tasks = load_kagbench_source_task_ids(resolve_path(root, DEFAULT_KAGBENCH_SOURCE_LEDGER))
        if len(source_tasks) >= KAGBENCH_TASK_COUNT:
            reference_tasks = source_tasks

    if not reference_tasks:
        reference_tasks = sorted(selected)

    selected_order = reference_tasks + sorted(task_id for task_id in selected if task_id not in reference_tasks)
    selected_summaries = [selected[task_id] for task_id in selected_order if task_id in selected]
    missing_tasks = [task_id for task_id in reference_tasks if task_id not in selected]

    owned_eval_rows: list[dict[str, Any]] = []
    for summary in selected_summaries:
        source = summary["__source_dir"]
        task_id = summary["task_id"]
        owned_eval_rows.extend(eval_rows_by_source_task.get((source, task_id), []))

    status_counts = Counter(summary.get("status") for summary in selected_summaries)
    public_successes = sum(1 for summary in selected_summaries if public_passed(summary))
    hidden_successes = sum(1 for summary in selected_summaries if hidden_passed(summary))
    expected = max(KAGBENCH_TASK_COUNT, len(reference_tasks)) if reference_tasks else KAGBENCH_TASK_COUNT
    import_error_rows = [
        row
        for row in owned_eval_rows
        if row.get("error_type") == "ImportError"
        and LIBSTDCXX_IMPORT_FRAGMENT in str(row.get("error_message", ""))
    ]
    invalid_environment = bool(import_error_rows)

    generation_kind_counts = Counter(row.get("kind") for row in generation_rows)
    candidate_rows = [row for row in generation_rows if row.get("kind") == "candidate_generation"]
    candidate_by_source_id = {
        (row.get("__source_dir"), row.get("candidate_id")): row
        for row in candidate_rows
        if isinstance(row.get("candidate_id"), str)
        and isinstance(row.get("__source_dir"), str)
    }
    repair_rows = [row for row in generation_rows if row.get("kind") == "repair_directive"]
    attempts = [
        float(summary.get("attempts"))
        for summary in selected_summaries
        if isinstance(summary.get("attempts"), int | float)
    ]
    candidates_generated = [
        float(summary.get("candidates_generated"))
        for summary in selected_summaries
        if isinstance(summary.get("candidates_generated"), int | float)
    ]

    grammar_backends = Counter(manifest.get("grammar_backend") for manifest in manifests)
    grammar_present = Counter(bool(manifest.get("grammar")) for manifest in manifests)
    if "XGrammar" in label or "xgrammar" in label.lower():
        if any(manifest.get("grammar_backend") != "xgrammar" for manifest in manifests):
            warnings.append(
                f"{label}: requested/found label mentions XGrammar, but manifests report "
                f"grammar_backend={dict(grammar_backends)}"
            )

    return {
        "label": label,
        "dirs": [str(path.relative_to(root)) for path in resolved_dirs],
        "expected_tasks": expected,
        "reference_tasks": reference_tasks,
        "summaries": selected_summaries,
        "summary_by_task": {summary["task_id"]: summary for summary in selected_summaries},
        "missing_tasks": missing_tasks,
        "duplicates": duplicates,
        "complete": len(selected_summaries) == expected and not missing_tasks,
        "invalid_environment": invalid_environment,
        "libstdcxx_import_errors": len(import_error_rows),
        "status_counts": dict(status_counts),
        "source_counts": dict(Counter(summary["__source_dir"] for summary in selected_summaries)),
        "public": wilson(public_successes, expected),
        "hidden": wilson(hidden_successes, expected),
        "hidden_conditional": wilson(hidden_successes, public_successes),
        "owned_eval_rows": len(owned_eval_rows),
        "eval_error_types": dict(Counter(row.get("error_type") for row in owned_eval_rows)),
        "generation_kind_counts": dict(generation_kind_counts),
        "candidate_rows": len(candidate_rows),
        "candidate_by_source_id": candidate_by_source_id,
        "repair_rows": len(repair_rows),
        "attempts": summarize_numeric(attempts),
        "candidates_generated": summarize_numeric(candidates_generated),
        "grammar_backends": dict(grammar_backends),
        "grammar_present": dict(grammar_present),
        "warnings": warnings,
    }


def kagbench_pair_comparison(
    *,
    metric: str,
    treatment: dict[str, Any],
    baseline: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> PairedComparison | None:
    if not treatment["complete"] or not baseline["complete"]:
        return None
    treatment_by_task = treatment["summary_by_task"]
    baseline_by_task = baseline["summary_by_task"]
    common_tasks = sorted(set(treatment_by_task) & set(baseline_by_task))
    if len(common_tasks) != KAGBENCH_TASK_COUNT:
        return None

    treatment_successes = 0
    baseline_successes = 0
    improved = 0
    regressed = 0
    both_success = 0
    both_fail = 0
    diffs: list[int] = []
    for task_id in common_tasks:
        if metric == "public":
            treatment_value = public_passed(treatment_by_task[task_id])
            baseline_value = public_passed(baseline_by_task[task_id])
        elif metric == "hidden":
            treatment_value = hidden_passed(treatment_by_task[task_id])
            baseline_value = hidden_passed(baseline_by_task[task_id])
        else:
            raise ValueError(f"Unsupported KAGBench metric: {metric}")

        treatment_successes += int(treatment_value)
        baseline_successes += int(baseline_value)
        diff = int(treatment_value) - int(baseline_value)
        diffs.append(diff)
        if treatment_value and baseline_value:
            both_success += 1
        elif treatment_value and not baseline_value:
            improved += 1
        elif not treatment_value and baseline_value:
            regressed += 1
        else:
            both_fail += 1

    diff_rate = sum(diffs) / len(diffs)
    ci_low, ci_high = bootstrap_diff_ci(diffs, samples=bootstrap_samples, seed=seed)
    return PairedComparison(
        name="KAGBench constrained vs no-grammar",
        metric=metric,
        treatment_label=treatment["label"],
        baseline_label=baseline["label"],
        common_n=len(common_tasks),
        treatment_successes=treatment_successes,
        baseline_successes=baseline_successes,
        improved=improved,
        regressed=regressed,
        both_success=both_success,
        both_fail=both_fail,
        diff=diff_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=exact_mcnemar_p(improved, regressed),
    )


def kagbench_common_task_comparison(
    *,
    name: str,
    metric: str,
    treatment: dict[str, Any],
    baseline: dict[str, Any],
    bootstrap_samples: int,
    seed: int,
) -> PairedComparison | None:
    """Compare KAGBench summaries on the common task subset, even if incomplete."""
    treatment_by_task = treatment["summary_by_task"]
    baseline_by_task = baseline["summary_by_task"]
    common_tasks = sorted(set(treatment_by_task) & set(baseline_by_task))
    if not common_tasks:
        return None

    treatment_successes = 0
    baseline_successes = 0
    improved = 0
    regressed = 0
    both_success = 0
    both_fail = 0
    diffs: list[int] = []
    for task_id in common_tasks:
        if metric == "public":
            treatment_value = public_passed(treatment_by_task[task_id])
            baseline_value = public_passed(baseline_by_task[task_id])
        elif metric == "hidden":
            treatment_value = hidden_passed(treatment_by_task[task_id])
            baseline_value = hidden_passed(baseline_by_task[task_id])
        else:
            raise ValueError(f"Unsupported KAGBench metric: {metric}")

        treatment_successes += int(treatment_value)
        baseline_successes += int(baseline_value)
        diff = int(treatment_value) - int(baseline_value)
        diffs.append(diff)
        if treatment_value and baseline_value:
            both_success += 1
        elif treatment_value and not baseline_value:
            improved += 1
        elif not treatment_value and baseline_value:
            regressed += 1
        else:
            both_fail += 1

    diff_rate = sum(diffs) / len(diffs)
    ci_low, ci_high = bootstrap_diff_ci(diffs, samples=bootstrap_samples, seed=seed)
    return PairedComparison(
        name=name,
        metric=metric,
        treatment_label=treatment["label"],
        baseline_label=baseline["label"],
        common_n=len(common_tasks),
        treatment_successes=treatment_successes,
        baseline_successes=baseline_successes,
        improved=improved,
        regressed=regressed,
        both_success=both_success,
        both_fail=both_fail,
        diff=diff_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=exact_mcnemar_p(improved, regressed),
    )


def subset_kagbench_run(run: dict[str, Any], task_ids: set[str]) -> dict[str, Any]:
    return {
        "label": run["label"],
        "summary_by_task": {
            task_id: summary
            for task_id, summary in run["summary_by_task"].items()
            if task_id in task_ids
        },
    }


def load_kagbench_task_metadata(root: Path) -> dict[str, dict[str, Any]]:
    metadata = {}
    source_path = resolve_path(root, DEFAULT_KAGBENCH_SOURCE_LEDGER)
    if not source_path.exists():
        return metadata
    for row in read_jsonl(source_path):
        task_id = row.get("id") or row.get("task_id")
        if not isinstance(task_id, str):
            continue
        tags = row.get("tags") if isinstance(row.get("tags"), list) else []
        metadata[task_id] = {
            "task_id": task_id,
            "entry_file": row.get("entry_file"),
            "prefix": task_id.split("/", 1)[0],
            "tags": [tag for tag in tags if isinstance(tag, str)],
        }
    return metadata


def build_kagbench_subgroup_comparisons(
    *,
    constrained: dict[str, Any],
    nogrammar: dict[str, Any],
    metadata: dict[str, dict[str, Any]],
    min_tasks: int,
    bootstrap_samples: int,
    seed: int,
) -> list[PairedComparison]:
    groups: dict[str, set[str]] = defaultdict(set)
    for task_id, meta in metadata.items():
        groups[f"prefix:{meta['prefix']}"].add(task_id)
        for tag in meta.get("tags", []):
            if tag in {"cuda", "torch"}:
                continue
            groups[f"tag:{tag}"].add(task_id)

    comparisons: list[PairedComparison] = []
    for group_index, (group_name, task_ids) in enumerate(sorted(groups.items())):
        common_ids = set(constrained["summary_by_task"]) & set(nogrammar["summary_by_task"]) & task_ids
        if len(common_ids) < min_tasks:
            continue
        constrained_subset = subset_kagbench_run(constrained, common_ids)
        nogrammar_subset = subset_kagbench_run(nogrammar, common_ids)
        for metric_index, metric in enumerate(["public", "hidden"]):
            comparison = kagbench_common_task_comparison(
                name=group_name,
                metric=metric,
                treatment=constrained_subset,
                baseline=nogrammar_subset,
                bootstrap_samples=bootstrap_samples,
                seed=seed + group_index * 10 + metric_index,
            )
            if comparison is not None:
                comparisons.append(comparison)
    return benjamini_hochberg(comparisons)


def tritonbench_families(entry_file: str) -> set[str]:
    stem = entry_file.lower().removesuffix(".py")
    families: set[str] = set()
    if any(token in stem for token in ["softmax", "cross_entropy", "logsumexp"]):
        families.add("softmax/logsumexp")
    if any(token in stem for token in ["matmul", "mm", "bmm", "dot", "mv", "gemm"]):
        families.add("matmul/dot")
    if "conv" in stem:
        families.add("conv")
    if any(token in stem for token in ["svd", "eig", "lu", "qr", "det", "cholesky", "inverse", "pinv", "solve"]):
        families.add("linalg-decomp")
    if any(token in stem for token in ["sum", "mean", "max", "min", "norm", "prod", "argmax", "argmin"]):
        families.add("reduction/norm")
    if any(token in stem for token in ["relu", "gelu", "sigmoid", "tanh", "silu", "softplus", "elu", "mish"]):
        families.add("activation")
    if "pool" in stem:
        families.add("pooling")
    if any(token in stem for token in ["dropout", "rand", "bernoulli"]):
        families.add("random/dropout")
    if any(
        token in stem
        for token in [
            "add",
            "sub",
            "mul",
            "div",
            "sqrt",
            "abs",
            "cos",
            "sin",
            "exp",
            "log",
            "pow",
            "clamp",
            "where",
            "trunc",
            "ceil",
            "floor",
        ]
    ):
        families.add("elementwise-ish")
    if not families:
        families.add("other")
    return families


def build_tritonbench_family_comparisons(
    *,
    treatment_label: str,
    treatment_rows: list[dict[str, Any]],
    baseline_label: str,
    baseline_rows: list[dict[str, Any]],
    min_tasks: int,
    bootstrap_samples: int,
    seed: int,
) -> list[PairedComparison]:
    common_files = {
        row["entry_file"] for row in treatment_rows if row.get("entry_file")
    } & {row["entry_file"] for row in baseline_rows if row.get("entry_file")}
    groups: dict[str, set[str]] = defaultdict(set)
    for entry_file in common_files:
        for family in tritonbench_families(entry_file):
            groups[family].add(entry_file)

    comparisons: list[PairedComparison] = []
    for group_index, (family, files) in enumerate(sorted(groups.items())):
        if len(files) < min_tasks:
            continue
        treatment_subset = [row for row in treatment_rows if row.get("entry_file") in files]
        baseline_subset = [row for row in baseline_rows if row.get("entry_file") in files]
        for metric_index, metric in enumerate(["call", "exe"]):
            comparisons.append(
                paired_comparison(
                    name=family,
                    metric=metric,
                    treatment_label=treatment_label,
                    treatment_rows=treatment_subset,
                    baseline_label=baseline_label,
                    baseline_rows=baseline_subset,
                    bootstrap_samples=bootstrap_samples,
                    seed=seed + group_index * 10 + metric_index,
                )
            )
    return benjamini_hochberg(comparisons)


def kagbench_discordant_tasks(constrained: dict[str, Any], nogrammar: dict[str, Any]) -> dict[str, list[str]]:
    constrained_by_task = constrained["summary_by_task"]
    nogrammar_by_task = nogrammar["summary_by_task"]
    common_tasks = sorted(set(constrained_by_task) & set(nogrammar_by_task))
    result: dict[str, list[str]] = {
        "public_constrained_only": [],
        "public_nogrammar_only": [],
        "hidden_constrained_only": [],
        "hidden_nogrammar_only": [],
    }
    for task_id in common_tasks:
        constrained_public = public_passed(constrained_by_task[task_id])
        nogrammar_public = public_passed(nogrammar_by_task[task_id])
        constrained_hidden = hidden_passed(constrained_by_task[task_id])
        nogrammar_hidden = hidden_passed(nogrammar_by_task[task_id])
        if constrained_public and not nogrammar_public:
            result["public_constrained_only"].append(task_id)
        if nogrammar_public and not constrained_public:
            result["public_nogrammar_only"].append(task_id)
        if constrained_hidden and not nogrammar_hidden:
            result["hidden_constrained_only"].append(task_id)
        if nogrammar_hidden and not constrained_hidden:
            result["hidden_nogrammar_only"].append(task_id)
    return result


def stats_to_jsonable(value: Any) -> Any:
    if isinstance(value, RateSummary):
        return value.__dict__
    if isinstance(value, PairedComparison):
        return value.__dict__
    if isinstance(value, dict):
        return {str(key): stats_to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [stats_to_jsonable(item) for item in value]
    return value


def top_exploratory_comparisons(
    comparisons: list[PairedComparison], *, top_n: int
) -> list[PairedComparison]:
    return sorted(
        comparisons,
        key=lambda comparison: (
            -abs(comparison.diff),
            comparison.q_value if comparison.q_value is not None else 1.0,
            comparison.name,
            comparison.metric,
        ),
    )[:top_n]


def fdr_summary(comparisons: list[PairedComparison]) -> str:
    if not comparisons:
        return "no tests"
    q_values = [comparison.q_value for comparison in comparisons if comparison.q_value is not None]
    significant_05 = sum(1 for q_value in q_values if q_value < 0.05)
    significant_10 = sum(1 for q_value in q_values if q_value < 0.10)
    min_q = min(q_values) if q_values else None
    return (
        f"{len(comparisons)} tests; min BH q={format_p(min_q)}; "
        f"q<0.05: {significant_05}; q<0.10: {significant_10}"
    )


def append_comparison_table(
    lines: list[str],
    comparisons: list[PairedComparison],
    *,
    treatment_header: str = "Treatment",
    baseline_header: str = "Baseline",
) -> None:
    lines.append(
        f"| Group / comparison | Metric | n | {treatment_header} | {baseline_header} | "
        "Δ treatment-baseline | Improved / regressed | McNemar p | BH q |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for comparison in comparisons:
        lines.append(
            "| "
            f"{comparison.name} | {comparison.metric} | {comparison.common_n} | "
            f"{comparison.treatment_successes} | {comparison.baseline_successes} | "
            f"{signed_pct(comparison.diff)} "
            f"[{signed_pct(comparison.ci_low)}, {signed_pct(comparison.ci_high)}] | "
            f"{comparison.improved} / {comparison.regressed} | "
            f"{format_p(comparison.p_value)} | {format_p(comparison.q_value)} |"
        )


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# KernelForge final statistics")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    lines.append("")

    lines.append("## TritonBench-T headline metrics")
    lines.append("")
    lines.append(
        "| Model / run | Rows | Unique tasks | call@1 | exe@1 | Median speedup | "
        "Speed rows | exe unknown | ref failed |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    tb_summaries = report["tritonbench"]["summaries"]
    ordered_labels = [
        label for label in TRITONBENCH_LABEL_ORDER if label in tb_summaries
    ] + sorted(label for label in tb_summaries if label not in TRITONBENCH_LABEL_ORDER)
    for label in ordered_labels:
        summary = tb_summaries[label]
        speedup = summary.get("speedup") or {}
        speedup_text = f"{speedup.get('median', 0.0):.2f}x" if speedup else "n/a"
        speedup_count = speedup.get("count", 0) if speedup else 0
        lines.append(
            "| "
            f"{label} | {summary['total']} | {summary['unique_files']} | "
            f"{format_rate(summary['call'])} | {format_rate(summary['exe'])} | "
            f"{speedup_text} | {speedup_count} | "
            f"{summary['exe_unknown']} | {summary['ref_failed']} |"
        )
    lines.append("")

    speed_report = report["tritonbench"].get("speed_benchmarks") or {}
    if speed_report.get("enabled"):
        lines.append(
            "Speed benchmarks use `triton.testing.do_bench` on correctness-passing kernels "
            f"with warmup={speed_report['warmup']} and rep={speed_report['rep']}; "
            f"eligible={speed_report['eligible']}, measured={speed_report['measured']}, "
            f"cached={speed_report['cached']}, failed={speed_report['failed']}."
        )
        lines.append("")

    lines.append("## TritonBench-T paired primary comparison")
    lines.append("")
    tb_comparisons = report["tritonbench"]["comparisons"]
    if tb_comparisons:
        lines.append(
            "| Comparison | Metric | n | Treatment | Baseline | Δ treatment-baseline | "
            "Improved / regressed | McNemar p | BH q |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for comparison in tb_comparisons:
            lines.append(
                "| "
                f"{comparison.name} | {comparison.metric} | {comparison.common_n} | "
                f"{comparison.treatment_successes} | {comparison.baseline_successes} | "
                f"{signed_pct(comparison.diff)} "
                f"[{signed_pct(comparison.ci_low)}, {signed_pct(comparison.ci_high)}] | "
                f"{comparison.improved} / {comparison.regressed} | "
                f"{format_p(comparison.p_value)} | {format_p(comparison.q_value)} |"
            )
    else:
        lines.append("No paired TritonBench comparison could be computed.")
    lines.append("")

    lines.append("## KAGBench workflow metrics")
    lines.append("")
    lines.append(
        "| Run | Complete? | Valid env? | Summaries | Public pass | Hidden pass | "
        "Median speedup | Speed rows | Hidden / public-pass | Status counts | libstdc++ ImportErrors |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|")
    for run in report["kagbench"]["runs"]:
        status_counts = ", ".join(
            f"{key}={value}" for key, value in sorted(run["status_counts"].items())
        )
        speedup = run.get("speedup") or {}
        speedup_text = f"{speedup.get('median', 0.0):.2f}x" if speedup else "n/a"
        speedup_count = speedup.get("count", 0) if speedup else 0
        lines.append(
            "| "
            f"{run['label']} | {run['complete']} | {not run['invalid_environment']} | "
            f"{len(run['summaries'])}/{run['expected_tasks']} | "
            f"{format_rate(run['public'])} | {format_rate(run['hidden'])} | "
            f"{speedup_text} | {speedup_count} | "
            f"{format_rate(run['hidden_conditional'])} | {status_counts} | "
            f"{run['libstdcxx_import_errors']} |"
        )
    lines.append("")

    kag_speed_report = report["kagbench"].get("speed_benchmarks") or {}
    if kag_speed_report.get("enabled"):
        lines.append(
            "KAGBench speed benchmarks use `triton.testing.do_bench` on hidden-passing final "
            f"candidates with warmup={kag_speed_report['warmup']} and rep={kag_speed_report['rep']}; "
            f"eligible={kag_speed_report['eligible']}, measured={kag_speed_report['measured']}, "
            f"cached={kag_speed_report['cached']}, failed={kag_speed_report['failed']}."
        )
        lines.append("")

    lines.append("## KAGBench paired comparison")
    lines.append("")
    kagbench_comparisons = report["kagbench"]["comparisons"]
    if kagbench_comparisons:
        lines.append(
            "| Comparison | Metric | n | Treatment | Baseline | Δ treatment-baseline | "
            "Improved / regressed | McNemar p | BH q |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for comparison in kagbench_comparisons:
            lines.append(
                "| "
                f"{comparison.name} | {comparison.metric} | {comparison.common_n} | "
                f"{comparison.treatment_successes} | {comparison.baseline_successes} | "
                f"{signed_pct(comparison.diff)} "
                f"[{signed_pct(comparison.ci_low)}, {signed_pct(comparison.ci_high)}] | "
                f"{comparison.improved} / {comparison.regressed} | "
                f"{format_p(comparison.p_value)} | {format_p(comparison.q_value)} |"
            )
    else:
        lines.append(
            "Skipped: no-grammar KAGBench is currently incomplete and/or contaminated by "
            "the local libstdc++ ImportError."
        )
    lines.append("")

    clean_subset = report["kagbench"].get("clean_subset")
    clean_subset_comparisons = report["kagbench"].get("clean_subset_comparisons") or []
    if clean_subset:
        lines.append("## KAGBench final no-grammar shard provenance")
        lines.append("")
        status_counts = ", ".join(
            f"{key}={value}" for key, value in sorted(clean_subset["status_counts"].items())
        )
        source_counts = "; ".join(
            f"{Path(key).name}={value}"
            for key, value in sorted(clean_subset["source_counts"].items())
        )
        lines.append(
            f"Clean selected summaries: {len(clean_subset['summaries'])} tasks; "
            f"status counts: {status_counts}; sources: {source_counts}."
        )
        lines.append("")
        if clean_subset_comparisons:
            lines.append(
                "| Comparison | Metric | n | No grammar clean | Constrained | "
                "Δ no-grammar-constrained | Improved / regressed | McNemar p | BH q |"
            )
            lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
            for comparison in clean_subset_comparisons:
                lines.append(
                    "| "
                    f"{comparison.name} | {comparison.metric} | {comparison.common_n} | "
                    f"{comparison.treatment_successes} | {comparison.baseline_successes} | "
                    f"{signed_pct(comparison.diff)} "
                    f"[{signed_pct(comparison.ci_low)}, {signed_pct(comparison.ci_high)}] | "
                    f"{comparison.improved} / {comparison.regressed} | "
                    f"{format_p(comparison.p_value)} | {format_p(comparison.q_value)} |"
                )
        else:
            lines.append("No clean-subset paired comparison could be computed yet.")
        lines.append("")

    discordants = report["kagbench"].get("discordant_tasks") or {}
    if discordants:
        lines.append("## KAGBench discordant tasks")
        lines.append("")
        lines.append(
            "These are the task-level flips behind the paired KAGBench headline numbers."
        )
        lines.append("")
        for key, tasks in discordants.items():
            rendered_tasks = ", ".join(tasks) if tasks else "none"
            lines.append(f"- `{key}` ({len(tasks)}): {rendered_tasks}")
        lines.append("")

    exploratory_top_n = int(report.get("exploratory_top_n", 10))
    kagbench_subgroups = report["kagbench"].get("subgroup_comparisons") or []
    tritonbench_families = report["tritonbench"].get("family_comparisons") or []
    if kagbench_subgroups or tritonbench_families:
        lines.append("## Exploratory subgroup checks with FDR correction")
        lines.append("")
        lines.append(
            "These are hypothesis-generating subgroup scans. BH q-values are adjusted within "
            "each scan family; showing the largest apparent effects by absolute Δ."
        )
        lines.append("")
        if kagbench_subgroups:
            lines.append(
                "### KAGBench tags/prefixes: constrained decoding vs no grammar"
            )
            lines.append("")
            lines.append(
                f"FDR summary: {fdr_summary(kagbench_subgroups)}. "
                "Treatment is constrained decoding; baseline is no grammar."
            )
            lines.append("")
            append_comparison_table(
                lines,
                top_exploratory_comparisons(
                    kagbench_subgroups, top_n=exploratory_top_n
                ),
                treatment_header="Constrained",
                baseline_header="No grammar",
            )
            lines.append("")
        if tritonbench_families:
            lines.append(
                "### TritonBench filename families: LLGuidance vs no-grammar Gemma 4 E4B"
            )
            lines.append("")
            lines.append(
                f"FDR summary: {fdr_summary(tritonbench_families)}. "
                "Treatment is llama.cpp + LLGuidance; baseline is Modal vLLM no grammar."
            )
            lines.append("")
            append_comparison_table(
                lines,
                top_exploratory_comparisons(
                    tritonbench_families, top_n=exploratory_top_n
                ),
                treatment_header="LLGuidance",
                baseline_header="No grammar",
            )
            lines.append("")

    lines.append("## Generation ledger sanity checks")
    lines.append("")
    lines.append(
        "| Ledger | Rows | Unique tasks | Status | Finish reasons | Median latency | "
        "Median completion tokens | Max completion tokens | Total tokens |"
    )
    lines.append("|---|---:|---:|---|---|---:|---:|---:|---:|")
    for ledger in report["generation_ledgers"]:
        if ledger.get("missing"):
            lines.append(
                f"| {ledger['path']} | missing | missing | missing | missing | "
                "n/a | n/a | n/a | n/a |"
            )
            continue
        latency = ledger.get("latency_s") or {}
        completion = ledger.get("completion_tokens") or {}
        total_tokens = ledger.get("total_tokens") or {}
        lines.append(
            "| "
            f"{ledger['path']} | {ledger['rows']} | {ledger['unique_files']} | "
            f"{ledger['status']} | {ledger['finish_reason']} | "
            f"{latency.get('median', 0.0):.2f}s | "
            f"{completion.get('median', 0.0):.0f} | "
            f"{completion.get('max', 0.0):.0f} | "
            f"{total_tokens.get('sum', 0.0):.0f} |"
        )
    lines.append("")

    warnings = report["warnings"]
    if warnings:
        lines.append("## Warnings and caveats")
        lines.append("")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo_root.resolve()
    warnings: list[str] = []

    tritonbench_eval_sets, tb_warnings = load_tritonbench_eval_sets(root)
    warnings.extend(tb_warnings)
    if args.skip_tritonbench_speed_benchmarks:
        tritonbench_speed_benchmarks = {"enabled": False}
    else:
        tritonbench_speed_benchmarks = benchmark_tritonbench_speedups(
            root,
            tritonbench_eval_sets,
            args,
            warnings,
        )
    tritonbench_summaries = {
        label: summarize_tritonbench_eval(rows) for label, rows in tritonbench_eval_sets.items()
    }

    tb_comparisons: list[PairedComparison] = []
    baseline_label = "Modal Gemma 4 E4B vLLM"
    treatment_label = "Gemma 4 E4B llama.cpp + LLGuidance"
    if baseline_label in tritonbench_eval_sets and treatment_label in tritonbench_eval_sets:
        for index, metric in enumerate(["call", "exe"]):
            tb_comparisons.append(
                paired_comparison(
                    name="LLGuidance vs no-grammar Gemma 4 E4B",
                    metric=metric,
                    treatment_label=treatment_label,
                    treatment_rows=tritonbench_eval_sets[treatment_label],
                    baseline_label=baseline_label,
                    baseline_rows=tritonbench_eval_sets[baseline_label],
                    bootstrap_samples=args.bootstrap_samples,
                    seed=args.seed + index,
                )
            )
    else:
        warnings.append(
            "Could not compute TritonBench LLGuidance-vs-no-grammar paired comparison; "
            "one or both eval sets are missing."
        )
    tb_comparisons = benjamini_hochberg(tb_comparisons)
    tritonbench_family_comparisons = []
    if baseline_label in tritonbench_eval_sets and treatment_label in tritonbench_eval_sets:
        tritonbench_family_comparisons = build_tritonbench_family_comparisons(
            treatment_label=treatment_label,
            treatment_rows=tritonbench_eval_sets[treatment_label],
            baseline_label=baseline_label,
            baseline_rows=tritonbench_eval_sets[baseline_label],
            min_tasks=args.min_subgroup_tasks,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed + 300,
        )

    constrained_dirs = args.kagbench_constrained_dir or DEFAULT_KAGBENCH_CONSTRAINED_DIRS
    nogrammar_dirs = args.kagbench_nogrammar_dir or DEFAULT_KAGBENCH_NOGRAMMAR_DIRS
    kagbench_constrained = load_kagbench_stitch(
        root,
        "KAGBench constrained (found: llama.cpp GBNF)",
        constrained_dirs,
    )
    kagbench_nogrammar = load_kagbench_stitch(
        root,
        "KAGBench no grammar (final clean)",
        nogrammar_dirs,
    )
    clean_nogrammar_dirs = (
        args.kagbench_clean_nogrammar_dir or DEFAULT_KAGBENCH_NOGRAMMAR_CLEAN_REPLACEMENT_DIRS
    )
    kagbench_nogrammar_clean = load_kagbench_stitch(
        root,
        "KAGBench no grammar clean Nix set",
        clean_nogrammar_dirs,
    )
    if args.skip_kagbench_speed_benchmarks:
        kagbench_speed_benchmarks = {"enabled": False}
        for run in [kagbench_constrained, kagbench_nogrammar, kagbench_nogrammar_clean]:
            run.pop("candidate_by_source_id", None)
    else:
        kagbench_speed_benchmarks = benchmark_kagbench_speedups(
            root,
            [kagbench_constrained, kagbench_nogrammar],
            args,
            warnings,
        )
        kagbench_nogrammar_clean.pop("candidate_by_source_id", None)
    constrained_backends = kagbench_constrained["grammar_backends"]
    if constrained_backends and set(constrained_backends) != {"xgrammar"}:
        warnings.append(
            "The complete constrained KAGBench run found by the script is grammar-constrained, "
            f"but manifests report grammar_backend={constrained_backends}, not XGrammar."
        )
    for run in [kagbench_constrained, kagbench_nogrammar]:
        warnings.extend(run["warnings"])
        if not run["complete"]:
            warnings.append(
                f"{run['label']} is incomplete: "
                f"{len(run['summaries'])}/{run['expected_tasks']} summaries; "
                f"missing={run['missing_tasks']}"
            )
        if run["invalid_environment"]:
            warnings.append(
                f"{run['label']} has {run['libstdcxx_import_errors']} owned eval rows with "
                f"{LIBSTDCXX_IMPORT_FRAGMENT} ImportError; do not use it for final comparison."
            )
    if kagbench_nogrammar_clean["invalid_environment"]:
        warnings.append(
            "The clean no-grammar replacement subset unexpectedly still contains "
            f"{LIBSTDCXX_IMPORT_FRAGMENT} ImportErrors."
        )

    kagbench_comparisons = []
    if (
        kagbench_constrained["complete"]
        and kagbench_nogrammar["complete"]
        and not kagbench_constrained["invalid_environment"]
        and not kagbench_nogrammar["invalid_environment"]
    ):
        for index, metric in enumerate(["public", "hidden"]):
            comparison = kagbench_pair_comparison(
                metric=metric,
                treatment=kagbench_constrained,
                baseline=kagbench_nogrammar,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + 100 + index,
            )
            if comparison is not None:
                kagbench_comparisons.append(comparison)
        kagbench_comparisons = benjamini_hochberg(kagbench_comparisons)

    clean_subset_comparisons = []
    if not kagbench_nogrammar_clean["invalid_environment"]:
        for index, metric in enumerate(["public", "hidden"]):
            comparison = kagbench_common_task_comparison(
                name="Clean no-grammar subset vs constrained",
                metric=metric,
                treatment=kagbench_nogrammar_clean,
                baseline=kagbench_constrained,
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + 200 + index,
            )
            if comparison is not None:
                clean_subset_comparisons.append(comparison)
        clean_subset_comparisons = benjamini_hochberg(clean_subset_comparisons)

    kagbench_metadata = load_kagbench_task_metadata(root)
    kagbench_subgroup_comparisons = build_kagbench_subgroup_comparisons(
        constrained=kagbench_constrained,
        nogrammar=kagbench_nogrammar,
        metadata=kagbench_metadata,
        min_tasks=args.min_subgroup_tasks,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + 400,
    )
    kagbench_discordants = kagbench_discordant_tasks(kagbench_constrained, kagbench_nogrammar)

    generation_ledgers = [
        summarize_generation_ledger(resolve_path(root, TRITONBENCH_MODAL_GEMMA_LEDGER)),
        summarize_generation_ledger(resolve_path(root, TRITONBENCH_LLGUIDANCE_LEDGER)),
    ]
    manifest_path = resolve_path(root, TRITONBENCH_LLGUIDANCE_MANIFEST)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        models = manifest.get("models") or []
        if "google/gemma-4-E2B-it" in models:
            warnings.append(
                "LLGuidance manifest labels the model as google/gemma-4-E2B-it, but "
                "the server log showed gemma-4-E4B-it-UD-Q8_K_XL.gguf. The report uses "
                "the corrected E4B presentation label."
            )

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "tritonbench": {
            "summaries": tritonbench_summaries,
            "comparisons": tb_comparisons,
            "family_comparisons": tritonbench_family_comparisons,
            "speed_benchmarks": tritonbench_speed_benchmarks,
        },
        "kagbench": {
            "runs": [kagbench_constrained, kagbench_nogrammar],
            "comparisons": kagbench_comparisons,
            "clean_subset": kagbench_nogrammar_clean,
            "clean_subset_comparisons": clean_subset_comparisons,
            "subgroup_comparisons": kagbench_subgroup_comparisons,
            "discordant_tasks": kagbench_discordants,
            "speed_benchmarks": kagbench_speed_benchmarks,
        },
        "generation_ledgers": generation_ledgers,
        "exploratory_top_n": args.exploratory_top_n,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate KernelForge final run statistics into presentation-ready Markdown."
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-subgroup-tasks", type=int, default=5)
    parser.add_argument("--exploratory-top-n", type=int, default=10)
    parser.add_argument(
        "--skip-tritonbench-speed-benchmarks",
        action="store_true",
        help="Do not run missing TritonBench speed benchmarks before aggregating stats.",
    )
    parser.add_argument(
        "--skip-kagbench-speed-benchmarks",
        action="store_true",
        help="Do not run missing KAGBench speed benchmarks before aggregating stats.",
    )
    parser.add_argument("--speed-benchmark-cache", type=Path, default=TRITONBENCH_SPEED_CACHE)
    parser.add_argument("--kagbench-speed-cache", type=Path, default=KAGBENCH_SPEED_CACHE)
    parser.add_argument("--speed-benchmark-warmup", type=int, default=25)
    parser.add_argument("--speed-benchmark-rep", type=int, default=100)
    parser.add_argument("--speed-benchmark-timeout", type=int, default=180)
    parser.add_argument(
        "--speed-benchmark-limit",
        type=int,
        help="Only run this many uncached speed benchmarks; useful for smoke tests.",
    )
    parser.add_argument(
        "--kagbench-speed-limit",
        type=int,
        help="Only run this many uncached KAGBench speed benchmarks; useful for smoke tests.",
    )
    parser.add_argument(
        "--kagbench-constrained-dir",
        type=Path,
        action="append",
        help="Override constrained KAGBench shard dirs. Repeat to pass multiple shards.",
    )
    parser.add_argument(
        "--kagbench-nogrammar-dir",
        type=Path,
        action="append",
        help="Override no-grammar KAGBench shard dirs. Repeat to pass multiple shards.",
    )
    parser.add_argument(
        "--kagbench-clean-nogrammar-dir",
        type=Path,
        action="append",
        help="Override clean no-grammar replacement shard dirs for initial subset comparisons.",
    )
    parser.add_argument("--markdown-out", type=Path, help="Optional path to write Markdown report.")
    parser.add_argument("--json-out", type=Path, help="Optional path to write machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    markdown = render_markdown(report)
    print(markdown)

    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown + "\n", encoding="utf-8")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(stats_to_jsonable(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
