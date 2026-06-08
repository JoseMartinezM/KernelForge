from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Literal

from .schemas import EvaluationResult, KAGBenchTask

EvalPhase = Literal["public", "hidden"]
RESULT_PREFIX = "__KERNELFORGE_EVAL_RESULT__"


def _test_function_name(phase: EvalPhase) -> str:
    return "public_tests" if phase == "public" else "unit_tests"


def _evaluation_script(phase: EvalPhase) -> str:
    test_function_name = _test_function_name(phase)
    return f"""
import importlib
import json
import traceback

RESULT_PREFIX = {RESULT_PREFIX!r}


def emit(result):
    print(RESULT_PREFIX + json.dumps(result), flush=True)


try:
    import torch
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    candidate = importlib.import_module("candidate")
    tests_module = importlib.import_module("tests_module")
    test_fn = getattr(tests_module, {test_function_name!r})
    test_fn(candidate)
except AssertionError as exc:
    message = str(exc) or "assertion failed"
    emit({{
        "call@1": True,
        "exe@1": False,
        "passed": False,
        "error_type": "AssertionError",
        "error_message": message,
        "mismatches": [message],
        "traceback": traceback.format_exc(),
    }})
except BaseException as exc:
    emit({{
        "call@1": False,
        "exe@1": False,
        "passed": False,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "mismatches": [],
        "traceback": traceback.format_exc(),
    }})
else:
    emit({{
        "call@1": True,
        "exe@1": True,
        "passed": True,
        "error_type": None,
        "error_message": None,
        "mismatches": [],
        "traceback": "",
    }})
"""


def _parse_result(stdout: str) -> dict[str, object] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            payload = line[len(RESULT_PREFIX) :]
            parsed = json.loads(payload)
            return parsed if isinstance(parsed, dict) else None
    return None


def _parse_last_json_object(stdout: str) -> dict[str, object] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


def evaluate_task_local(
    task: KAGBenchTask,
    *,
    candidate_code: str,
    candidate_id: str,
    run_id: str,
    phase: EvalPhase = "public",
    timeout_s: int = 180,
) -> EvaluationResult:
    """Evaluate one candidate by running KAGBench tests in a subprocess."""
    tests_code = task.public_tests if phase == "public" else task.unit_tests
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"kernelforge_{task.entry_file}_") as workdir:
        workdir_path = Path(workdir)
        (workdir_path / "candidate.py").write_text(candidate_code, encoding="utf-8")
        (workdir_path / "pytorch_reference.py").write_text(
            task.pytorch_reference, encoding="utf-8"
        )
        (workdir_path / "tests_module.py").write_text(tests_code, encoding="utf-8")
        (workdir_path / "run_eval.py").write_text(_evaluation_script(phase), encoding="utf-8")

        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(workdir_path)
            if not existing_pythonpath
            else f"{workdir_path}{os.pathsep}{existing_pythonpath}"
        )
        try:
            run = subprocess.run(
                [sys.executable, str(workdir_path / "run_eval.py")],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return EvaluationResult(
                run_id=run_id,
                task_id=task.task_id,
                entry_file=task.entry_file,
                candidate_id=candidate_id,
                phase=phase,
                backend="local",
                call_at_1=False,
                exe_at_1=False,
                passed=False,
                mismatches=[],
                error_type="TimeoutExpired",
                error_message=f"timed out after {timeout_s}s",
                stdout=exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or ""),
                stderr=exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or ""),
                latency_s=time.monotonic() - start,
            )

    parsed = _parse_result(run.stdout)
    if parsed is None:
        return EvaluationResult(
            run_id=run_id,
            task_id=task.task_id,
            entry_file=task.entry_file,
            candidate_id=candidate_id,
            phase=phase,
            backend="local",
            call_at_1=False,
            exe_at_1=False,
            passed=False,
            mismatches=[],
            error_type="NoEvaluationResult",
            error_message=f"subprocess exited {run.returncode} without result marker",
            stdout=run.stdout,
            stderr=run.stderr,
            latency_s=time.monotonic() - start,
        )

    mismatches = parsed.get("mismatches")
    error_type_value = parsed.get("error_type")
    error_message_value = parsed.get("error_message")
    error_type = error_type_value if isinstance(error_type_value, str) else None
    error_message = error_message_value if isinstance(error_message_value, str) else None
    return EvaluationResult(
        run_id=run_id,
        task_id=task.task_id,
        entry_file=task.entry_file,
        candidate_id=candidate_id,
        phase=phase,
        backend="local",
        call_at_1=bool(parsed.get("call@1")),
        exe_at_1=bool(parsed.get("exe@1")) if parsed.get("exe@1") is not None else None,
        passed=bool(parsed.get("passed")),
        mismatches=[str(item) for item in mismatches] if isinstance(mismatches, list) else [],
        error_type=error_type,
        error_message=error_message,
        stdout=run.stdout,
        stderr="\n".join(part for part in [run.stderr, str(parsed.get("traceback") or "")] if part),
        latency_s=time.monotonic() - start,
    )


def evaluate_task(
    task: KAGBenchTask,
    *,
    candidate_code: str,
    candidate_id: str,
    run_id: str,
    phase: EvalPhase = "public",
    backend: str = "local",
    timeout_s: int = 180,
) -> EvaluationResult:
    """Evaluate one candidate with the requested backend."""
    if backend == "local":
        return evaluate_task_local(
            task,
            candidate_code=candidate_code,
            candidate_id=candidate_id,
            run_id=run_id,
            phase=phase,
            timeout_s=timeout_s,
        )
    if backend == "modal":
        return evaluate_task_modal(
            task,
            candidate_code=candidate_code,
            candidate_id=candidate_id,
            run_id=run_id,
            phase=phase,
            timeout_s=timeout_s,
        )
    raise ValueError(f"unsupported evaluation backend: {backend}")


def evaluate_task_modal(
    task: KAGBenchTask,
    *,
    candidate_code: str,
    candidate_id: str,
    run_id: str,
    phase: EvalPhase = "public",
    timeout_s: int = 180,
) -> EvaluationResult:
    """Evaluate one candidate through the Modal KAGBench evaluation entrypoint."""
    start = time.monotonic()
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as handle:
        handle.write(candidate_code)
        candidate_path = Path(handle.name)
    result_path = candidate_path.with_suffix(".result.json")

    try:
        try:
            run = subprocess.run(
                [
                    "uv",
                    "run",
                    "modal",
                    "run",
                    "scripts/modal_kagbench_eval.py",
                    "--task-id",
                    task.task_id,
                    "--candidate-file",
                    str(candidate_path),
                    "--candidate-id",
                    candidate_id,
                    "--phase",
                    phase,
                    "--timeout-s",
                    str(timeout_s),
                    "--run-id",
                    run_id,
                    "--out",
                    str(result_path),
                ],
                capture_output=True,
                text=True,
                timeout=max(timeout_s + 300, 600),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return EvaluationResult(
                run_id=run_id,
                task_id=task.task_id,
                entry_file=task.entry_file,
                candidate_id=candidate_id,
                phase=phase,
                backend="modal",
                call_at_1=False,
                exe_at_1=False,
                passed=False,
                mismatches=[],
                error_type="TimeoutExpired",
                error_message=f"modal command timed out after {max(timeout_s + 300, 600)}s",
                stdout=(
                    exc.stdout.decode(errors="replace")
                    if isinstance(exc.stdout, bytes)
                    else (exc.stdout or "")
                ),
                stderr=(
                    exc.stderr.decode(errors="replace")
                    if isinstance(exc.stderr, bytes)
                    else (exc.stderr or "")
                ),
                latency_s=time.monotonic() - start,
            )
    finally:
        try:
            candidate_path.unlink()
        except FileNotFoundError:
            pass

    parsed = None
    if result_path.exists():
        try:
            parsed = json.loads(result_path.read_text(encoding="utf-8"))
        finally:
            result_path.unlink(missing_ok=True)
    if not isinstance(parsed, dict):
        parsed = _parse_last_json_object(run.stdout)
    if run.returncode != 0 or parsed is None:
        return EvaluationResult(
            run_id=run_id,
            task_id=task.task_id,
            entry_file=task.entry_file,
            candidate_id=candidate_id,
            phase=phase,
            backend="modal",
            call_at_1=False,
            exe_at_1=False,
            passed=False,
            mismatches=[],
            error_type="ModalCommandFailed" if run.returncode != 0 else "NoEvaluationResult",
            error_message=f"modal command exited {run.returncode}",
            stdout=run.stdout,
            stderr=run.stderr,
            latency_s=time.monotonic() - start,
        )

    mismatches = parsed.get("mismatches")
    error_type_value = parsed.get("error_type")
    error_message_value = parsed.get("error_message")
    call_value = parsed.get("call@1", parsed.get("call_at_1"))
    exe_value = parsed.get("exe@1", parsed.get("exe_at_1"))
    return EvaluationResult(
        run_id=run_id,
        task_id=task.task_id,
        entry_file=task.entry_file,
        candidate_id=candidate_id,
        phase=phase,
        backend="modal",
        call_at_1=bool(call_value),
        exe_at_1=bool(exe_value) if exe_value is not None else None,
        passed=bool(parsed.get("passed")),
        mismatches=[str(item) for item in mismatches] if isinstance(mismatches, list) else [],
        error_type=error_type_value if isinstance(error_type_value, str) else None,
        error_message=error_message_value if isinstance(error_message_value, str) else None,
        stdout=run.stdout,
        stderr=run.stderr,
        latency_s=time.monotonic() - start,
    )
