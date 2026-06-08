"""Modal evaluator for KAGBench candidates.

This is the GPU-backed evaluator used by `kernelforge.agent` when
`--eval-backend modal` is selected. It mirrors the local KAGBench evaluator but
runs inside a CUDA Modal container.

Example:
    uv run modal run scripts/modal_kagbench_eval.py \
      --task-id tritonbench_g/vector_addition \
      --candidate-file /tmp/candidate.py \
      --phase public
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

eval_image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("gcc")
    .pip_install("triton", "numpy")
    .add_local_dir("src/kernelforge", remote_path="/app/src/kernelforge")
    .add_local_dir("benchmarks/KAGBench", remote_path="/app/benchmarks/KAGBench")
)

app = modal.App("kernelforge-kagbench-eval")


@app.function(image=eval_image, gpu="T4", timeout=300)
def evaluate_kagbench_candidate(
    candidate_code: str,
    *,
    task_id: str = "",
    entry_file: str = "",
    candidate_id: str = "candidate",
    phase: str = "public",
    timeout_s: int = 180,
    run_id: str = "modal",
) -> dict:
    sys.path.insert(0, "/app/src")

    from kernelforge.agent.evaluator import evaluate_task_local
    from kernelforge.agent.tasks import load_kagbench, select_tasks

    if phase not in {"public", "hidden"}:
        return {
            "schema_version": 1,
            "kind": "candidate_evaluation",
            "run_id": run_id,
            "task_id": task_id,
            "entry_file": entry_file,
            "candidate_id": candidate_id,
            "phase": phase,
            "backend": "modal",
            "call@1": False,
            "exe@1": False,
            "passed": False,
            "mismatches": [],
            "error_type": "InvalidPhase",
            "error_message": f"unsupported phase: {phase}",
            "stdout": "",
            "stderr": "",
            "latency_s": 0.0,
        }

    tasks = load_kagbench("/app/benchmarks/KAGBench/kagbench.jsonl")
    try:
        selected = select_tasks(
            tasks,
            task_ids=[task_id] if task_id else None,
            entry_files=[entry_file] if entry_file else None,
            limit=1,
        )
    except KeyError:
        selected = []
    if not selected:
        return {
            "schema_version": 1,
            "kind": "candidate_evaluation",
            "run_id": run_id,
            "task_id": task_id,
            "entry_file": entry_file,
            "candidate_id": candidate_id,
            "phase": phase,
            "backend": "modal",
            "call@1": False,
            "exe@1": False,
            "passed": False,
            "mismatches": [],
            "error_type": "TaskNotFound",
            "error_message": "No KAGBench task matched task_id or entry_file",
            "stdout": "",
            "stderr": "",
            "latency_s": 0.0,
        }

    result = evaluate_task_local(
        selected[0],
        candidate_code=candidate_code,
        candidate_id=candidate_id,
        run_id=run_id,
        phase=phase,  # type: ignore[arg-type]
        timeout_s=timeout_s,
    ).to_row()
    result["backend"] = "modal"
    return result


@app.local_entrypoint()
def main(
    candidate_file: str,
    task_id: str = "",
    entry_file: str = "",
    candidate_id: str = "candidate",
    phase: str = "public",
    timeout_s: int = 180,
    run_id: str = "modal",
    out: str = "",
):
    candidate_code = Path(candidate_file).read_text(encoding="utf-8")
    result = evaluate_kagbench_candidate.remote(
        candidate_code,
        task_id=task_id,
        entry_file=entry_file,
        candidate_id=candidate_id,
        phase=phase,
        timeout_s=timeout_s,
        run_id=run_id,
    )
    if out:
        Path(out).write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result))
