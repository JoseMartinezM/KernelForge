from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .grammar import grammar_metadata
from .schemas import KAGBenchTask, WorkflowConfig, now_iso, to_jsonable


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one row to a JSONL ledger."""
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")


class AgentRunLedger:
    """Path manager and writer for one agent workflow run directory."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.manifest_path = self.out_dir / "manifest.json"
        self.generation_path = self.out_dir / "generation.jsonl"
        self.evaluation_path = self.out_dir / "evaluation.jsonl"
        self.summary_path = self.out_dir / "summary.jsonl"

    @property
    def paths(self) -> dict[str, str]:
        return {
            "manifest": str(self.manifest_path),
            "generation": str(self.generation_path),
            "evaluation": str(self.evaluation_path),
            "summary": str(self.summary_path),
        }

    def write_manifest(self, config: WorkflowConfig, tasks: Iterable[KAGBenchTask]) -> Path:
        """Write run-level metadata before executing the workflow."""
        task_list = list(tasks)
        manifest = {
            "schema_version": 1,
            "kind": "agent_workflow_manifest",
            "created_at": now_iso(),
            "run_id": config.run_id,
            "out_dir": str(self.out_dir),
            "teacher_model": config.teacher_model,
            "implementer_model": config.implementer_model,
            "candidates_per_attempt": config.candidates_per_attempt,
            "max_repairs": config.max_repairs,
            "eval_backend": config.eval_backend,
            "eval_timeout_s": config.eval_timeout_s,
            "hidden_eval": config.hidden_eval,
            "include_public_tests": config.include_public_tests,
            "grammar": grammar_metadata(config.grammar_file),
            "grammar_backend": config.grammar_backend,
            "teacher_generation": config.teacher_generation,
            "implementer_generation": config.implementer_generation,
            "task_count": len(task_list),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "entry_file": task.entry_file,
                    "tags": list(task.tags),
                }
                for task in task_list
            ],
            "ledger_paths": self.paths,
        }
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(to_jsonable(manifest), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return self.manifest_path

    def append_generation(self, row: dict[str, Any]) -> None:
        append_jsonl(self.generation_path, row)

    def append_evaluation(self, row: dict[str, Any]) -> None:
        append_jsonl(self.evaluation_path, row)

    def append_summary(self, row: dict[str, Any]) -> None:
        append_jsonl(self.summary_path, row)
