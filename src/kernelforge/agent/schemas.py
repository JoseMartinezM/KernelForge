from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_iso() -> str:
    """Return an ISO-8601 UTC timestamp for run ledgers."""
    return datetime.now(timezone.utc).isoformat()


def text_hash(value: str) -> str:
    """Hash text exactly as stored in a ledger."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json_hash(value: Any) -> str:
    """Hash JSON-compatible data with stable key ordering."""
    data = json.dumps(to_jsonable(value), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses and Paths into plain JSON-compatible values."""
    if is_dataclass(value) and not isinstance(value, type):
        return to_jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class KAGBenchTask:
    """One KAGBench synthesis task.

    `unit_tests` and source fields are intentionally kept off the prompt-facing
    payload helpers so the workflow can preserve public/hidden separation.
    """

    task_id: str
    entry_file: str
    prompt: str
    pytorch_reference: str
    public_tests: str
    unit_tests: str
    tags: tuple[str, ...] = ()
    case_dir: str | None = None
    source_file: str | None = None
    vendored_source: str | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "KAGBenchTask":
        task_id = row.get("id")
        entry_file = row.get("entry_file")
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("KAGBench row must define a non-empty string 'id'")
        if not isinstance(entry_file, str) or not entry_file:
            raise ValueError(f"KAGBench row {task_id!r} must define 'entry_file'")

        def required_text(name: str) -> str:
            value = row.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"KAGBench row {task_id!r} must define non-empty {name!r}")
            return value

        tags = row.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError(f"KAGBench row {task_id!r} field 'tags' must be a list of strings")

        return cls(
            task_id=task_id,
            entry_file=entry_file,
            prompt=required_text("prompt"),
            pytorch_reference=required_text("pytorch_reference"),
            public_tests=required_text("public_tests"),
            unit_tests=required_text("unit_tests"),
            tags=tuple(tags),
            case_dir=row.get("case_dir") if isinstance(row.get("case_dir"), str) else None,
            source_file=(
                row.get("source_file") if isinstance(row.get("source_file"), str) else None
            ),
            vendored_source=(
                row.get("vendored_source")
                if isinstance(row.get("vendored_source"), str)
                else None
            ),
        )

    def prompt_payload(self, *, include_public_tests: bool = True) -> dict[str, str]:
        """Return the generation-visible task payload.

        Hidden unit tests, source paths, and vendored source are deliberately not
        included here.
        """
        payload = {
            "task_id": self.task_id,
            "entry_file": self.entry_file,
            "task_prompt": self.prompt,
            "pytorch_reference": self.pytorch_reference,
        }
        if include_public_tests:
            payload["public_tests"] = self.public_tests
        return payload


@dataclass(frozen=True)
class LLMCallResult:
    """Normalized result from one OpenAI-compatible chat completion call."""

    model: str
    content: str
    response: dict[str, Any]
    usage: dict[str, Any]
    finish_reason: str | None
    latency_s: float


@dataclass(frozen=True)
class TeacherPlan:
    """Tagged-prose plan produced by the teacher model."""

    run_id: str
    task_id: str
    entry_file: str
    model: str
    content: str
    sections: dict[str, str]
    parse_warnings: list[str]
    response: dict[str, Any]
    usage: dict[str, Any]
    latency_s: float
    messages: list[dict[str, str]]
    created_at: str = field(default_factory=now_iso)

    @property
    def plan_id(self) -> str:
        return text_hash(f"{self.task_id}\n{self.model}\n{self.content}")

    def to_row(self) -> dict[str, Any]:
        row = to_jsonable(self)
        row.update(
            {
                "schema_version": 1,
                "kind": "teacher_plan",
                "plan_id": self.plan_id,
                "teacher_model": self.model,
            }
        )
        return row


@dataclass(frozen=True)
class RepairDirective:
    """Tagged-prose repair guidance produced after public-test failures."""

    run_id: str
    task_id: str
    entry_file: str
    model: str
    attempt: int
    content: str
    sections: dict[str, str]
    parse_warnings: list[str]
    response: dict[str, Any]
    usage: dict[str, Any]
    latency_s: float
    messages: list[dict[str, str]]
    created_at: str = field(default_factory=now_iso)

    @property
    def repair_id(self) -> str:
        return text_hash(f"{self.task_id}\n{self.model}\n{self.attempt}\n{self.content}")

    def to_row(self) -> dict[str, Any]:
        row = to_jsonable(self)
        row.update(
            {
                "schema_version": 1,
                "kind": "repair_directive",
                "repair_id": self.repair_id,
                "teacher_model": self.model,
            }
        )
        return row


@dataclass(frozen=True)
class StaticCheckResult:
    """CPU-only syntax, semantic, and anti-cheat signals for candidate code."""

    syntax_ok: bool
    syntax_error: str | None
    content_chars: int
    content_lines: int
    starts_with_import: bool
    markdown_fence_count: int
    triton_jit_count: int
    torch_call_count: int
    torch_calls: list[str]
    semantic_warnings: list[str]
    flags: list[str]
    flags_text: str

    @property
    def evaluatable(self) -> bool:
        return self.syntax_ok and self.content_chars > 0


@dataclass(frozen=True)
class CandidateGeneration:
    """One implementer-produced candidate module."""

    run_id: str
    task_id: str
    entry_file: str
    plan_id: str
    attempt: int
    candidate_index: int
    model: str
    content: str
    raw_content: str
    finish_reason: str | None
    usage: dict[str, Any]
    response: dict[str, Any]
    latency_s: float
    generation: dict[str, Any]
    static: StaticCheckResult
    messages: list[dict[str, str]]
    repair_id: str | None = None
    created_at: str = field(default_factory=now_iso)

    @property
    def candidate_id(self) -> str:
        return text_hash(
            "\n".join(
                [
                    self.task_id,
                    self.plan_id,
                    str(self.attempt),
                    str(self.candidate_index),
                    self.model,
                    self.content,
                ]
            )
        )

    def to_row(self) -> dict[str, Any]:
        row = to_jsonable(self)
        row.update(
            {
                "schema_version": 1,
                "kind": "candidate_generation",
                "candidate_id": self.candidate_id,
                "implementer_model": self.model,
            }
        )
        return row


@dataclass(frozen=True)
class EvaluationResult:
    """Public or hidden evaluation result for one candidate."""

    run_id: str
    task_id: str
    entry_file: str
    candidate_id: str
    phase: str
    backend: str
    call_at_1: bool
    exe_at_1: bool | None
    passed: bool
    mismatches: list[str]
    error_type: str | None
    error_message: str | None
    stdout: str
    stderr: str
    latency_s: float
    created_at: str = field(default_factory=now_iso)

    def to_row(self) -> dict[str, Any]:
        row = to_jsonable(self)
        row.update(
            {
                "schema_version": 1,
                "kind": "candidate_evaluation",
                "call@1": self.call_at_1,
                "exe@1": self.exe_at_1,
            }
        )
        return row


@dataclass(frozen=True)
class WorkflowConfig:
    """Configuration for one deterministic agent workflow run."""

    teacher_model: str
    implementer_model: str
    out_dir: Path
    run_id: str = field(default_factory=lambda: now_iso().replace(":", "").replace("+", "Z"))
    candidates_per_attempt: int = 1
    max_repairs: int = 0
    eval_backend: str = "none"
    eval_timeout_s: int = 180
    hidden_eval: bool = True
    include_public_tests: bool = True
    grammar_file: Path | None = Path("grammar/triton.gbnf")
    grammar_backend: str = "xgrammar"
    config_path: Path = Path("src/kernelforge/benchmark/llm_models.json")
    teacher_generation: dict[str, Any] = field(default_factory=dict)
    implementer_generation: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowSummary:
    """One visualization-friendly final row per task."""

    run_id: str
    task_id: str
    entry_file: str
    status: str
    teacher_model: str
    implementer_model: str
    attempts: int
    candidates_generated: int
    final_candidate_id: str | None
    public_passed: bool | None
    hidden_passed: bool | None
    plan_id: str | None
    ledger_paths: dict[str, str]
    created_at: str = field(default_factory=now_iso)

    def to_row(self) -> dict[str, Any]:
        row = to_jsonable(self)
        row.update({"schema_version": 1, "kind": "workflow_summary"})
        return row
