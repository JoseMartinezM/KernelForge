from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .evaluator import evaluate_task
from .grammar import grammar_extra_body, redacted_grammar_extra_body
from .ledgers import AgentRunLedger
from .llm import call_model, merge_generation
from .prompts import (
    build_direct_implementer_messages,
    build_implementer_messages,
    build_repair_messages,
    build_teacher_messages,
    parse_repair_directive,
    parse_teacher_plan,
)
from .schemas import (
    CandidateGeneration,
    KAGBenchTask,
    RepairDirective,
    StaticCheckResult,
    TeacherPlan,
    WorkflowConfig,
    WorkflowSummary,
)
from .static_checks import run_static_checks
from kernelforge.benchmark.tritonbench import cleanup_generated_code


def _dict_or_empty(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    messages = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if isinstance(role, str) and isinstance(content, str):
            messages.append({"role": role, "content": content})
    return messages


def _static_check_from_row(value: Any) -> StaticCheckResult:
    data = _dict_or_empty(value)
    return StaticCheckResult(
        syntax_ok=bool(data.get("syntax_ok")),
        syntax_error=data.get("syntax_error") if isinstance(data.get("syntax_error"), str) else None,
        content_chars=int(data.get("content_chars") or 0),
        content_lines=int(data.get("content_lines") or 0),
        starts_with_import=bool(data.get("starts_with_import")),
        markdown_fence_count=int(data.get("markdown_fence_count") or 0),
        triton_jit_count=int(data.get("triton_jit_count") or 0),
        torch_call_count=int(data.get("torch_call_count") or 0),
        torch_calls=_str_list(data.get("torch_calls")),
        semantic_warnings=_str_list(data.get("semantic_warnings")),
        flags=_str_list(data.get("flags")),
        flags_text=str(data.get("flags_text") or ""),
    )


class GenerationReuseCache:
    """Read-only cache for replaying useful generations from an earlier run."""

    def __init__(self, source: str | Path) -> None:
        self.source = Path(source)
        generation_path = self.source / "generation.jsonl" if self.source.is_dir() else self.source
        if not generation_path.exists():
            raise FileNotFoundError(f"generation reuse ledger not found: {generation_path}")
        self.generation_path = generation_path
        self._teacher_plan_rows: dict[str, dict[str, Any]] = {}
        self._candidate_rows: dict[str, list[dict[str, Any]]] = {}
        with generation_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {generation_path}:{line_number}: {exc}") from exc
                if not isinstance(row, dict):
                    continue
                task_id = row.get("task_id")
                if not isinstance(task_id, str):
                    continue
                kind = row.get("kind")
                if kind == "teacher_plan" and task_id not in self._teacher_plan_rows:
                    self._teacher_plan_rows[task_id] = row
                elif kind == "candidate_generation":
                    self._candidate_rows.setdefault(task_id, []).append(row)

    def teacher_plan(self, task: KAGBenchTask, *, run_id: str) -> TeacherPlan | None:
        row = self._teacher_plan_rows.get(task.task_id)
        if row is None:
            return None
        model = row.get("model") or row.get("teacher_model")
        if not isinstance(model, str):
            return None
        return TeacherPlan(
            run_id=run_id,
            task_id=task.task_id,
            entry_file=task.entry_file,
            model=model,
            content=str(row.get("content") or ""),
            sections=_dict_or_empty(row.get("sections")),
            parse_warnings=_str_list(row.get("parse_warnings")),
            response=_dict_or_empty(row.get("response")),
            usage=_dict_or_empty(row.get("usage")),
            latency_s=float(row.get("latency_s") or 0.0),
            messages=_messages(row.get("messages")),
            created_at=str(row.get("created_at") or ""),
        )

    def evaluatable_candidates(
        self,
        task: KAGBenchTask,
        *,
        run_id: str,
    ) -> list[CandidateGeneration]:
        rows = sorted(
            self._candidate_rows.get(task.task_id, []),
            key=lambda row: (
                int(row.get("attempt") or 0),
                int(row.get("candidate_index") or 0),
                str(row.get("candidate_id") or ""),
            ),
        )
        candidates = []
        seen_candidate_ids: set[str] = set()
        for row in rows:
            candidate = self._candidate_from_row(row, task=task, run_id=run_id)
            if not candidate.static.evaluatable:
                continue
            if candidate.candidate_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(candidate.candidate_id)
            candidates.append(candidate)
        return candidates

    def _candidate_from_row(
        self,
        row: dict[str, Any],
        *,
        task: KAGBenchTask,
        run_id: str,
    ) -> CandidateGeneration:
        model = row.get("model") or row.get("implementer_model")
        repair_id = row.get("repair_id")
        return CandidateGeneration(
            run_id=run_id,
            task_id=task.task_id,
            entry_file=task.entry_file,
            plan_id=str(row.get("plan_id") or ""),
            attempt=int(row.get("attempt") or 0),
            candidate_index=int(row.get("candidate_index") or 0),
            model=str(model or ""),
            content=str(row.get("content") or ""),
            raw_content=str(row.get("raw_content") or row.get("content") or ""),
            finish_reason=(
                row.get("finish_reason") if isinstance(row.get("finish_reason"), str) else None
            ),
            usage=_dict_or_empty(row.get("usage")),
            response=_dict_or_empty(row.get("response")),
            latency_s=float(row.get("latency_s") or 0.0),
            generation=_dict_or_empty(row.get("generation")),
            static=_static_check_from_row(row.get("static")),
            messages=_messages(row.get("messages")),
            repair_id=repair_id if isinstance(repair_id, str) else None,
            created_at=str(row.get("created_at") or ""),
        )


class KernelGenerationWorkflow:
    """Deterministic teacher-plan -> implementer -> evaluation workflow."""

    def __init__(self, config: WorkflowConfig) -> None:
        if config.mode not in {"full-agent", "direct-best-of-k"}:
            raise ValueError(f"unsupported workflow mode: {config.mode}")
        if config.candidates_per_attempt < 1:
            raise ValueError("candidates_per_attempt must be >= 1")
        if config.repair_candidates_per_attempt is not None and config.repair_candidates_per_attempt < 1:
            raise ValueError("repair_candidates_per_attempt must be >= 1")
        if config.max_repairs < 0:
            raise ValueError("max_repairs must be >= 0")
        self.config = config
        self.ledger = AgentRunLedger(config.out_dir)
        self.reuse_cache = (
            GenerationReuseCache(config.reuse_generations_from)
            if config.reuse_generations_from is not None
            else None
        )

    def dry_run_payload(self, tasks: list[KAGBenchTask]) -> dict[str, Any]:
        """Return prompts and redacted generation config without calling providers."""
        previews = []
        for task in tasks:
            teacher_messages = build_teacher_messages(
                task, include_public_tests=self.config.include_public_tests
            )
            placeholder_plan = (
                "<kernel_plan>\n"
                "<pattern>teacher plan placeholder</pattern>\n"
                "<api_contract>teacher plan placeholder</api_contract>\n"
                "<program_mapping>teacher plan placeholder</program_mapping>\n"
                "<numerics>teacher plan placeholder</numerics>\n"
                "<non_defaults>none</non_defaults>\n"
                "<risks>teacher plan placeholder</risks>\n"
                "</kernel_plan>"
            )
            implementer_messages = build_implementer_messages(
                task,
                teacher_plan=placeholder_plan,
                include_public_tests=self.config.include_public_tests,
            )
            preview = {
                "task_id": task.task_id,
                "entry_file": task.entry_file,
                "teacher_messages": teacher_messages,
                "implementer_messages": implementer_messages,
            }
            if self.config.mode == "direct-best-of-k":
                preview["teacher_messages"] = []
                preview["implementer_messages"] = build_direct_implementer_messages(
                    task,
                    include_public_tests=self.config.include_public_tests,
                )
            previews.append(
                preview
            )
        implementer_generation = dict(self.config.implementer_generation)
        redacted_extra_body = redacted_grammar_extra_body(
            self.config.grammar_file,
            backend=self.config.grammar_backend,
        )
        if redacted_extra_body is not None:
            implementer_generation = merge_generation(
                implementer_generation, extra_body=redacted_extra_body
            )
        return {
            "run_id": self.config.run_id,
            "mode": self.config.mode,
            "teacher_model": self.config.teacher_model,
            "implementer_model": self.config.implementer_model,
            "candidates_per_attempt": self.config.candidates_per_attempt,
            "max_repairs": self.config.max_repairs,
            "eval_backend": self.config.eval_backend,
            "teacher_generation": self.config.teacher_generation,
            "implementer_generation": implementer_generation,
            "tasks": previews,
            "repair_candidates_per_attempt": self.config.repair_candidates_per_attempt,
        }

    def run(self, tasks: list[KAGBenchTask]) -> list[WorkflowSummary]:
        """Run the workflow for each selected task and return final summaries."""
        self.ledger.write_manifest(self.config, tasks)
        summaries = [self.run_task(task) for task in tasks]
        return summaries

    def run_task(self, task: KAGBenchTask) -> WorkflowSummary:
        if self.config.mode == "direct-best-of-k":
            return self._run_direct_best_of_k_task(task)

        plan = self._make_teacher_plan(task)
        repair: RepairDirective | None = None
        all_candidates: list[CandidateGeneration] = []
        public_results_by_candidate: dict[str, Any] = {}
        final_candidate: CandidateGeneration | None = None
        hidden_passed: bool | None = None
        status = "generated"
        start_attempt = 0

        reused_candidates = self._reused_evaluatable_candidates(task)
        if reused_candidates and self.config.eval_backend == "none":
            all_candidates.extend(reused_candidates)
            final_candidate = reused_candidates[0]
        elif reused_candidates:
            all_candidates.extend(reused_candidates)
            reused_public_results = []
            for candidate in reused_candidates:
                result = evaluate_task(
                    task,
                    candidate_code=candidate.content,
                    candidate_id=candidate.candidate_id,
                    run_id=self.config.run_id,
                    phase="public",
                    backend=self.config.eval_backend,
                    timeout_s=self.config.eval_timeout_s,
                )
                reused_public_results.append(result)
                public_results_by_candidate[candidate.candidate_id] = result
                self.ledger.append_evaluation(result.to_row())

            passing = [result for result in reused_public_results if result.passed]
            if passing:
                winning_result = passing[0]
                final_candidate = next(
                    candidate
                    for candidate in reused_candidates
                    if candidate.candidate_id == winning_result.candidate_id
                )
                status = "public_pass"
            elif self.config.max_repairs > 0 and reused_public_results:
                repair = self._make_repair_directive(
                    task,
                    plan=plan,
                    failed_evaluations=reused_public_results,
                    failed_candidates=reused_candidates,
                    attempt=1,
                )
                start_attempt = 1
            else:
                start_attempt = self.config.max_repairs + 1

        for attempt in range(start_attempt, self.config.max_repairs + 1):
            if final_candidate is not None:
                break
            attempt_candidates = self._make_attempt_candidates(
                task,
                plan=plan,
                attempt=attempt,
                repair=repair,
            )
            all_candidates.extend(attempt_candidates)

            if self.config.eval_backend == "none":
                final_candidate = attempt_candidates[0] if attempt_candidates else None
                status = "generated"
                break

            public_results = []
            for candidate in attempt_candidates:
                if not candidate.static.evaluatable:
                    continue
                result = evaluate_task(
                    task,
                    candidate_code=candidate.content,
                    candidate_id=candidate.candidate_id,
                    run_id=self.config.run_id,
                    phase="public",
                    backend=self.config.eval_backend,
                    timeout_s=self.config.eval_timeout_s,
                )
                public_results.append(result)
                public_results_by_candidate[candidate.candidate_id] = result
                self.ledger.append_evaluation(result.to_row())

            passing = [result for result in public_results if result.passed]
            if passing:
                winning_result = passing[0]
                final_candidate = next(
                    candidate
                    for candidate in all_candidates
                    if candidate.candidate_id == winning_result.candidate_id
                )
                status = "public_pass"
                break

            if attempt < self.config.max_repairs and public_results:
                repair = self._make_repair_directive(
                    task,
                    plan=plan,
                    failed_evaluations=public_results,
                    failed_candidates=attempt_candidates,
                    attempt=attempt + 1,
                )

        public_passed = None
        if final_candidate and final_candidate.candidate_id in public_results_by_candidate:
            public_passed = bool(public_results_by_candidate[final_candidate.candidate_id].passed)

        if (
            final_candidate is not None
            and self.config.eval_backend != "none"
            and self.config.hidden_eval
            and public_passed
        ):
            hidden_result = evaluate_task(
                task,
                candidate_code=final_candidate.content,
                candidate_id=final_candidate.candidate_id,
                run_id=self.config.run_id,
                phase="hidden",
                backend=self.config.eval_backend,
                timeout_s=self.config.eval_timeout_s,
            )
            self.ledger.append_evaluation(hidden_result.to_row())
            hidden_passed = hidden_result.passed
            status = "hidden_pass" if hidden_result.passed else "hidden_fail"
        elif self.config.eval_backend != "none" and not public_passed:
            status = "public_fail"

        summary = WorkflowSummary(
            run_id=self.config.run_id,
            task_id=task.task_id,
            entry_file=task.entry_file,
            status=status,
            teacher_model=self.config.teacher_model,
            implementer_model=self.config.implementer_model,
            attempts=max((candidate.attempt for candidate in all_candidates), default=0) + 1,
            candidates_generated=len(all_candidates),
            final_candidate_id=final_candidate.candidate_id if final_candidate else None,
            public_passed=public_passed,
            hidden_passed=hidden_passed,
            plan_id=plan.plan_id,
            ledger_paths=self.ledger.paths,
        )
        self.ledger.append_summary(summary.to_row())
        return summary

    def _run_direct_best_of_k_task(self, task: KAGBenchTask) -> WorkflowSummary:
        all_candidates = self._make_direct_candidates(task)
        public_results_by_candidate: dict[str, Any] = {}
        final_candidate: CandidateGeneration | None = None
        hidden_passed: bool | None = None
        status = "generated"

        if self.config.eval_backend == "none":
            final_candidate = all_candidates[0] if all_candidates else None
        else:
            public_results = []
            for candidate in all_candidates:
                if not candidate.static.evaluatable:
                    continue
                result = evaluate_task(
                    task,
                    candidate_code=candidate.content,
                    candidate_id=candidate.candidate_id,
                    run_id=self.config.run_id,
                    phase="public",
                    backend=self.config.eval_backend,
                    timeout_s=self.config.eval_timeout_s,
                )
                public_results.append(result)
                public_results_by_candidate[candidate.candidate_id] = result
                self.ledger.append_evaluation(result.to_row())

            passing = [result for result in public_results if result.passed]
            if passing:
                winning_result = passing[0]
                final_candidate = next(
                    candidate
                    for candidate in all_candidates
                    if candidate.candidate_id == winning_result.candidate_id
                )
                status = "public_pass"

        public_passed = None
        if final_candidate and final_candidate.candidate_id in public_results_by_candidate:
            public_passed = bool(public_results_by_candidate[final_candidate.candidate_id].passed)

        if (
            final_candidate is not None
            and self.config.eval_backend != "none"
            and self.config.hidden_eval
            and public_passed
        ):
            hidden_result = evaluate_task(
                task,
                candidate_code=final_candidate.content,
                candidate_id=final_candidate.candidate_id,
                run_id=self.config.run_id,
                phase="hidden",
                backend=self.config.eval_backend,
                timeout_s=self.config.eval_timeout_s,
            )
            self.ledger.append_evaluation(hidden_result.to_row())
            hidden_passed = hidden_result.passed
            status = "hidden_pass" if hidden_result.passed else "hidden_fail"
        elif self.config.eval_backend != "none" and not public_passed:
            status = "public_fail"

        summary = WorkflowSummary(
            run_id=self.config.run_id,
            task_id=task.task_id,
            entry_file=task.entry_file,
            status=status,
            teacher_model=self.config.teacher_model,
            implementer_model=self.config.implementer_model,
            attempts=1,
            candidates_generated=len(all_candidates),
            final_candidate_id=final_candidate.candidate_id if final_candidate else None,
            public_passed=public_passed,
            hidden_passed=hidden_passed,
            plan_id=None,
            ledger_paths=self.ledger.paths,
        )
        self.ledger.append_summary(summary.to_row())
        return summary

    def _make_teacher_plan(self, task: KAGBenchTask) -> TeacherPlan:
        if self.reuse_cache is not None:
            plan = self.reuse_cache.teacher_plan(task, run_id=self.config.run_id)
            if plan is not None:
                self.ledger.append_generation(plan.to_row())
                return plan

        messages = build_teacher_messages(task, include_public_tests=self.config.include_public_tests)
        result, _generation = call_model(
            self.config.teacher_model,
            messages,
            config_path=self.config.config_path,
            generation_overrides=self.config.teacher_generation,
        )
        sections, parse_warnings = parse_teacher_plan(result.content)
        plan = TeacherPlan(
            run_id=self.config.run_id,
            task_id=task.task_id,
            entry_file=task.entry_file,
            model=self.config.teacher_model,
            content=result.content,
            sections=sections,
            parse_warnings=parse_warnings,
            response=result.response,
            usage=result.usage,
            latency_s=result.latency_s,
            messages=messages,
        )
        self.ledger.append_generation(plan.to_row())
        return plan

    def _reused_evaluatable_candidates(self, task: KAGBenchTask) -> list[CandidateGeneration]:
        if self.reuse_cache is None:
            return []
        candidates = self.reuse_cache.evaluatable_candidates(task, run_id=self.config.run_id)
        for candidate in candidates:
            self.ledger.append_generation(candidate.to_row())
        return candidates

    def _make_attempt_candidates(
        self,
        task: KAGBenchTask,
        *,
        plan: TeacherPlan,
        attempt: int,
        repair: RepairDirective | None,
    ) -> list[CandidateGeneration]:
        """Generate all candidates for one attempt concurrently."""
        candidates_per_attempt = (
            self.config.repair_candidates_per_attempt
            if attempt > 0 and self.config.repair_candidates_per_attempt is not None
            else self.config.candidates_per_attempt
        )
        if candidates_per_attempt == 1:
            return [
                self._make_candidate(
                    task,
                    plan=plan,
                    attempt=attempt,
                    candidate_index=0,
                    repair=repair,
                )
            ]

        with ThreadPoolExecutor(max_workers=candidates_per_attempt) as executor:
            futures = [
                executor.submit(
                    self._make_candidate,
                    task,
                    plan=plan,
                    attempt=attempt,
                    candidate_index=index,
                    repair=repair,
                )
                for index in range(candidates_per_attempt)
            ]
            return [future.result() for future in futures]

    def _make_direct_candidates(self, task: KAGBenchTask) -> list[CandidateGeneration]:
        """Generate the direct best-of-K ablation candidates concurrently."""
        if self.config.candidates_per_attempt == 1:
            return [self._make_direct_candidate(task, candidate_index=0)]

        with ThreadPoolExecutor(max_workers=self.config.candidates_per_attempt) as executor:
            futures = [
                executor.submit(
                    self._make_direct_candidate,
                    task,
                    candidate_index=index,
                )
                for index in range(self.config.candidates_per_attempt)
            ]
            return [future.result() for future in futures]

    def _make_direct_candidate(
        self,
        task: KAGBenchTask,
        *,
        candidate_index: int,
    ) -> CandidateGeneration:
        messages = build_direct_implementer_messages(
            task,
            include_public_tests=self.config.include_public_tests,
        )
        extra_body = (
            grammar_extra_body(self.config.grammar_file, backend=self.config.grammar_backend)
            if self.config.grammar_file
            else None
        )
        result, generation = call_model(
            self.config.implementer_model,
            messages,
            config_path=self.config.config_path,
            generation_overrides=self.config.implementer_generation,
            extra_body=extra_body,
        )
        content = cleanup_generated_code(result.content)
        static = run_static_checks(content)
        if self.config.grammar_file:
            generation = dict(generation)
            redacted_extra_body = redacted_grammar_extra_body(
                self.config.grammar_file,
                backend=self.config.grammar_backend,
            )
            if redacted_extra_body is not None:
                generation = merge_generation(generation, extra_body=redacted_extra_body)
        candidate = CandidateGeneration(
            run_id=self.config.run_id,
            task_id=task.task_id,
            entry_file=task.entry_file,
            plan_id="direct-best-of-k",
            attempt=0,
            candidate_index=candidate_index,
            model=self.config.implementer_model,
            content=content,
            raw_content=result.content,
            finish_reason=result.finish_reason,
            usage=result.usage,
            response=result.response,
            latency_s=result.latency_s,
            generation=generation,
            static=static,
            messages=messages,
        )
        self.ledger.append_generation(candidate.to_row())
        return candidate

    def _make_candidate(
        self,
        task: KAGBenchTask,
        *,
        plan: TeacherPlan,
        attempt: int,
        candidate_index: int,
        repair: RepairDirective | None,
    ) -> CandidateGeneration:
        messages = build_implementer_messages(
            task,
            teacher_plan=plan.content,
            repair_directive=repair.content if repair is not None else None,
            include_public_tests=self.config.include_public_tests,
        )
        extra_body = (
            grammar_extra_body(self.config.grammar_file, backend=self.config.grammar_backend)
            if self.config.grammar_file
            else None
        )
        result, generation = call_model(
            self.config.implementer_model,
            messages,
            config_path=self.config.config_path,
            generation_overrides=self.config.implementer_generation,
            extra_body=extra_body,
        )
        content = cleanup_generated_code(result.content)
        static = run_static_checks(content)
        if self.config.grammar_file:
            generation = dict(generation)
            redacted_extra_body = redacted_grammar_extra_body(
                self.config.grammar_file,
                backend=self.config.grammar_backend,
            )
            if redacted_extra_body is not None:
                generation = merge_generation(generation, extra_body=redacted_extra_body)
        candidate = CandidateGeneration(
            run_id=self.config.run_id,
            task_id=task.task_id,
            entry_file=task.entry_file,
            plan_id=plan.plan_id,
            attempt=attempt,
            candidate_index=candidate_index,
            model=self.config.implementer_model,
            content=content,
            raw_content=result.content,
            finish_reason=result.finish_reason,
            usage=result.usage,
            response=result.response,
            latency_s=result.latency_s,
            generation=generation,
            static=static,
            messages=messages,
            repair_id=repair.repair_id if repair is not None else None,
        )
        self.ledger.append_generation(candidate.to_row())
        return candidate

    def _make_repair_directive(
        self,
        task: KAGBenchTask,
        *,
        plan: TeacherPlan,
        failed_evaluations: list[Any],
        failed_candidates: list[CandidateGeneration],
        attempt: int,
    ) -> RepairDirective:
        messages = build_repair_messages(
            task,
            teacher_plan=plan.content,
            failed_evaluations=failed_evaluations,
            failed_candidates=failed_candidates,
            include_public_tests=self.config.include_public_tests,
        )
        result, _generation = call_model(
            self.config.teacher_model,
            messages,
            config_path=self.config.config_path,
            generation_overrides=self.config.teacher_generation,
        )
        sections, parse_warnings = parse_repair_directive(result.content)
        repair = RepairDirective(
            run_id=self.config.run_id,
            task_id=task.task_id,
            entry_file=task.entry_file,
            model=self.config.teacher_model,
            attempt=attempt,
            content=result.content,
            sections=sections,
            parse_warnings=parse_warnings,
            response=result.response,
            usage=result.usage,
            latency_s=result.latency_s,
            messages=messages,
        )
        self.ledger.append_generation(repair.to_row())
        return repair
