from __future__ import annotations

from typing import Any

from .evaluator import evaluate_task
from .grammar import grammar_extra_body, redacted_grammar_extra_body
from .ledgers import AgentRunLedger
from .llm import call_model, merge_generation
from .prompts import (
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
    TeacherPlan,
    WorkflowConfig,
    WorkflowSummary,
)
from .static_checks import run_static_checks
from kernelforge.benchmark.tritonbench import cleanup_generated_code


class KernelGenerationWorkflow:
    """Deterministic teacher-plan -> implementer -> evaluation workflow."""

    def __init__(self, config: WorkflowConfig) -> None:
        if config.candidates_per_attempt < 1:
            raise ValueError("candidates_per_attempt must be >= 1")
        if config.max_repairs < 0:
            raise ValueError("max_repairs must be >= 0")
        self.config = config
        self.ledger = AgentRunLedger(config.out_dir)

    def dry_run_payload(self, tasks: list[KAGBenchTask]) -> dict[str, Any]:
        """Return prompts and redacted generation config without calling providers."""
        previews = []
        for task in tasks:
            teacher_messages = build_teacher_messages(
                task, include_public_tests=self.config.include_public_tests
            )
            placeholder_plan = "<kernel_plan>\n<api>teacher plan placeholder</api>\n</kernel_plan>"
            implementer_messages = build_implementer_messages(
                task,
                teacher_plan=placeholder_plan,
                include_public_tests=self.config.include_public_tests,
            )
            previews.append(
                {
                    "task_id": task.task_id,
                    "entry_file": task.entry_file,
                    "teacher_messages": teacher_messages,
                    "implementer_messages": implementer_messages,
                }
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
            "teacher_model": self.config.teacher_model,
            "implementer_model": self.config.implementer_model,
            "candidates_per_attempt": self.config.candidates_per_attempt,
            "max_repairs": self.config.max_repairs,
            "eval_backend": self.config.eval_backend,
            "teacher_generation": self.config.teacher_generation,
            "implementer_generation": implementer_generation,
            "tasks": previews,
        }

    def run(self, tasks: list[KAGBenchTask]) -> list[WorkflowSummary]:
        """Run the workflow for each selected task and return final summaries."""
        self.ledger.write_manifest(self.config, tasks)
        summaries = [self.run_task(task) for task in tasks]
        return summaries

    def run_task(self, task: KAGBenchTask) -> WorkflowSummary:
        plan = self._make_teacher_plan(task)
        repair: RepairDirective | None = None
        all_candidates: list[CandidateGeneration] = []
        public_results_by_candidate: dict[str, Any] = {}
        final_candidate: CandidateGeneration | None = None
        hidden_passed: bool | None = None
        status = "generated"

        for attempt in range(self.config.max_repairs + 1):
            attempt_candidates = [
                self._make_candidate(
                    task,
                    plan=plan,
                    attempt=attempt,
                    candidate_index=index,
                    repair=repair,
                )
                for index in range(self.config.candidates_per_attempt)
            ]
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

    def _make_teacher_plan(self, task: KAGBenchTask) -> TeacherPlan:
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
