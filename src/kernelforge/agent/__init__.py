"""KernelForge agent workflow scaffolding.

The package contains the reusable Python side of the final teacher-plan plus
Gemma/xgrammar kernel generation workflow. Thin agent adapters should invoke the
CLI exposed by :mod:`kernelforge.agent.__main__` rather than reimplementing the
orchestration logic.
"""

from .schemas import (
    CandidateGeneration,
    EvaluationResult,
    KAGBenchTask,
    StaticCheckResult,
    TeacherPlan,
    WorkflowConfig,
    WorkflowSummary,
)

__all__ = [
    "CandidateGeneration",
    "EvaluationResult",
    "KAGBenchTask",
    "StaticCheckResult",
    "TeacherPlan",
    "WorkflowConfig",
    "WorkflowSummary",
]
