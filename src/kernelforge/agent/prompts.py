from __future__ import annotations

import re
import textwrap
from typing import Iterable

from .schemas import CandidateGeneration, EvaluationResult, KAGBenchTask

TEACHER_PLAN_SECTIONS = (
    "pattern",
    "api_contract",
    "program_mapping",
    "numerics",
    "non_defaults",
    "risks",
)

REPAIR_SECTIONS = (
    "diagnosis",
    "required_changes",
    "avoid",
)

TEACHER_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a senior Triton kernel engineer. Your job is to produce a concise
    implementation plan for a kernel-generation model. Do not write source code.

    The downstream model will write Python/Triton code from your plan. Your plan
    should state the task-specific choices only. Use the default Triton contract
    unless the task prompt, reference, or public tests require a deviation.

    Default Triton contract:
    - Pass torch tensors directly to Triton kernels; do not pass data_ptr()
      integers, dtype objects, Python lists/tuples, or CUDA streams.
    - Leave pointer parameters in @triton.jit kernels unannotated. Triton
      specializes from tensor argument dtypes automatically.
    - Index pointers in tensor elements, not bytes.
    - Use tl.constexpr only for compile-time block sizes, tile sizes, and mode
      constants.
    - Wrapper code handles allocation, shape extraction, optional contiguity
      conversion when needed, and kernel launch. Do not add stricter validation
      than the reference behavior requires.
    - Prefer simple, standard idioms: 1D elementwise offsets; one-program-per-row
      reductions; 2D tiles with row/column masks.

    Use only these simple XML-style sections:
    <kernel_plan>
    <pattern>elementwise | rowwise_reduction | tiled_2d | embedding | other</pattern>
    <api_contract>Only shape, dtype, return, aliasing, and in-place requirements specific to this task.</api_contract>
    <program_mapping>Only the grid, tile, indexing, and mask details needed for this task.</program_mapping>
    <numerics>Only task-specific casts, reductions, epsilons, random behavior, or approximations.</numerics>
    <non_defaults>Only required deviations from the default Triton contract, or "none".</non_defaults>
    <risks>At most three task-specific pitfalls; omit generic Triton advice.</risks>
    </kernel_plan>

    Rules:
    - Do not include Python source code.
    - Do not include Markdown fences.
    - Be selectively verbose: for simple elementwise tasks, keep the plan under
      200 words; use more detail only when shapes, strides, reductions, RNG, or
      numerics make it necessary.
    - Mention only details inferable from the task prompt, reference, and public tests.
    - Do not over-generalize beyond the requested API contract. Prefer the
      simplest reliable mapping that satisfies the prompt and visible tests.
    """
)

IMPLEMENTER_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert GPU kernel engineer specializing in Triton.

    Write a complete Python module that satisfies the task and follows the
    provided teacher plan. Prefer the default Triton contract below; if the
    teacher plan conflicts with it without an explicit task requirement, ignore
    that conflicting part and use the default.

    Default Triton contract:
    - Pass torch tensors directly to kernels; never pass tensor.data_ptr(),
      torch.dtype objects, Python lists/tuples, or CUDA streams as kernel launch
      arguments.
    - Leave JIT pointer parameters unannotated; Triton infers pointer and dtype
      specialization from tensor arguments.
    - Index pointers in tensor elements, not bytes.
    - Use tl.constexpr for block/tile sizes and compile-time mode constants.
    - Keep wrapper code to allocation, shape extraction, necessary contiguity
      handling, and launch setup. Do not add validation stricter than the
      reference behavior.

    Mini-cookbook:
    - 1D elementwise: offsets = pid * BLOCK + tl.arange(0, BLOCK); mask = offsets < N.
    - Rowwise reduction: map rows or row tiles to programs; reduce a constexpr
      column block with tl.sum/tl.max/etc. and masks.
    - 2D tile: use row offsets[:, None], column offsets[None, :], and a combined
      bounds mask.

    Hard requirements:
    - Output only valid Python source code: no Markdown fences and no prose.
    - Include all required imports.
    - Preserve the exact public API requested by the task.
    - Use @triton.jit for the core GPU computation.
    - Access global memory through tl.load and tl.store inside JIT kernels.
    - Use tl.program_id and tl.arange for tiled program mapping.
    - Use mask= on vectorized tl.load and tl.store calls.
    - Expose block sizes as tl.constexpr parameters.
    - Only call Triton language intrinsics such as tl.program_id, tl.arange,
      tl.load, tl.store, tl.zeros, tl.full, tl.sqrt, and tl.sum inside
      @triton.jit functions; wrappers should use Python, torch allocation, and
      triton launch helpers only.
    - Do not use `tl.device_ptr` or other non-existent Triton type annotations.
    - Pass scalar kernel arguments as Python numbers, not as `torch.tensor(...)`
      objects. Inside kernels, use scalar values directly; do not wrap scalar
      arguments in pointer-like constructs or unnecessary `tl.full` calls.
    - Inside wrappers, access tensor strides with `x.stride(dim)` or
      `x.stride()`, not `x.stride[dim]`; PyTorch exposes stride as a method.
    - If the task requires non-contiguous logical inputs and your kernel uses
      flat contiguous pointer offsets, explicitly make a contiguous input copy
      in the wrapper before flattening. Do not call `reshape` on a non-contiguous
      tensor and then assume simple contiguous pointer arithmetic is valid.
    - Triton vector shapes must be compile-time constants. Use
      `tl.arange(0, BLOCK_SIZE)`, `tl.zeros([BLOCK_SIZE], ...)`, or
      `tl.full([BLOCK_SIZE], ...)` where `BLOCK_SIZE` is tl.constexpr; never use
      runtime sizes such as `D` in vector shape positions.
    - Implement dtype support only when the task prompt, reference, or visible
      tests require or allow it. When lower-precision inputs such as fp16 or
      bfloat16 are in scope and the kernel uses math intrinsics such as tl.exp,
      tl.sigmoid, tl.sqrt, or division-heavy numerics, cast loaded values to
      tl.float32 before those operations; store the result back to the output
      tensor dtype as needed.
    - Do not use PyTorch for the core computation. PyTorch is allowed only for
      wrapper allocation, shape/dtype handling, contiguity, random input creation
      when explicitly required by the task, and Triton launch setup.
    """
)

DIRECT_IMPLEMENTER_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert GPU kernel engineer specializing in Triton.

    Write a complete Python module that satisfies the task. Prefer the default
    Triton contract below unless the task prompt, reference, or public tests
    require a deviation.

    Default Triton contract:
    - Pass torch tensors directly to kernels; never pass tensor.data_ptr(),
      torch.dtype objects, Python lists/tuples, or CUDA streams as kernel launch
      arguments.
    - Leave JIT pointer parameters unannotated; Triton infers pointer and dtype
      specialization from tensor arguments.
    - Index pointers in tensor elements, not bytes.
    - Use tl.constexpr for block/tile sizes and compile-time mode constants.
    - Keep wrapper code to allocation, shape extraction, necessary contiguity
      handling, and launch setup. Do not add validation stricter than the
      reference behavior.

    Mini-cookbook:
    - 1D elementwise: offsets = pid * BLOCK + tl.arange(0, BLOCK); mask = offsets < N.
    - Rowwise reduction: map rows or row tiles to programs; reduce a constexpr
      column block with tl.sum/tl.max/etc. and masks.
    - 2D tile: use row offsets[:, None], column offsets[None, :], and a combined
      bounds mask.

    Hard requirements:
    - Output only valid Python source code: no Markdown fences and no prose.
    - Include all required imports.
    - Preserve the exact public API requested by the task.
    - Use @triton.jit for the core GPU computation.
    - Access global memory through tl.load and tl.store inside JIT kernels.
    - Use tl.program_id and tl.arange for tiled program mapping.
    - Use mask= on vectorized tl.load and tl.store calls.
    - Expose block sizes as tl.constexpr parameters.
    - Only call Triton language intrinsics such as tl.program_id, tl.arange,
      tl.load, tl.store, tl.zeros, tl.full, tl.sqrt, and tl.sum inside
      @triton.jit functions; wrappers should use Python, torch allocation, and
      triton launch helpers only.
    - Do not use `tl.device_ptr` or other non-existent Triton type annotations.
    - Pass scalar kernel arguments as Python numbers, not as `torch.tensor(...)`
      objects. Inside kernels, use scalar values directly; do not wrap scalar
      arguments in pointer-like constructs or unnecessary `tl.full` calls.
    - Inside wrappers, access tensor strides with `x.stride(dim)` or
      `x.stride()`, not `x.stride[dim]`; PyTorch exposes stride as a method.
    - If the task requires non-contiguous logical inputs and your kernel uses
      flat contiguous pointer offsets, explicitly make a contiguous input copy
      in the wrapper before flattening. Do not call `reshape` on a non-contiguous
      tensor and then assume simple contiguous pointer arithmetic is valid.
    - Triton vector shapes must be compile-time constants. Use
      `tl.arange(0, BLOCK_SIZE)`, `tl.zeros([BLOCK_SIZE], ...)`, or
      `tl.full([BLOCK_SIZE], ...)` where `BLOCK_SIZE` is tl.constexpr; never use
      runtime sizes such as `D` in vector shape positions.
    - Implement dtype support only when the task prompt, reference, or visible
      tests require or allow it. When lower-precision inputs such as fp16 or
      bfloat16 are in scope and the kernel uses math intrinsics such as tl.exp,
      tl.sigmoid, tl.sqrt, or division-heavy numerics, cast loaded values to
      tl.float32 before those operations; store the result back to the output
      tensor dtype as needed.
    - Do not use PyTorch for the core computation. PyTorch is allowed only for
      wrapper allocation, shape/dtype handling, contiguity, random input creation
      when explicitly required by the task, and Triton launch setup.
    """
)

REPAIR_SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a senior Triton kernel engineer diagnosing failed public tests.
    Produce a concise repair directive for the downstream kernel-generation
    model. Do not write source code.

    Use only these simple XML-style sections:
    <repair_directive>
    <diagnosis>...</diagnosis>
    <required_changes>...</required_changes>
    <avoid>...</avoid>
    </repair_directive>

    Rules:
    - Use only evidence from the task, teacher plan, candidate static warnings,
      and public-test failures.
    - Do not include Python source code or Markdown fences.
    """
)


def tagged_block(tag: str, content: str) -> str:
    """Wrap raw text in a simple XML-style tag without escaping model-facing text."""
    return f"<{tag}>\n{str(content).strip()}\n</{tag}>"


def parse_tagged_sections(
    content: str,
    section_names: Iterable[str],
    *,
    root: str | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Best-effort parser for model-emitted XML-ish section tags.

    This intentionally avoids strict XML parsing because model output may include
    unescaped comparison operators or other prose that is not XML-safe.
    """
    text = content or ""
    warnings: list[str] = []
    if root is not None and f"<{root}" not in text.lower():
        warnings.append(f"missing <{root}> root tag")

    sections: dict[str, str] = {}
    for name in section_names:
        pattern = re.compile(rf"<{re.escape(name)}>\s*(.*?)\s*</{re.escape(name)}>", re.DOTALL | re.IGNORECASE)
        match = pattern.search(text)
        if match is None:
            warnings.append(f"missing <{name}> section")
            continue
        sections[name] = match.group(1).strip()
    return sections, warnings


def parse_teacher_plan(content: str) -> tuple[dict[str, str], list[str]]:
    return parse_tagged_sections(content, TEACHER_PLAN_SECTIONS, root="kernel_plan")


def parse_repair_directive(content: str) -> tuple[dict[str, str], list[str]]:
    return parse_tagged_sections(content, REPAIR_SECTIONS, root="repair_directive")


def build_teacher_messages(
    task: KAGBenchTask,
    *,
    include_public_tests: bool = True,
) -> list[dict[str, str]]:
    """Build the teacher-model planning prompt for one task."""
    payload = task.prompt_payload(include_public_tests=include_public_tests)
    user_parts = [
        tagged_block("task_id", payload["task_id"]),
        tagged_block("entry_file", payload["entry_file"]),
        tagged_block("task_prompt", payload["task_prompt"]),
        tagged_block("pytorch_reference", payload["pytorch_reference"]),
    ]
    if include_public_tests and "public_tests" in payload:
        user_parts.append(tagged_block("public_tests", payload["public_tests"]))
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_implementer_messages(
    task: KAGBenchTask,
    *,
    teacher_plan: str,
    repair_directive: str | None = None,
    include_public_tests: bool = True,
) -> list[dict[str, str]]:
    """Build the Gemma implementer prompt for one candidate generation."""
    payload = task.prompt_payload(include_public_tests=include_public_tests)
    user_parts = [
        tagged_block("teacher_plan", teacher_plan),
    ]
    if repair_directive:
        user_parts.append(tagged_block("repair_directive", repair_directive))
    user_parts.extend(
        [
            tagged_block("task_id", payload["task_id"]),
            tagged_block("entry_file", payload["entry_file"]),
            tagged_block("task_prompt", payload["task_prompt"]),
            tagged_block("pytorch_reference", payload["pytorch_reference"]),
        ]
    )
    if include_public_tests and "public_tests" in payload:
        user_parts.append(tagged_block("public_tests", payload["public_tests"]))
    return [
        {"role": "system", "content": IMPLEMENTER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def build_direct_implementer_messages(
    task: KAGBenchTask,
    *,
    include_public_tests: bool = True,
) -> list[dict[str, str]]:
    """Build a no-teacher-plan implementer prompt for direct best-of-K ablations."""
    payload = task.prompt_payload(include_public_tests=include_public_tests)
    user_parts = [
        tagged_block("task_id", payload["task_id"]),
        tagged_block("entry_file", payload["entry_file"]),
        tagged_block("task_prompt", payload["task_prompt"]),
        tagged_block("pytorch_reference", payload["pytorch_reference"]),
    ]
    if include_public_tests and "public_tests" in payload:
        user_parts.append(tagged_block("public_tests", payload["public_tests"]))
    return [
        {"role": "system", "content": DIRECT_IMPLEMENTER_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def summarize_failures(evaluations: Iterable[EvaluationResult], *, max_chars: int = 6000) -> str:
    """Compact public-test failures for repair prompting."""
    chunks = []
    for result in evaluations:
        chunks.append(
            "\n".join(
                [
                    f"candidate_id: {result.candidate_id}",
                    f"phase: {result.phase}",
                    f"call@1: {result.call_at_1}",
                    f"exe@1: {result.exe_at_1}",
                    f"error_type: {result.error_type}",
                    f"error_message: {result.error_message}",
                    "mismatches:",
                    *(f"- {mismatch}" for mismatch in result.mismatches[:8]),
                    "stderr:",
                    result.stderr[-1500:],
                ]
            )
        )
    text = "\n\n---\n\n".join(chunks).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 200] + "\n...[failure summary truncated]"


def summarize_static_checks(
    candidates: Iterable[CandidateGeneration], *, max_chars: int = 4000
) -> str:
    """Compact candidate static diagnostics for repair prompting."""
    chunks = []
    for candidate in candidates:
        static = candidate.static
        chunks.append(
            "\n".join(
                [
                    f"candidate_id: {candidate.candidate_id}",
                    f"attempt: {candidate.attempt}",
                    f"candidate_index: {candidate.candidate_index}",
                    f"syntax_ok: {static.syntax_ok}",
                    f"triton_jit_count: {static.triton_jit_count}",
                    f"flags: {static.flags_text}",
                    "semantic_warnings:",
                    *(f"- {warning}" for warning in static.semantic_warnings[:8]),
                    "torch_calls:",
                    *(f"- {call}" for call in static.torch_calls[:12]),
                ]
            )
        )
    text = "\n\n---\n\n".join(chunks).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 200] + "\n...[static-check summary truncated]"


def build_repair_messages(
    task: KAGBenchTask,
    *,
    teacher_plan: str,
    failed_evaluations: Iterable[EvaluationResult],
    failed_candidates: Iterable[CandidateGeneration] = (),
    include_public_tests: bool = True,
) -> list[dict[str, str]]:
    """Build the teacher-model repair directive prompt."""
    payload = task.prompt_payload(include_public_tests=include_public_tests)
    user_parts = [
        tagged_block("teacher_plan", teacher_plan),
        tagged_block("task_prompt", payload["task_prompt"]),
        tagged_block("pytorch_reference", payload["pytorch_reference"]),
    ]
    if include_public_tests and "public_tests" in payload:
        user_parts.append(tagged_block("public_tests", payload["public_tests"]))
    static_summary = summarize_static_checks(failed_candidates)
    if static_summary:
        user_parts.append(tagged_block("candidate_static_checks", static_summary))
    user_parts.append(tagged_block("public_failures", summarize_failures(failed_evaluations)))
    return [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]
