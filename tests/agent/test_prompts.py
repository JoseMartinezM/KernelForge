from __future__ import annotations

import json

from kernelforge.agent.prompts import (
    build_direct_implementer_messages,
    build_implementer_messages,
    build_teacher_messages,
    parse_teacher_plan,
)
from kernelforge.agent.schemas import KAGBenchTask


def fake_task() -> KAGBenchTask:
    return KAGBenchTask(
        task_id="suite/example",
        entry_file="example.py",
        prompt="Implement add(x, y).",
        pytorch_reference="def add(x, y): return x + y",
        public_tests="PUBLIC_VISIBLE_TESTS",
        unit_tests="SECRET_HIDDEN_TESTS",
        tags=("elementwise",),
        source_file="SECRET_SOURCE_PATH",
        vendored_source="SECRET_SOURCE_CONTENT",
    )


def test_teacher_prompt_uses_visible_task_fields_only():
    messages = build_teacher_messages(fake_task())
    rendered = json.dumps(messages)

    assert "Implement add" in rendered
    assert "PUBLIC_VISIBLE_TESTS" in rendered
    assert "SECRET_HIDDEN_TESTS" not in rendered
    assert "SECRET_SOURCE_PATH" not in rendered
    assert "SECRET_SOURCE_CONTENT" not in rendered


def test_implementer_prompt_uses_teacher_plan_and_visible_task_fields_only():
    messages = build_implementer_messages(
        fake_task(),
        teacher_plan="<kernel_plan><api>Use add(x, y)</api></kernel_plan>",
    )
    rendered = json.dumps(messages)

    assert "Use add" in rendered
    assert "PUBLIC_VISIBLE_TESTS" in rendered
    assert "SECRET_HIDDEN_TESTS" not in rendered
    assert "SECRET_SOURCE_PATH" not in rendered
    assert "tl.device_ptr" in rendered
    assert "tensor.data_ptr()" in rendered
    assert "tl.float32" in rendered
    assert "torch.dtype" in rendered
    assert "CUDA streams" in rendered
    assert "torch.tensor" in rendered
    assert "tl.arange(0, BLOCK_SIZE)" in rendered
    assert "x.stride(dim)" in rendered
    assert "contiguous input copy" in rendered


def test_direct_implementer_prompt_has_no_teacher_plan_and_uses_visible_task_fields_only():
    messages = build_direct_implementer_messages(fake_task())
    rendered = json.dumps(messages)

    assert "Implement add" in rendered
    assert "PUBLIC_VISIBLE_TESTS" in rendered
    assert "SECRET_HIDDEN_TESTS" not in rendered
    assert "SECRET_SOURCE_PATH" not in rendered
    assert "SECRET_SOURCE_CONTENT" not in rendered
    assert "teacher_plan" not in rendered
    assert "tl.device_ptr" in rendered


def test_teacher_plan_parser_is_tolerant_xmlish_not_strict_xml():
    sections, warnings = parse_teacher_plan(
        """
        <kernel_plan>
        <pattern>elementwise</pattern>
        <api_contract>Implement add(x, y)</api_contract>
        <program_mapping>1D grid; offsets < n</program_mapping>
        <numerics>preserve dtype</numerics>
        <non_defaults>none</non_defaults>
        <risks>missing mask</risks>
        </kernel_plan>
        """
    )

    assert sections["api_contract"] == "Implement add(x, y)"
    assert sections["program_mapping"] == "1D grid; offsets < n"
    assert warnings == []
