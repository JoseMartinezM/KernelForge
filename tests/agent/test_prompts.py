from __future__ import annotations

import json

from kernelforge.agent.prompts import (
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
    assert "tl.dtype" in rendered
    assert "stream=" in rendered
    assert "torch.tensor" in rendered
    assert "tl.arange(0, BLOCK_SIZE)" in rendered
    assert "x.stride(dim)" in rendered
    assert "contiguous input copy" in rendered


def test_teacher_plan_parser_is_tolerant_xmlish_not_strict_xml():
    sections, warnings = parse_teacher_plan(
        """
        <kernel_plan>
        <api>Implement add(x, y)</api>
        <behavior>Return x < y comparison if requested by prompt.</behavior>
        <triton_mapping>1D grid</triton_mapping>
        <memory>load/store flattened offsets</memory>
        <masks>offsets < n</masks>
        <numerics>preserve dtype</numerics>
        <wrapper>allocate output</wrapper>
        <edge_cases>odd sizes</edge_cases>
        <failure_modes>missing mask</failure_modes>
        </kernel_plan>
        """
    )

    assert sections["api"] == "Implement add(x, y)"
    assert sections["behavior"] == "Return x < y comparison if requested by prompt."
    assert warnings == []
