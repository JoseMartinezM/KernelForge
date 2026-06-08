from __future__ import annotations

import json
from pathlib import Path

from kernelforge.agent.schemas import KAGBenchTask, WorkflowConfig
from kernelforge.agent.workflow import KernelGenerationWorkflow


def test_workflow_dry_run_redacts_grammar_and_preserves_prompt_isolation(tmp_path):
    grammar = tmp_path / "triton.gbnf"
    grammar.write_text('root ::= "x"\n', encoding="utf-8")
    task = KAGBenchTask(
        task_id="suite/add",
        entry_file="add.py",
        prompt="Implement add.",
        pytorch_reference="def add(x, y): return x + y",
        public_tests="PUBLIC_TESTS",
        unit_tests="SECRET_UNIT_TESTS",
        source_file="SECRET_SOURCE.py",
    )
    config = WorkflowConfig(
        run_id="dry-run",
        teacher_model="teacher/model",
        implementer_model="gemma/model",
        out_dir=Path("runs/agent/dry-run"),
        grammar_file=grammar,
    )

    payload = KernelGenerationWorkflow(config).dry_run_payload([task])
    rendered = json.dumps(payload)

    assert payload["implementer_generation"]["extra_body"]["guided_decoding_backend"] == "xgrammar"
    assert payload["implementer_generation"]["extra_body"]["guided_grammar"]["bytes"] == len(
        'root ::= "x"\n'.encode("utf-8")
    )
    assert 'root ::= "x"' not in rendered
    assert "PUBLIC_TESTS" in rendered
    assert "SECRET_UNIT_TESTS" not in rendered
    assert "SECRET_SOURCE" not in rendered


def test_workflow_dry_run_uses_llama_cpp_grammar_shape(tmp_path):
    grammar = tmp_path / "triton.gbnf"
    grammar.write_text('root ::= "x"\n', encoding="utf-8")
    task = KAGBenchTask(
        task_id="suite/add",
        entry_file="add.py",
        prompt="Implement add.",
        pytorch_reference="def add(x, y): return x + y",
        public_tests="PUBLIC_TESTS",
        unit_tests="SECRET_UNIT_TESTS",
    )
    config = WorkflowConfig(
        run_id="dry-run",
        teacher_model="teacher/model",
        implementer_model="gemma/model",
        out_dir=Path("runs/agent/dry-run"),
        grammar_file=grammar,
        grammar_backend="llama-cpp",
    )

    payload = KernelGenerationWorkflow(config).dry_run_payload([task])

    assert payload["implementer_generation"]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert "grammar" in payload["implementer_generation"]["extra_body"]
    assert "guided_grammar" not in payload["implementer_generation"]["extra_body"]
