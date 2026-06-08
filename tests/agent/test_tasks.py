from __future__ import annotations

import json

from kernelforge.agent.tasks import load_kagbench, select_tasks


def test_load_kagbench_and_select_tasks(tmp_path):
    ledger = tmp_path / "kagbench.jsonl"
    ledger.write_text(
        json.dumps(
            {
                "id": "suite/add",
                "entry_file": "add.py",
                "prompt": "Implement add.",
                "pytorch_reference": "def add(x, y): return x + y",
                "public_tests": "def public_tests(candidate): pass",
                "unit_tests": "SECRET_UNIT_TESTS",
                "tags": ["elementwise", "cuda"],
                "source_file": "SECRET_SOURCE.py",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    tasks = load_kagbench(ledger)
    selected = select_tasks(tasks, tags=["elementwise"])

    assert len(selected) == 1
    assert selected[0].task_id == "suite/add"
    assert selected[0].prompt_payload() == {
        "task_id": "suite/add",
        "entry_file": "add.py",
        "task_prompt": "Implement add.",
        "pytorch_reference": "def add(x, y): return x + y",
        "public_tests": "def public_tests(candidate): pass",
    }
    assert "SECRET_UNIT_TESTS" not in json.dumps(selected[0].prompt_payload())
    assert "SECRET_SOURCE" not in json.dumps(selected[0].prompt_payload())
