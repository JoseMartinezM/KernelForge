from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schemas import WorkflowConfig, to_jsonable
from .tasks import DEFAULT_KAGBENCH_LEDGER, load_kagbench, select_tasks
from .workflow import KernelGenerationWorkflow


def _json_generation_args(max_tokens: int | None, temperature: float | None) -> dict[str, Any]:
    generation: dict[str, Any] = {}
    if max_tokens is not None:
        generation["max_tokens"] = max_tokens
    if temperature is not None:
        generation["temperature"] = temperature
    return generation


def _selected_tasks(args: argparse.Namespace) -> list[Any]:
    tasks = load_kagbench(args.kagbench)
    selected = select_tasks(
        tasks,
        task_ids=args.task_id,
        entry_files=args.entry_file,
        tags=args.tag,
        limit=args.limit,
    )
    if not selected:
        raise SystemExit("no KAGBench tasks selected")
    return selected


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kagbench", type=Path, default=DEFAULT_KAGBENCH_LEDGER)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--entry-file", action="append", default=[])
    parser.add_argument("--tag", action="append", default=[])
    parser.add_argument("--limit", type=int)


def cmd_list_tasks(args: argparse.Namespace) -> int:
    tasks = load_kagbench(args.kagbench)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "task_id": task.task_id,
                        "entry_file": task.entry_file,
                        "tags": list(task.tags),
                    }
                    for task in tasks
                ],
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    for task in tasks:
        tags = ",".join(task.tags)
        print(f"{task.task_id}\t{task.entry_file}\t{tags}")
    return 0


def cmd_show_task(args: argparse.Namespace) -> int:
    task = _selected_tasks(args)[0]
    print(
        json.dumps(
            {
                "task_id": task.task_id,
                "entry_file": task.entry_file,
                "tags": list(task.tags),
                "prompt": task.prompt,
                "pytorch_reference": task.pytorch_reference,
                "public_tests": task.public_tests,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    tasks = _selected_tasks(args)
    run_id = args.run_id
    out_dir = args.out
    if out_dir is None:
        out_dir = Path("runs/agent") / run_id

    grammar_file = None if args.no_grammar else args.grammar_file
    config = WorkflowConfig(
        run_id=run_id,
        teacher_model=args.teacher_model,
        implementer_model=args.implementer_model,
        out_dir=out_dir,
        candidates_per_attempt=args.candidates,
        max_repairs=args.max_repairs,
        eval_backend=args.eval_backend,
        eval_timeout_s=args.eval_timeout,
        hidden_eval=not args.no_hidden_eval,
        include_public_tests=not args.no_public_tests_in_prompts,
        grammar_file=grammar_file,
        grammar_backend=args.grammar_backend,
        config_path=args.config,
        teacher_generation=_json_generation_args(args.teacher_max_tokens, args.teacher_temperature),
        implementer_generation=_json_generation_args(
            args.implementer_max_tokens, args.implementer_temperature
        ),
    )
    workflow = KernelGenerationWorkflow(config)
    if args.dry_run:
        print(json.dumps(to_jsonable(workflow.dry_run_payload(tasks)), ensure_ascii=False, indent=2))
        return 0

    summaries = workflow.run(tasks)
    print(
        json.dumps(
            {
                "run_id": config.run_id,
                "out_dir": str(config.out_dir),
                "summaries": [summary.to_row() for summary in summaries],
            },
            ensure_ascii=False,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KernelForge agent workflow CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-tasks", help="List KAGBench tasks.")
    list_parser.add_argument("--kagbench", type=Path, default=DEFAULT_KAGBENCH_LEDGER)
    list_parser.add_argument("--json", action="store_true")
    list_parser.set_defaults(func=cmd_list_tasks)

    show_parser = subparsers.add_parser("show-task", help="Show one generation-visible task.")
    _add_selection_args(show_parser)
    show_parser.set_defaults(func=cmd_show_task)

    run_parser = subparsers.add_parser("run", help="Run the teacher/Gemma workflow.")
    _add_selection_args(run_parser)
    run_parser.add_argument("--config", type=Path, default=Path("src/kernelforge/benchmark/llm_models.json"))
    run_parser.add_argument("--run-id", default=WorkflowConfig("", "", Path(".")).run_id)
    run_parser.add_argument("--out", type=Path)
    run_parser.add_argument("--teacher-model", default="lightning-ai/deepseek-v4-pro")
    run_parser.add_argument("--implementer-model", default="google/gemma-4-E4B-it")
    run_parser.add_argument("--candidates", type=int, default=1)
    run_parser.add_argument("--max-repairs", type=int, default=0)
    run_parser.add_argument("--eval-backend", choices=("none", "local", "modal"), default="none")
    run_parser.add_argument("--eval-timeout", type=int, default=180)
    run_parser.add_argument("--no-hidden-eval", action="store_true")
    run_parser.add_argument("--no-public-tests-in-prompts", action="store_true")
    run_parser.add_argument("--grammar-file", type=Path, default=Path("grammar/triton.gbnf"))
    run_parser.add_argument("--grammar-backend", choices=("xgrammar", "llama-cpp"), default="xgrammar")
    run_parser.add_argument("--no-grammar", action="store_true")
    run_parser.add_argument("--teacher-max-tokens", type=int)
    run_parser.add_argument("--implementer-max-tokens", type=int)
    run_parser.add_argument("--teacher-temperature", type=float)
    run_parser.add_argument("--implementer-temperature", type=float)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
