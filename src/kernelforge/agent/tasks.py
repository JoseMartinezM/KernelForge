from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import KAGBenchTask

DEFAULT_KAGBENCH_LEDGER = Path("benchmarks/KAGBench/kagbench.jsonl")


def load_kagbench(path: str | Path = DEFAULT_KAGBENCH_LEDGER) -> list[KAGBenchTask]:
    """Load KAGBench tasks from the fused JSONL ledger."""
    ledger_path = Path(path)
    rows: list[KAGBenchTask] = []
    with ledger_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {ledger_path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object row in {ledger_path}:{line_number}")
            try:
                rows.append(KAGBenchTask.from_row(row))
            except ValueError as exc:
                raise ValueError(f"Invalid KAGBench row in {ledger_path}:{line_number}: {exc}") from exc
    return rows


def task_by_id(tasks: Iterable[KAGBenchTask], task_id: str) -> KAGBenchTask:
    """Return exactly one task by stable task id."""
    for task in tasks:
        if task.task_id == task_id:
            return task
    raise KeyError(f"unknown KAGBench task id: {task_id}")


def task_by_entry_file(tasks: Iterable[KAGBenchTask], entry_file: str) -> KAGBenchTask:
    """Return exactly one task by entry filename."""
    matches = [task for task in tasks if task.entry_file == entry_file]
    if len(matches) != 1:
        raise KeyError(f"expected one task with entry_file={entry_file!r}, found {len(matches)}")
    return matches[0]


def select_tasks(
    tasks: Iterable[KAGBenchTask],
    *,
    task_ids: Iterable[str] | None = None,
    entry_files: Iterable[str] | None = None,
    tags: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[KAGBenchTask]:
    """Select tasks while preserving the input ledger order."""
    selected = list(tasks)
    task_id_set = set(task_ids or [])
    entry_file_set = set(entry_files or [])
    tag_set = set(tags or [])

    if task_id_set:
        missing = task_id_set - {task.task_id for task in selected}
        if missing:
            raise KeyError(f"unknown task id(s): {', '.join(sorted(missing))}")
        selected = [task for task in selected if task.task_id in task_id_set]

    if entry_file_set:
        missing = entry_file_set - {task.entry_file for task in selected}
        if missing:
            raise KeyError(f"unknown entry file(s): {', '.join(sorted(missing))}")
        selected = [task for task in selected if task.entry_file in entry_file_set]

    if tag_set:
        selected = [task for task in selected if tag_set.issubset(set(task.tags))]

    if limit is not None:
        selected = selected[:limit]
    return selected
