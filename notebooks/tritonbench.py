import marimo

__generated_with = "0.23.6"
app = marimo.App(width="full")


@app.cell
def _():
    from collections import Counter, defaultdict
    from pathlib import Path
    from typing import Any

    import json
    import marimo as mo

    from kernelforge.benchmark import load_jsonl, load_t_simple_entries

    return Any, Counter, Path, defaultdict, json, load_jsonl, load_t_simple_entries, mo


@app.cell
def _(Path, defaultdict, json, load_jsonl, load_t_simple_entries):
    with open("notebooks/results/eval_results.json", encoding="utf-8") as _f:
        eval_results = json.load(_f)

    T_simple, _errors, _, _ = load_t_simple_entries("vendor/TritonBench")
    assert len(_errors) == 0
    entry_by_file = {_entry["file"]: _entry for _entry in T_simple}

    inference_runs = []
    for _file in sorted(Path("notebooks/data").glob("*.jsonl")):
        inference_runs.extend(load_jsonl(_file))

    runs_by_file = defaultdict(dict)
    for _run in inference_runs:
        runs_by_file[_run["entry_file"]][_run["model"]] = dict(_run)

    for _eval in eval_results["per_kernel"]:
        _run = runs_by_file[_eval["file"]].get(_eval["model"])
        if _run is None:
            _run = {
                "entry_file": _eval["file"],
                "model": _eval["model"],
                "model_label": _eval["model_label"],
                "content": "",
                "response": {},
            }
            runs_by_file[_eval["file"]][_eval["model"]] = _run
        _run["eval"] = _eval

    all_results = [
        _run
        for _runs_for_file in runs_by_file.values()
        for _run in _runs_for_file.values()
        if "eval" in _run
    ]

    return T_simple, all_results, entry_by_file, eval_results, runs_by_file


@app.cell
def _(Any, Counter, all_results, eval_results):
    def percent(numerator: int, denominator: int) -> str:
        if denominator == 0:
            return "—"
        return f"{100 * numerator / denominator:.1f}%"


    def token_count(result: dict[str, Any]) -> int | None:
        usage = result.get("response", {}).get("usage", {})
        if isinstance(usage, dict):
            return usage.get("completion_tokens") or usage.get("output_tokens")
        return None


    def first_traceback_line(stderr: str) -> str:
        lines = [line.strip() for line in str(stderr or "").splitlines() if line.strip()]
        if not lines:
            return "No stderr captured"
        for line in reversed(lines):
            if line and not line.startswith("File ") and set(line) != {"^"}:
                return line
        return lines[-1]


    def failure_kind(eval_record: dict[str, Any]) -> str:
        if eval_record["call@1"] and eval_record["exe@1"]:
            return "Passed"
        if eval_record.get("mismatches"):
            return "Wrong answer"
        if not eval_record.get("pred", {}).get("ok", False):
            return "Runtime error"
        if not eval_record["call@1"]:
            return "Call failed"
        return "Execution failed"


    def failure_reason(eval_record: dict[str, Any], max_len: int = 180) -> str:
        if eval_record["call@1"] and eval_record["exe@1"]:
            return "—"
        mismatches = eval_record.get("mismatches") or []
        if mismatches:
            reason = "; ".join(mismatches[:2])
            if len(mismatches) > 2:
                reason += f"; +{len(mismatches) - 2} more"
        else:
            reason = first_traceback_line(eval_record.get("pred", {}).get("stderr", ""))
        return reason if len(reason) <= max_len else reason[: max_len - 1] + "…"


    def make_row(result: dict[str, Any]) -> dict[str, Any]:
        eval_record = result["eval"]
        completion_tokens = token_count(result)
        return {
            "Case": result["entry_file"].removesuffix(".py"),
            "Model": result["model_label"],
            "call@1": "✅" if eval_record["call@1"] else "❌",
            "exe@1": "✅" if eval_record["exe@1"] else "❌",
            "Outcome": failure_kind(eval_record),
            "Reason": failure_reason(eval_record),
            "Tokens": completion_tokens if completion_tokens is not None else "—",
            "Latency (s)": round(result.get("latency_s", 0), 2)
            if result.get("latency_s") is not None
            else "—",
        }


    def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(results)
        call_pass = sum(1 for result in results if result["eval"]["call@1"])
        exe_pass = sum(1 for result in results if result["eval"]["exe@1"])
        wrong_answer = sum(1 for result in results if result["eval"].get("mismatches"))
        runtime_error = sum(
            1
            for result in results
            if not result["eval"].get("pred", {}).get("ok", False)
        )
        return {
            "total": total,
            "call_pass": call_pass,
            "exe_pass": exe_pass,
            "wrong_answer": wrong_answer,
            "runtime_error": runtime_error,
            "call_rate": percent(call_pass, total),
            "exe_rate": percent(exe_pass, total),
        }


    evaluation_rows = [make_row(_result) for _result in all_results]
    model_labels = sorted({_result["model_label"] for _result in all_results})
    case_names = sorted({_result["entry_file"].removesuffix(".py") for _result in all_results})
    outcome_counts = Counter(_row["Outcome"] for _row in evaluation_rows)
    overall_summary = summarize(all_results)

    return (
        case_names,
        evaluation_rows,
        failure_kind,
        failure_reason,
        first_traceback_line,
        make_row,
        model_labels,
        outcome_counts,
        overall_summary,
        percent,
        summarize,
        token_count,
    )


@app.cell
def _(case_names, mo, model_labels):
    model_filter = mo.ui.dropdown(
        ["All models", *model_labels],
        value="All models",
        searchable=True,
        label="Model",
        full_width=True,
    )
    case_filter = mo.ui.dropdown(
        ["All cases", *case_names],
        value="All cases",
        searchable=True,
        label="Case",
        full_width=True,
    )
    outcome_filter = mo.ui.dropdown(
        ["All outcomes", "Passed", "Runtime error", "Wrong answer", "Call failed", "Execution failed"],
        value="All outcomes",
        label="Outcome",
        full_width=True,
    )
    search_filter = mo.ui.text(
        placeholder="Search case, model, or failure text…",
        label="Text search",
        full_width=True,
    )

    return case_filter, model_filter, outcome_filter, search_filter


@app.cell
def _(case_filter, mo, model_filter, outcome_filter, search_filter):
    mo.sidebar(
        mo.vstack(
            [
                mo.md("## TritonBench explorer"),
                mo.md(
                    "Use the stateful controls below to filter the table. "
                    "Selecting a row updates the detail view without crowding the page."
                ),
                model_filter,
                case_filter,
                outcome_filter,
                search_filter,
            ],
            gap=1,
        ),
        width=340,
    )
    return


@app.cell
def _(
    all_results,
    case_filter,
    make_row,
    model_filter,
    outcome_filter,
    search_filter,
    summarize,
):
    def matches_filters(result):
        row = make_row(result)
        case_ok = case_filter.value == "All cases" or row["Case"] == case_filter.value
        model_ok = model_filter.value == "All models" or row["Model"] == model_filter.value
        outcome_ok = outcome_filter.value == "All outcomes" or row["Outcome"] == outcome_filter.value
        query = (search_filter.value or "").strip().lower()
        text_ok = not query or query in " ".join(str(value).lower() for value in row.values())
        return case_ok and model_ok and outcome_ok and text_ok


    filtered_results = [_result for _result in all_results if matches_filters(_result)]
    filtered_rows = [make_row(_result) for _result in filtered_results]
    filtered_summary = summarize(filtered_results)
    return filtered_results, filtered_rows, filtered_summary, matches_filters


@app.cell
def _(filtered_summary, mo, overall_summary):
    mo.vstack(
        [
            mo.md("# TritonBench evaluation results"),
            mo.hstack(
                [
                    mo.stat(
                        value=str(filtered_summary["total"]),
                        label="Runs shown",
                        caption=f"{overall_summary['total']} total",
                        bordered=True,
                    ),
                    mo.stat(
                        value=filtered_summary["call_rate"],
                        label="call@1 pass rate",
                        caption=f"{filtered_summary['call_pass']} passing",
                        bordered=True,
                    ),
                    mo.stat(
                        value=filtered_summary["exe_rate"],
                        label="exe@1 pass rate",
                        caption=f"{filtered_summary['exe_pass']} passing",
                        bordered=True,
                    ),
                    mo.stat(
                        value=str(filtered_summary["runtime_error"]),
                        label="Runtime errors",
                        caption=f"{filtered_summary['wrong_answer']} wrong answers",
                        bordered=True,
                    ),
                ],
                widths="equal",
                gap=1,
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _(eval_results, mo, outcome_counts, percent):
    _model_rows = []
    for _model, _stats in eval_results["model_stats"].items():
        _model_rows.append(
            {
                "Model": _model,
                "Runs": _stats["total"],
                "call@1": f"{_stats['call_pass']} ({percent(_stats['call_pass'], _stats['total'])})",
                "exe@1": f"{_stats['exe_pass']} ({percent(_stats['exe_pass'], _stats['total'])})",
            }
        )

    _outcome_rows = [
        {"Outcome": _outcome, "Runs": _count}
        for _outcome, _count in outcome_counts.most_common()
    ]

    mo.accordion(
        {
            "Model summary": mo.ui.table(
                _model_rows,
                selection=None,
                pagination=False,
                show_data_types=False,
                show_download=False,
            ),
            "Outcome breakdown": mo.ui.table(
                _outcome_rows,
                selection=None,
                pagination=False,
                show_data_types=False,
                show_download=False,
            ),
        },
        multiple=True,
    )
    return


@app.cell
def _(filtered_rows, mo):
    eval_table = mo.ui.table(
        filtered_rows,
        selection="single",
        initial_selection=[0] if filtered_rows else None,
        pagination=True,
        page_size=15,
        show_data_types=False,
        show_download=True,
        freeze_columns_left=["Case", "Model"],
        wrapped_columns=["Reason"],
        max_height=520,
        label="Select a case/model row to inspect details",
    )
    eval_table
    return (eval_table,)


@app.cell
def _(all_results, eval_table, filtered_results, filtered_rows, make_row):
    selected_row = None
    if eval_table.value:
        selected_row = eval_table.value[0]
    elif filtered_rows:
        selected_row = filtered_rows[0]

    selected_result = None
    if selected_row is not None:
        for _result in all_results:
            _row = make_row(_result)
            if _row["Case"] == selected_row["Case"] and _row["Model"] == selected_row["Model"]:
                selected_result = _result
                break
    elif filtered_results:
        selected_result = filtered_results[0]

    return selected_result, selected_row


@app.cell
def _(Any, entry_by_file, failure_kind, failure_reason, first_traceback_line, token_count):
    def code_block(code: str, language: str = "python") -> str:
        return f"````{language}\n{str(code or '').rstrip()}\n````"


    def result_metadata(result: dict[str, Any]) -> dict[str, Any]:
        eval_record = result["eval"]
        entry = entry_by_file.get(result["entry_file"], {})
        return {
            "Case": result["entry_file"].removesuffix(".py"),
            "Model": result["model_label"],
            "Provider": result.get("provider", "—"),
            "Difficulty": entry.get("difficulty", "—"),
            "Params": entry.get("params_cnt", "—"),
            "Torch ops": entry.get("torch_cnt", "—"),
            "Tokens": token_count(result) or "—",
            "Latency (s)": round(result.get("latency_s", 0), 2)
            if result.get("latency_s") is not None
            else "—",
            "call@1": "Pass" if eval_record["call@1"] else "Fail",
            "exe@1": "Pass" if eval_record["exe@1"] else "Fail",
            "Outcome": failure_kind(eval_record),
            "Reason": failure_reason(eval_record, max_len=500),
        }


    def stderr_tail(result: dict[str, Any], max_lines: int = 80) -> str:
        stderr = result["eval"].get("pred", {}).get("stderr", "")
        lines = str(stderr or "").splitlines()
        return "\n".join(lines[-max_lines:]) if lines else ""


    def stdout_tail(result: dict[str, Any], max_lines: int = 80) -> str:
        stdout = result["eval"].get("pred", {}).get("stdout", "")
        lines = str(stdout or "").splitlines()
        return "\n".join(lines[-max_lines:]) if lines else ""

    return code_block, result_metadata, stderr_tail, stdout_tail


@app.cell
def _(
    code_block,
    entry_by_file,
    mo,
    result_metadata,
    selected_result,
    stderr_tail,
    stdout_tail,
):
    if selected_result is None:
        detail_view = mo.md("_No results match the current filters._")
    else:
        _entry = entry_by_file.get(selected_result["entry_file"], {})
        _eval = selected_result["eval"]
        _metadata = result_metadata(selected_result)
        _metadata_rows = [{"Field": key, "Value": value} for key, value in _metadata.items()]
        _mismatches = _eval.get("mismatches") or []
        _stderr = stderr_tail(selected_result)
        _stdout = stdout_tail(selected_result)

        _overview = mo.vstack(
            [
                mo.md(
                    f"## `{_metadata['Case']}` answered by **{_metadata['Model']}**\n"
                    f"**Outcome:** {_metadata['Outcome']}  \n"
                    f"**Reason:** {_metadata['Reason']}"
                ),
                mo.ui.table(
                    _metadata_rows,
                    selection=None,
                    pagination=False,
                    show_data_types=False,
                    show_download=False,
                ),
                mo.accordion(
                    {
                        "Mismatches": mo.md("\n".join(f"- {item}" for item in _mismatches) or "No mismatches captured."),
                        "stderr tail": mo.md(code_block(_stderr or "No stderr captured.", "text")),
                        "stdout tail": mo.md(code_block(_stdout or "No stdout captured.", "text")),
                    },
                    multiple=False,
                ),
            ],
            gap=1,
        )

        _spec = mo.vstack(
            [
                mo.md(f"### Functional description\n{_entry.get('description', '—')}"),
                mo.accordion(
                    {
                        "Function signature & arguments": mo.md(code_block(_entry.get("func_inputs", ""), "text")),
                        "Mathematical formulation": mo.md(_entry.get("math") or "—"),
                        "Additional information": mo.md(_entry.get("other") or "—"),
                    },
                    multiple=False,
                ),
            ],
            gap=1,
        )

        detail_view = mo.ui.tabs(
            {
                "Overview": _overview,
                "Generated Triton": mo.md(code_block(selected_result.get("content", ""))),
                "Reference PyTorch": mo.md(code_block(_entry.get("ref_code", ""))),
                "Tests": mo.md(code_block(_entry.get("test_code", ""))),
                "Task spec": _spec,
            },
            value="Overview",
            lazy=True,
        )

    detail_view
    return (detail_view,)


if __name__ == "__main__":
    app.run()
