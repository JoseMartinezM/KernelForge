import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import defaultdict
    from pathlib import Path

    import marimo as mo
    from benchmark import (
        analysis_rows,
        cleanup_generated_code,
        evaluate_entry,
        load_jsonl,
        load_results,
        load_t_simple_entries,
        make_prompt,
        model_summaries,
        summarize_analysis,
    )

    return (
        Path,
        analysis_rows,
        cleanup_generated_code,
        defaultdict,
        evaluate_entry,
        load_jsonl,
        load_results,
        load_t_simple_entries,
        make_prompt,
        mo,
        model_summaries,
        summarize_analysis,
    )


@app.cell
def _(Path):
    NOTEBOOK_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = NOTEBOOK_DIR.parent
    DATA_DIR = NOTEBOOK_DIR / "data"
    TRITONBENCH_ROOT = PROJECT_ROOT / "vendor" / "TritonBench"

    result_paths = sorted(DATA_DIR.glob("*.jsonl"))
    if not result_paths:
        raise FileNotFoundError(f"No JSONL ledgers found in {DATA_DIR}")

    result_path_map = {path.stem: path for path in result_paths}
    return TRITONBENCH_ROOT, result_paths


@app.cell
def _(TRITONBENCH_ROOT, load_t_simple_entries):
    T_simple, dataset_errors, T_json, T_simple_alpaca = load_t_simple_entries(
        TRITONBENCH_ROOT
    )

    entries_by_index = {index: entry for index, entry in enumerate(T_simple)}
    entries_by_file = {entry["file"]: entry for entry in T_simple}

    dataset_overview = {
        "t_simple_entries": len(T_simple),
        "dataset_errors": len(dataset_errors),
        "t_json_entries": len(T_json),
        "t_simple_alpaca_entries": len(T_simple_alpaca),
    }
    return dataset_overview, entries_by_file, entries_by_index


@app.cell
def _(
    analysis_rows,
    load_jsonl,
    load_results,
    model_summaries,
    result_paths,
    summarize_analysis,
):
    rows_by_ledger = {path.stem: load_jsonl(path) for path in result_paths}
    raw_results = load_results(result_paths)
    analysis_table = analysis_rows(raw_results)
    overall_summary = summarize_analysis(analysis_table)
    per_model_summary = model_summaries(analysis_table)
    return analysis_table, overall_summary, per_model_summary, raw_results


@app.cell
def _(
    analysis_table,
    defaultdict,
    entries_by_file,
    entries_by_index,
    make_prompt,
    raw_results,
):
    analysis_table_enriched = []
    analysis_by_model = defaultdict(list)
    analysis_by_entry_file = defaultdict(list)
    raw_results_by_model = defaultdict(list)
    raw_results_by_entry_file = defaultdict(list)
    prompts_by_entry_file = {}

    for row in analysis_table:
        entry = entries_by_file.get(row.get("entry_file"))
        if entry is None and row.get("entry_index") is not None:
            entry = entries_by_index.get(row["entry_index"])

        enriched = {
            **row,
            "func_inputs": entry.get("func_inputs") if entry else None,
            "description": entry.get("description") if entry else None,
            "source_path": entry.get("source_path") if entry else None,
        }
        analysis_table_enriched.append(enriched)
        analysis_by_model[enriched.get("model")].append(enriched)
        analysis_by_entry_file[enriched.get("entry_file")].append(enriched)

        if entry is not None and entry["file"] not in prompts_by_entry_file:
            prompts_by_entry_file[entry["file"]] = make_prompt(entry)

    for raw_row in raw_results:
        raw_results_by_model[raw_row.get("model")].append(raw_row)
        raw_results_by_entry_file[raw_row.get("entry_file")].append(raw_row)

    available_models = sorted(
        model for model in analysis_by_model.keys() if isinstance(model, str)
    )
    available_entry_files = sorted(
        entry_file
        for entry_file in analysis_by_entry_file.keys()
        if isinstance(entry_file, str)
    )
    return (analysis_table_enriched,)


@app.cell(hide_code=True)
def _(dataset_overview, overall_summary, per_model_summary, result_paths):
    notebook_overview = {
        "ledger_files": [path.name for path in result_paths],
        "dataset": dataset_overview,
        "results": overall_summary,
        "models": per_model_summary,
    }
    return


@app.cell
def _(analysis_table_enriched, mo):
    selected_result_row = mo.ui.slider(
        0,
        len(analysis_table_enriched) - 1,
        value=0,
        label="Result row",
    )
    return (selected_result_row,)


@app.cell
def _(
    analysis_table_enriched,
    cleanup_generated_code,
    entries_by_file,
    entries_by_index,
    make_prompt,
    raw_results,
    selected_result_row,
):
    selected_row_index = selected_result_row.value
    selected_analysis_row = analysis_table_enriched[selected_row_index]
    selected_raw_row = raw_results[selected_row_index]

    selected_entry = entries_by_file.get(selected_analysis_row.get("entry_file"))
    if selected_entry is None and selected_analysis_row.get("entry_index") is not None:
        selected_entry = entries_by_index.get(selected_analysis_row["entry_index"])

    selected_prompt = (
        make_prompt(selected_entry) if selected_entry is not None else None
    )
    selected_code = cleanup_generated_code(selected_raw_row.get("content"))

    selected_row_summary = {
        "row_index": selected_row_index,
        "model": selected_analysis_row.get("model"),
        "entry_file": selected_analysis_row.get("entry_file"),
        "status": selected_analysis_row.get("status"),
        "finish_reason": selected_analysis_row.get("finish_reason"),
        "latency_s": selected_analysis_row.get("latency_s"),
        "total_tokens": selected_analysis_row.get("total_tokens"),
        "cost_usd": selected_analysis_row.get("cost_usd"),
        "flags": selected_analysis_row.get("flags"),
    }
    return selected_analysis_row, selected_row_summary


@app.cell(hide_code=True)
def _(mo, selected_analysis_row, selected_row_summary):
    mo.vstack(
        [
            mo.md("## Selected result row"),
            selected_row_summary,
            selected_analysis_row,
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    Available convenience vars: `raw_results`, `rows_by_ledger`, `analysis_table`, "
        "`analysis_table_enriched`, `analysis_by_model`, `analysis_by_entry_file`, "
        "`selected_analysis_row`, `selected_raw_row`, `selected_prompt`, `selected_code`, "
        "and `evaluate_loaded_result(...)`.
    """)
    return


@app.cell
def _(entries_by_file, entries_by_index, evaluate_entry):
    def evaluate_loaded_result(result_row, timeout: int = 180):
        entry = entries_by_file.get(result_row.get("entry_file"))
        if entry is None and result_row.get("entry_index") is not None:
            entry = entries_by_index.get(result_row["entry_index"])
        if entry is None:
            raise KeyError(
                f"Could not find TritonBench entry for row: {result_row.get('entry_file')}"
            )
        return evaluate_entry(
            entry,
            pred_code=result_row.get("content", ""),
            timeout=timeout,
        )

    return


if __name__ == "__main__":
    app.run()
