import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    from collections import defaultdict
    from pathlib import Path

    import marimo as mo

    from benchmark import (
        evaluate_entry,
        load_jsonl,
        load_t_simple_entries,
    )

    return (
        Path,
        defaultdict,
        evaluate_entry,
        load_jsonl,
        load_t_simple_entries,
        mo,
    )


@app.cell
def _(Path, load_jsonl, load_t_simple_entries):
    NOTEBOOK_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = NOTEBOOK_DIR.parent
    DATA_DIR = NOTEBOOK_DIR / "data"
    TRITONBENCH_ROOT = PROJECT_ROOT / "vendor" / "TritonBench"

    T_simple, _errors, _T_json, _T_simple_alpaca = load_t_simple_entries(TRITONBENCH_ROOT)
    entries_by_file = {entry["file"]: entry for entry in T_simple}
    entries_by_index = {i: entry for i, entry in enumerate(T_simple)}

    jsonl_paths = sorted(DATA_DIR.glob("*.jsonl"))
    raw_results = []
    for _path in jsonl_paths:
        for _row in load_jsonl(_path):
            raw_results.append(_row)

    {
        "jsonl_files": [p.name for p in jsonl_paths],
        "total_results": len(raw_results),
        "tritonbench_entries": len(T_simple),
    }
    return entries_by_file, entries_by_index, raw_results


@app.cell
def _(entries_by_file, entries_by_index, evaluate_entry, mo, raw_results):
    @mo.persistent_cache
    def _batch_evaluate(results_key, results):
        evals = []
        for row in results:
            entry = entries_by_file.get(row.get("entry_file"))
            if entry is None and row.get("entry_index") is not None:
                entry = entries_by_index.get(row["entry_index"])

            if entry is None:
                evals.append(
                    {
                        "file": row.get("entry_file"),
                        "model": row.get("model"),
                        "model_label": row.get("model_label"),
                        "call@1": False,
                        "exe@1": False,
                        "mismatches": ["entry not found in dataset"],
                    }
                )
                continue

            try:
                ev = evaluate_entry(
                    entry,
                    pred_code=row.get("content", ""),
                    timeout=180,
                )
                ev["model"] = row.get("model")
                ev["model_label"] = row.get("model_label")
            except Exception as exc:
                ev = {
                    "file": row.get("entry_file"),
                    "model": row.get("model"),
                    "model_label": row.get("model_label"),
                    "call@1": False,
                    "exe@1": False,
                    "mismatches": [str(exc)],
                }
            evals.append(ev)
        return evals

    _results_key = tuple(r.get("request_hash", "") for r in raw_results)
    all_eval_results = _batch_evaluate(_results_key, raw_results)
    return (all_eval_results,)


@app.cell(hide_code=True)
def _(all_eval_results, defaultdict, mo):
    _model_stats = defaultdict(lambda: {"total": 0, "call_pass": 0, "exe_pass": 0})
    for _ev in all_eval_results:
        _label = _ev.get("model_label") or _ev.get("model") or "unknown"
        _model_stats[_label]["total"] += 1
        if _ev.get("call@1"):
            _model_stats[_label]["call_pass"] += 1
        if _ev.get("exe@1"):
            _model_stats[_label]["exe_pass"] += 1

    _summary_rows = []
    for _label, _s in sorted(_model_stats.items()):
        _n = _s["total"]
        _summary_rows.append(
            {
                "model": _label,
                "total kernels": _n,
                "call@1 passed": _s["call_pass"],
                "call@1 %": f"{100 * _s['call_pass'] / _n:.1f}%" if _n else "N/A",
                "exe@1 passed": _s["exe_pass"],
                "exe@1 %": f"{100 * _s['exe_pass'] / _n:.1f}%" if _n else "N/A",
            }
        )

    mo.vstack(
        [
            mo.md("## TritonBench — Resultados por modelo"),
            mo.md(
                "**call@1**: el kernel corrió sin crash.  \n"
                "**exe@1**: el output del kernel coincide con la referencia PyTorch."
            ),
            mo.ui.table(_summary_rows),
        ]
    )


@app.cell(hide_code=True)
def _(all_eval_results, mo):
    _detail_rows = [
        {
            "file": ev.get("file"),
            "model": ev.get("model_label") or ev.get("model"),
            "call@1": "✓" if ev.get("call@1") else "✗",
            "exe@1": "✓" if ev.get("exe@1") else "✗",
            "mismatches": "; ".join(ev.get("mismatches") or []),
        }
        for ev in all_eval_results
    ]

    mo.vstack(
        [
            mo.md("## Detalle por kernel"),
            mo.ui.table(_detail_rows),
        ]
    )


if __name__ == "__main__":
    app.run()
