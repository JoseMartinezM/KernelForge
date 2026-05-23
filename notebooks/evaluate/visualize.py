import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from collections import defaultdict
    from pathlib import Path

    import marimo as mo

    EVALUATE_DIR = Path(__file__).resolve().parent
    NOTEBOOKS_DIR = EVALUATE_DIR.parent
    RESULTS_DIR = NOTEBOOKS_DIR / "results"
    RESULTS_FILE = RESULTS_DIR / "eval_results.json"

    return RESULTS_FILE, defaultdict, json, mo


@app.cell
def _(RESULTS_FILE, json, mo):
    if not RESULTS_FILE.exists():
        mo.stop(
            True,
            mo.callout(
                mo.md(
                    f"No se encontró `{RESULTS_FILE.name}` en `notebooks/results/`.  \n"
                    "Corre primero `notebooks/evaluate/run_eval.py` en una máquina Linux con GPU."
                ),
                kind="warn",
            ),
        )

    _data = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    per_kernel = _data["per_kernel"]
    mo.md(f"Resultados cargados: **{len(per_kernel)} kernels** desde `{RESULTS_FILE.name}`")
    return (per_kernel,)


@app.cell(hide_code=True)
def _(defaultdict, mo, per_kernel):
    _stats = defaultdict(lambda: {"total": 0, "call_pass": 0, "exe_pass": 0})
    for _ev in per_kernel:
        _label = _ev.get("model_label") or _ev.get("model") or "unknown"
        _stats[_label]["total"] += 1
        if _ev.get("call@1"):
            _stats[_label]["call_pass"] += 1
        if _ev.get("exe@1"):
            _stats[_label]["exe_pass"] += 1

    _rows = []
    for _label, _s in sorted(_stats.items()):
        _n = _s["total"]
        _rows.append(
            {
                "modelo": _label,
                "kernels": _n,
                "call@1 ✓": _s["call_pass"],
                "call@1 %": f"{100 * _s['call_pass'] / _n:.1f}%" if _n else "N/A",
                "exe@1 ✓": _s["exe_pass"],
                "exe@1 %": f"{100 * _s['exe_pass'] / _n:.1f}%" if _n else "N/A",
            }
        )

    mo.vstack(
        [
            mo.md("## Resultados por modelo"),
            mo.md(
                "**call@1**: el kernel corrió sin crash.  \n"
                "**exe@1**: el output coincide con la referencia PyTorch."
            ),
            mo.ui.table(_rows),
        ]
    )


@app.cell(hide_code=True)
def _(mo, per_kernel):
    _detail = [
        {
            "archivo": ev.get("file"),
            "modelo": ev.get("model_label") or ev.get("model"),
            "call@1": "✓" if ev.get("call@1") else "✗",
            "exe@1": "✓" if ev.get("exe@1") else "✗",
            "error": "; ".join(ev.get("mismatches") or []),
        }
        for ev in per_kernel
    ]

    mo.vstack(
        [
            mo.md("## Detalle por kernel"),
            mo.ui.table(_detail),
        ]
    )


if __name__ == "__main__":
    app.run()
