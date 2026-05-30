"""
Evaluate LLM kernels against TritonBench.

Requires: Linux + GPU + PyTorch + Triton
Saves results to: notebooks/results/eval_results.json

Google Colab instructions:
    !git clone https://github.com/JoseMartinezM/KernelForge.git
    import os; os.chdir("KernelForge")
    !pip install triton -q
    !python notebooks/evaluate/run_eval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

EVALUATE_DIR = Path(__file__).resolve().parent
NOTEBOOKS_DIR = EVALUATE_DIR.parent
PROJECT_ROOT = NOTEBOOKS_DIR.parent
RESULTS_DIR = NOTEBOOKS_DIR / "results"
DATA_DIR = NOTEBOOKS_DIR / "data"
TRITONBENCH_ROOT = PROJECT_ROOT / "vendor" / "TritonBench"
OUTPUT_PATH = RESULTS_DIR / "eval_results.json"

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kernelforge.benchmark import evaluate_entry, load_jsonl, load_t_simple_entries  # noqa: E402


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading TritonBench dataset...")
    T_simple, errors, _, _ = load_t_simple_entries(TRITONBENCH_ROOT)
    entries_by_file = {entry["file"]: entry for entry in T_simple}
    entries_by_index = {i: entry for i, entry in enumerate(T_simple)}
    print(f"  {len(T_simple)} entries, {len(errors)} matching errors")

    jsonl_paths = sorted(DATA_DIR.glob("*.jsonl"))
    if not jsonl_paths:
        print(f"ERROR: no .jsonl files found in {DATA_DIR}")
        sys.exit(1)

    raw_results: list[dict] = []
    for path in jsonl_paths:
        rows = load_jsonl(path)
        print(f"  {path.name}: {len(rows)} results")
        raw_results.extend(rows)

    print(f"\nEvaluating {len(raw_results)} kernels...\n")

    eval_results = []
    model_stats: dict[str, dict] = {}

    for i, row in enumerate(raw_results, start=1):
        entry_file = row.get("entry_file")
        model_label = row.get("model_label") or row.get("model") or "unknown"

        entry = entries_by_file.get(entry_file)
        if entry is None and row.get("entry_index") is not None:
            entry = entries_by_index.get(row["entry_index"])

        if entry is None:
            result = {
                "file": entry_file,
                "model": row.get("model"),
                "model_label": model_label,
                "call@1": False,
                "exe@1": False,
                "mismatches": ["entry not found in dataset"],
            }
        else:
            try:
                result = evaluate_entry(
                    entry,
                    pred_code=row.get("content", ""),
                    timeout=180,
                )
                result["model"] = row.get("model")
                result["model_label"] = model_label
            except Exception as exc:
                result = {
                    "file": entry_file,
                    "model": row.get("model"),
                    "model_label": model_label,
                    "call@1": False,
                    "exe@1": False,
                    "mismatches": [str(exc)],
                }

        eval_results.append(result)

        if model_label not in model_stats:
            model_stats[model_label] = {"total": 0, "call_pass": 0, "exe_pass": 0}
        model_stats[model_label]["total"] += 1
        if result.get("call@1"):
            model_stats[model_label]["call_pass"] += 1
        if result.get("exe@1"):
            model_stats[model_label]["exe_pass"] += 1

        status = "call✓ exe✓" if result.get("exe@1") else ("call✓" if result.get("call@1") else "✗✗")
        print(f"[{i:3}/{len(raw_results)}] {model_label:20} {str(entry_file):50} {status}")

    print("\n" + "=" * 60)
    print("RESULTADOS POR MODELO")
    print("=" * 60)
    for label, s in sorted(model_stats.items()):
        n = s["total"]
        call_pct = 100 * s["call_pass"] / n if n else 0
        exe_pct = 100 * s["exe_pass"] / n if n else 0
        print(f"\n{label}")
        print(f"  Total   : {n}")
        print(f"  call@1  : {s['call_pass']}/{n}  ({call_pct:.1f}%)")
        print(f"  exe@1   : {s['exe_pass']}/{n}  ({exe_pct:.1f}%)")

    output = {"model_stats": model_stats, "per_kernel": eval_results}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nGuardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
