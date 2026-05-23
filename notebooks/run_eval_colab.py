"""
Evaluacion de resultados LLM contra TritonBench.
Corre en Google Colab con GPU o cualquier maquina Linux con PyTorch + Triton.

Instrucciones Colab:
  1. Subir el repo completo (o clonar desde GitHub)
  2. !pip install triton torch  (Colab ya tiene torch, verificar triton)
  3. Cambiar al directorio del repo: import os; os.chdir("KernelForge")
  4. !python notebooks/run_eval_colab.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Para que importe benchmark sin instalar el paquete
NOTEBOOK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(NOTEBOOK_DIR))

from benchmark import evaluate_entry, load_jsonl, load_t_simple_entries

PROJECT_ROOT = NOTEBOOK_DIR.parent
DATA_DIR = NOTEBOOK_DIR / "data"
TRITONBENCH_ROOT = PROJECT_ROOT / "vendor" / "TritonBench"
OUTPUT_PATH = NOTEBOOK_DIR / "eval_results.json"


def main() -> None:
    print("Cargando dataset TritonBench...")
    T_simple, errors, _, _ = load_t_simple_entries(TRITONBENCH_ROOT)
    entries_by_file = {entry["file"]: entry for entry in T_simple}
    entries_by_index = {i: entry for i, entry in enumerate(T_simple)}
    print(f"  {len(T_simple)} entries cargados, {len(errors)} errores")

    jsonl_paths = sorted(DATA_DIR.glob("*.jsonl"))
    if not jsonl_paths:
        print(f"ERROR: No se encontraron archivos .jsonl en {DATA_DIR}")
        sys.exit(1)

    raw_results: list[dict] = []
    for path in jsonl_paths:
        rows = load_jsonl(path)
        print(f"  {path.name}: {len(rows)} resultados")
        raw_results.extend(rows)

    print(f"\nEvaluando {len(raw_results)} kernels...\n")

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

        status = "call✓ exe✓" if result.get("exe@1") else ("call✓" if result.get("call@1") else "✗")
        print(f"[{i:3}/{len(raw_results)}] {model_label:20} {entry_file:50} {status}")

    print("\n" + "=" * 60)
    print("RESULTADOS POR MODELO")
    print("=" * 60)
    for label, s in sorted(model_stats.items()):
        n = s["total"]
        call_pct = 100 * s["call_pass"] / n if n else 0
        exe_pct = 100 * s["exe_pass"] / n if n else 0
        print(f"\n{label}")
        print(f"  Total kernels : {n}")
        print(f"  call@1        : {s['call_pass']}/{n} ({call_pct:.1f}%)")
        print(f"  exe@1         : {s['exe_pass']}/{n} ({exe_pct:.1f}%)")

    output = {"model_stats": model_stats, "per_kernel": eval_results}
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
