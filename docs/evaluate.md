# Evaluate pipeline

## Archivos

- `notebooks/evaluate/run_eval.py` — evalúa los kernels generados contra TritonBench. Requiere Linux + GPU. Guarda resultados en `notebooks/results/eval_results.json`.
- `notebooks/evaluate/visualize.py` — visualiza los resultados en el browser. Corre en Windows.

## Correr la evaluación (Google Colab)

1. Abre [colab.research.google.com](https://colab.research.google.com), cambia el runtime a **T4 GPU**.

2. Pega estas celdas en orden:

```python
!git clone https://github.com/JoseMartinezM/KernelForge.git
import os; os.chdir("KernelForge")
```
```python
!pip install triton -q
```
```python
!python notebooks/evaluate/run_eval.py
```

3. Descarga `notebooks/results/eval_results.json` desde el panel de archivos de Colab y ponlo en `notebooks/results/` en tu máquina.

## Ver los resultados (Windows)

```powershell
uv run marimo edit notebooks/evaluate/visualize.py
```
