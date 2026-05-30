# Evaluate pipeline

## Files

- `notebooks/evaluate/run_eval.py` — evaluates generated kernels against
  TritonBench. Requires Linux and a compatible GPU. Saves results to
  `notebooks/results/eval_results.json`.
- `notebooks/evaluate/visualize.py` — visualizes results in a browser. This can
  run on Windows.

## Run evaluation in Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com) and switch
   the runtime to **T4 GPU**.

2. Paste these cells in order:

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

3. Download `notebooks/results/eval_results.json` from Colab's file panel and put
   it in `notebooks/results/` on your machine.

## View results on Windows

```powershell
uv run marimo edit notebooks/evaluate/visualize.py
```
