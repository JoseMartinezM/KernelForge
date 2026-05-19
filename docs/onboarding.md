# KernelForge onboarding

KernelForge is currently a research workspace for improving LLM code generation for Triton GPU kernels with constrained decoding. The repo is not yet a packaged application; the reliable entry points today are the uv-managed Python environment, the vendored TritonBench data/evaluation suite, and marimo notebooks for experiments.

## Prerequisites

- `uv` for Python environment management. Install it with the official installer if it is not already available:
  - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Python 3.11 or newer. `uv` can download a compatible Python if your system Python is too old.
- A `LIGHTNING_API_KEY` only when running notebook cells that call the Lightning AI API.
- A supported Linux GPU stack only when compiling/running Triton kernels locally.

## What is in the repo

- `pyproject.toml` / `uv.lock`: Python environment definition. Base dependencies include marimo, OpenAI, numpy, ruff, basedpyright, and ipykernel.
- `pyproject.toml` extra `rocm`: AMD ROCm PyTorch/Triton wheels from the PyTorch ROCm index.
- `notebooks/benchmark.py`: marimo notebook that loads TritonBench-T, builds prompts, calls the Lightning AI OpenAI-compatible API, and displays cleaned model output.
- `vendor/TritonBench`: vendored benchmark data, generated model outputs, upstream evaluation scripts, and performance metrics.
- `flake.nix`: optional Nix development shells that provide `uv` and, in the `rocm` shell, ROCm tools/libraries.

## Platform quick start

Always run commands from the repository root. Prefer `uv run ...` over manually activating the venv; it keeps the command tied to the locked project environment.

### Linux or macOS, CPU-only notebook work

Use this for dataset inspection, prompt development, and LLM calls that do not execute Triton kernels locally.

```bash
uv sync
export LIGHTNING_API_KEY="..."
uv run marimo edit notebooks/benchmark.py
```

### Linux with AMD ROCm GPU

Use this when you need local PyTorch/Triton execution on a supported AMD GPU.

```bash
uv sync --extra rocm
export LIGHTNING_API_KEY="..."
uv run python - <<'PY'
import torch
print("torch", torch.__version__)
print("hip", torch.version.hip)
print("gpu available", torch.cuda.is_available())
PY
uv run marimo edit notebooks/benchmark.py
```

### NixOS or Linux with Nix

The flake supplies `uv`; the `rocm` shell also supplies ROCm tools. If you use direnv, `.envrc` already selects `.#rocm` and loads `.env.dev`.

```bash
nix develop .#default        # CPU-only notebook work
# or
nix develop .#rocm           # ROCm work

uv sync                      # CPU-only dependencies
# or
uv sync --extra rocm         # ROCm dependencies

uv run marimo edit notebooks/benchmark.py
```

### Windows

Native Windows is useful for prompt/notebook development only. For GPU execution, use a Linux machine or WSL2 with a GPU stack supported by PyTorch/Triton.

```powershell
uv sync
$env:LIGHTNING_API_KEY = "..."
uv run marimo edit notebooks/benchmark.py
```

## Working with the benchmark notebook

`notebooks/benchmark.py` should be launched from the repo root because it uses relative paths under `vendor/TritonBench`.

The current flow is:

1. Load `vendor/TritonBench/data/TritonBench_T_simp_alpac_v1.json`.
2. Cross-reference entries with `vendor/TritonBench/data/TritonBench_T_v1.jsonl` to recover full metadata, reference PyTorch code, and function signatures.
3. Build a compact prompt from the functional description, wrapper signature, math notes, and reference PyTorch implementation.
4. Call the Lightning AI OpenAI-compatible endpoint using `LIGHTNING_API_KEY`.
5. Cache LLM responses with marimo persistent cache to avoid repeated API spend.
6. Parse/display generated Python source for inspection.

If the notebook cannot find data files, check that you started marimo from the repository root, not from `notebooks/`.

## Working with TritonBench

Useful paths:

- `vendor/TritonBench/data/TritonBench_T_*`: Triton operator generation tasks. This is the first target for KernelForge experiments.
- `vendor/TritonBench/data/TritonBench_G_*`: general-purpose Triton tasks with a separate evaluation flow.
- `vendor/TritonBench/EVAL/eval_T`: upstream call accuracy, execution accuracy, and efficiency scripts for TritonBench-T.
- `vendor/TritonBench/performance_metrics/perf_T`: upstream performance metric helpers and golden reports.

Treat the vendored evaluation scripts as upstream reference code, not turnkey project commands. Before using them for an experiment, review and adapt hardcoded interpreter paths, GPU selection, and expected input/output folders. Keep generated experiment outputs outside `vendor/` unless you intentionally want to vendor them.

## Common commands

```bash
# Start the notebook UI
uv run marimo edit notebooks/benchmark.py

# Run the notebook as a marimo app
uv run marimo run notebooks/benchmark.py

# Static checks for Python files
uv run ruff check .
uv run basedpyright
```

## Troubleshooting

- `LIGHTNING_API_KEY` assertion fails: export the variable in the shell that launches marimo, or put it in `.env.dev` if using direnv.
- Triton/PyTorch import or GPU checks fail: confirm you installed the correct extra (`uv sync --extra rocm`) and that the host GPU driver/runtime is visible before debugging KernelForge code.
- marimo opens but relative file reads fail: restart it from the repo root.
- TritonBench eval scripts fail immediately: inspect hardcoded paths first; several scripts were copied from upstream with machine-specific defaults.
