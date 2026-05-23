# KernelForge onboarding

KernelForge is a research workspace for generating and evaluating Triton GPU kernels with LLMs. It is not packaged as an application yet; the reliable entry points are the `uv`-managed Python environment, the `notebooks.benchmark` Python modules, the vendored TritonBench dataset/evaluation code, and a small set of marimo notebooks for inspection.

Run commands from the repository root unless a command explicitly says otherwise. Most paths in the benchmark helpers are relative to the repo root.

## Prerequisites

- `uv` for Python environment management.
  - Linux/macOS: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
- Python 3.11 or newer. `uv` can download a compatible interpreter if needed.
- Provider credentials for whichever inference backend you use:
  - Lightning AI: `LIGHTNING_API_KEY` for `lightning-ai/gemma-4-31B-it`, `lightning-ai/deepseek-v4-pro`, and `openai/gpt-5.4-2026-03-05` through the Lightning OpenAI-compatible API.
  - Modal: Modal CLI auth plus `MODAL_API_KEY` or `MODAL_API_TOKEN`, and `MODAL_API_SECRET`, for the proxy-authenticated Gemma 4 vLLM endpoint.
  - Google AI Studio: `GOOGLE_API_KEY` is only needed for the older `notebooks/compare_models.py` Google-client path.
- A supported Linux GPU stack only when locally compiling/running generated Triton kernels. LLM inference through Lightning or Modal does not require a local GPU.

## Repository map

- `pyproject.toml` / `uv.lock`: base Python environment. Base dependencies include marimo, OpenAI, Modal, numpy, ruff, basedpyright, and ipykernel.
- `pyproject.toml` extra `rocm`: AMD ROCm PyTorch/Triton wheels from the PyTorch ROCm index for local Triton execution.
- `notebooks/benchmark/llm_inference.py`: CLI and helper functions for batch LLM inference over TritonBench-T tasks.
- `notebooks/benchmark/llm_models.json`: provider/model registry, generation defaults, provider auth metadata, and Lightning pricing metadata.
- `notebooks/benchmark/tritonbench.py`: dataset loading, prompt construction, generated-code cleanup, and lightweight local call/execution eval helpers.
- `notebooks/benchmark/llm_results.py`: JSONL result loading, cost estimates, syntax/code metrics, and summary table helpers.
- `scripts/modal_vllm.py`: Modal deployment for the Gemma 4 E4B vLLM OpenAI-compatible backend.
- `notebooks/compare_models.py`: older marimo comparison notebook that still uses a Google AI Studio client alongside Lightning models.
- `vendor/TritonBench`: vendored benchmark data, upstream generated outputs, upstream evaluation scripts, and performance metrics.
- `flake.nix`: optional Nix development shells that provide `uv` and, in the `rocm` shell, ROCm tools/libraries.

Generated inference and evaluation ledgers go under `runs/`, which is git-ignored.

## Environment setup

Install the base environment:

```bash
uv sync
```

For local ROCm-based Triton/PyTorch execution:

```bash
uv sync --extra rocm
```

Set credentials in your shell, or create a local `.env`/`.env.dev` file if your shell tooling loads it. These files are git-ignored; never commit real secrets.

```bash
# Lightning-hosted models
export LIGHTNING_API_KEY="..."

# Modal proxy-authenticated endpoint
export MODAL_API_KEY="..."      # or MODAL_API_TOKEN
export MODAL_API_SECRET="..."

# Only for notebooks/compare_models.py's Google AI Studio path
export GOOGLE_API_KEY="..."
```

Windows PowerShell equivalent:

```powershell
$env:LIGHTNING_API_KEY = "..."
$env:MODAL_API_KEY = "..."
$env:MODAL_API_SECRET = "..."
$env:GOOGLE_API_KEY = "..."
```

Quick sanity checks:

```bash
uv run python -m notebooks.benchmark.llm_inference --list-models
uv run python -m notebooks.benchmark.llm_inference \
  --model lightning-ai/gemma-4-31B-it \
  --limit 1 \
  --dry-run \
  --dry-run-preview 1
```

The dry run should report `token_present: true` for the Lightning provider when `LIGHTNING_API_KEY` is set. Modal dry runs should report both required headers as present.

## NixOS or Linux with Nix

The flake supplies `uv`; the `rocm` shell also supplies ROCm tools. If you use direnv, keep machine-local secrets in a git-ignored env file.

```bash
nix develop .#default        # CPU-only notebook/inference work
# or
nix develop .#rocm           # local ROCm evaluation work

uv sync                      # base dependencies
# or
uv sync --extra rocm         # ROCm dependencies
```

## Running LLM inference batches

`notebooks.benchmark.llm_inference` loads the TritonBench-T simple task set, builds prompts, calls a configured OpenAI-compatible model, and appends one JSON object per request to a JSONL ledger.

List configured models:

```bash
uv run python -m notebooks.benchmark.llm_inference --list-models
```

Do a one-row dry run:

```bash
uv run python -m notebooks.benchmark.llm_inference \
  --model lightning-ai/gemma-4-31B-it \
  --limit 1 \
  --dry-run \
  --dry-run-preview 1
```

Run a small Lightning smoke test:

```bash
uv run python -m notebooks.benchmark.llm_inference \
  --model lightning-ai/gemma-4-31B-it \
  --limit 1 \
  --max-tokens 128 \
  --temperature 0 \
  --top-p 1 \
  --max-workers 1 \
  --stagger-seconds 0 \
  --max-attempts 1 \
  --out runs/tritonbench/smoke-gemma4-31b.jsonl \
  --force
```

Run the current Lightning Gemma 4 31B full-batch shape conservatively:

```bash
uv run python -m notebooks.benchmark.llm_inference \
  --model lightning-ai/gemma-4-31B-it \
  --max-tokens 4000 \
  --temperature 0 \
  --top-p 1 \
  --max-workers 1 \
  --stagger-seconds 5 \
  --max-attempts 2 \
  --out runs/tritonbench/gemma4-31b-t-simple-max4000-temp0.jsonl
```

`--resume` is enabled by default and skips already-successful request hashes in the target ledger. Use `--force` only when you intentionally want to rerun rows and append fresh results.

## Modal Gemma 4 backend

The Modal backend runs `google/gemma-4-E4B-it` behind a vLLM OpenAI-compatible server on an A100-80GB. The configured provider in `llm_models.json` uses Modal proxy auth headers rather than a bearer token.

### 1. Authenticate Modal locally

```bash
uv run modal token new
```

For inference through the deployed proxy-authenticated web endpoint, also set:

```bash
export MODAL_API_KEY="..."      # or MODAL_API_TOKEN
export MODAL_API_SECRET="..."
```

### 2. Serve or deploy the vLLM backend

For iterative development, serve it from the repo root:

```bash
uv run modal serve scripts/modal_vllm.py
```

For a persistent deployment:

```bash
uv run modal deploy scripts/modal_vllm.py
```

If Modal prints a different web URL than the one currently in `notebooks/benchmark/llm_models.json`, update the `modal.url` value before running `llm_inference` against it.

### 3. Check endpoint wiring

```bash
uv run python -m notebooks.benchmark.llm_inference \
  --model google/gemma-4-E4B-it \
  --limit 1 \
  --max-tokens 128 \
  --temperature 0 \
  --top-p 1 \
  --max-workers 1 \
  --stagger-seconds 0 \
  --max-attempts 1 \
  --timeout 1200 \
  --out runs/tritonbench/modal-gemma4-e4b-smoke.jsonl \
  --force
```

The first call can include Modal/vLLM startup time. Use a high timeout for smoke tests and the first batch request.

### 4. Run the Modal full batch

The current Modal container is configured for eight concurrent web inputs. A validated full-batch shape is:

```bash
uv run python -m notebooks.benchmark.llm_inference \
  --model google/gemma-4-E4B-it \
  --max-tokens 8000 \
  --temperature 0 \
  --top-p 1 \
  --max-workers 8 \
  --stagger-seconds 0 \
  --max-attempts 2 \
  --timeout 1200 \
  --out runs/tritonbench/modal-gemma4-e4b-t-simple-max8000-temp0.jsonl
```

The Modal provider currently has no project-side request/token rate limiter. Bound concurrency with `--max-workers` and keep it at or below the Modal app's `@modal.concurrent(max_inputs=...)` setting unless you are intentionally stress-testing.

## Evaluating generated kernels

For lightweight local checks, use `notebooks.benchmark.tritonbench.evaluate_entry` directly from Python. This runs the generated code and the reference code with the TritonBench test code, then compares saved outputs.

Example random subset eval:

```bash
uv run python - <<'PY'
import json, random
from pathlib import Path
from notebooks.benchmark.tritonbench import evaluate_entry, load_t_simple_entries

ledger = Path("runs/tritonbench/modal-gemma4-e4b-t-simple-max8000-temp0.jsonl")
out = Path("runs/tritonbench/modal-gemma4-e4b-random10-eval.jsonl")
entries, errors, *_ = load_t_simple_entries()
if errors:
    raise RuntimeError(errors)

rows = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
rows = [row for row in rows if row.get("status") == "success"]
random.Random(20260523).shuffle(rows)

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    for row in rows[:10]:
        entry = entries[row["entry_index"]]
        result = evaluate_entry(entry, row.get("content") or "", timeout=240)
        f.write(json.dumps({
            "entry_index": row["entry_index"],
            "entry_file": row["entry_file"],
            "call@1": result["call@1"],
            "exe@1": result["exe@1"],
            "mismatches": result.get("mismatches"),
        }) + "\n")
PY
```

The upstream scripts in `vendor/TritonBench/EVAL/eval_T` are useful references for call accuracy, execution accuracy, and performance evaluation, but they are not turnkey project commands. Review hardcoded interpreter paths, GPU selection, and expected folders before using them.

## Notebook workflows

Marimo notebooks are useful for inspection and interactive analysis:

```bash
uv run marimo edit notebooks/compare_models.py
uv run marimo edit notebooks/tritonbench.py
```

If a notebook cannot find data files, restart marimo from the repository root.

## Common commands

```bash
# Install/update dependencies
uv sync

# Include local ROCm execution dependencies
uv sync --extra rocm

# List configured inference models
uv run python -m notebooks.benchmark.llm_inference --list-models

# Static checks
uv run ruff check .
uv run basedpyright
```

## Troubleshooting

- Provider credentials are missing in dry-run output: export the needed variables in the shell that launches the command. Modal needs `MODAL_API_KEY` or `MODAL_API_TOKEN` plus `MODAL_API_SECRET`.
- Modal smoke test times out: use `--timeout 1200` for cold starts, confirm `modal serve`/`modal deploy` is healthy, and verify the configured provider URL matches the deployed endpoint.
- Modal requests fail with auth errors: run `uv run modal token new` for CLI auth and check the proxy-auth header env vars.
- Triton/PyTorch imports or GPU checks fail locally: confirm you installed the correct extra (`uv sync --extra rocm`) and that the host GPU driver/runtime is visible before debugging KernelForge code.
- Relative file reads fail: run commands from the repo root, not from `notebooks/` or `vendor/`.
- Vendored TritonBench eval scripts fail immediately: inspect hardcoded paths first; several scripts were copied from upstream with machine-specific defaults.
