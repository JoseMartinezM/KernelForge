# KernelForge Agent

Pi Agent package for the KernelForge validated Triton kernel loop.

## What It Does

1. Generates a Triton kernel for a TritonBench task.
2. Optionally searches local Triton guidance before generation.
3. Optionally constrains generation with `grammar/triton.gbnf` through XGrammar.
4. Runs a fast CPU-only semantic check.
5. Validates the kernel on a Modal T4 GPU.
6. Writes the result to `runs/agent/`.

## Setup

Run once:

```powershell
# Authenticate Modal for GPU validation.
uv run modal token new

# Install Node dependencies.
npm install
```

Set the credentials needed for each session:

```powershell
$env:ANTHROPIC_API_KEY = "..."   # Pi Agent brain

# Lightning-hosted models
$env:LIGHTNING_API_KEY = "..."

# Modal-hosted Gemma E4B
$env:MODAL_API_KEY = "..."
$env:MODAL_API_SECRET = "..."
```

## Run

```powershell
npx pi
```

Example prompts:

```text
Search Triton docs for masked softmax, then generate and validate softmax.py with google/gemma-4-E4B-it using grammar constraints.
Generate and validate a tanh kernel using lightning-ai/deepseek-v4-pro.
```

## Tools

- `search_triton_docs`: returns concise local Triton guidance for common kernel patterns.
- `generate_kernel`: calls `scripts/generate_kernel.py`; pass `use_grammar: true` to use `grammar/triton.gbnf` with XGrammar.
- `run_semantic_check`: runs the CPU-only semantic checker before GPU validation.
- `validate_kernel`: evaluates the generated kernel on Modal T4 and returns `call@1`, `exe@1`, mismatches, and semantic warnings.
- `write_ledger`: appends results to `runs/agent/{model}.jsonl`.

## Models

| Model | Provider | Required key |
|---|---|---|
| `google/gemma-4-E4B-it` | Modal | `MODAL_API_KEY` + `MODAL_API_SECRET` |
| `lightning-ai/deepseek-v4-pro` | Lightning | `LIGHTNING_API_KEY` |
| `openai/gpt-5.4-2026-03-05` | Lightning | `LIGHTNING_API_KEY` |
| `lightning-ai/gemma-4-31B-it` | Lightning | `LIGHTNING_API_KEY` |

## Results

Each processed kernel is saved under `runs/agent/` with generated code, `call@1`,
`exe@1`, `semantic_warnings`, and `attempts`.
