# KernelForge

KernelForge is a research monorepo for improving LLM generation of Triton
GPU kernels with constrained generation and validation. The repo combines
benchmark tooling, exploratory notebooks, grammar experiments, a course LEX/YACC
compiler component, inference backend experiments, and the planned
kernel-development agent loop.

The current project is research-first.
Reusable Python code lives under `src/kernelforge`; notebooks and scripts are
interfaces around that package.

## Project goals

- Measure how different LLMs perform on niche Triton kernel generation tasks.
- Validate generated kernels against known-good Triton kernels and their expected
  values before treating results as usable benchmark data.
- Design constrained-generation grammars that make invalid or non-existent Triton
  APIs harder for the model to emit.
- Build toward a differentiator: an agent loop for developing kernels. The
  current implementation uses **Pi Agent** tools under `apps/agent/`.
- Ship a first MVP validation loop with a basic semantic checker before expanding
  the agent workflow.
- Implement a LEX/YACC syntax validator for the course requirement.
- Keep benchmark, inference, evaluation, and notebook workflows reproducible.

## Current direction

The next milestone is not a broad product surface. It is a validated kernel
development loop:

1. Generate candidate Triton kernels.
2. Run a basic semantic checker over obvious Triton mistakes.
3. Validate candidates against known-good kernels and values.
4. Record benchmark results in reproducible ledgers.

Pi Agent is the current agent-loop adapter. Its tools call KernelForge scripts
for task loading, semantic checks, optional constrained decoding, Modal
validation, and result ledger writes without forcing a large repo reshaping.

The constrained-decoding path is now centered on XGrammar through the Modal vLLM
backend's structured-output support. `grammar/triton.gbnf` is the active grammar
artifact for those experiments; plain llama.cpp GBNF remains useful for local
debugging, but the production smoke tests should exercise the vLLM/XGrammar path.

## Repository map

```text
.
├── compiler/                # Course LEX/YACC syntax-validator home
├── docs/                    # Onboarding, evaluation, architecture notes
├── grammar/                 # GBNF grammar assets and viewers
├── notebooks/               # Marimo exploration and visualization notebooks
├── scripts/                 # Operational one-file scripts, e.g. Modal inference
├── src/kernelforge/         # Installable reusable Python package
│   ├── benchmark/           # TritonBench loaders, prompts, inference, results
│   └── grammar/             # Python grammar/constrained-generation utilities
├── tests/                   # Future CPU-first test suites
├── apps/                    # Agent-loop adapters and multi-file surfaces
├── vendor/TritonBench/      # Vendored upstream benchmark data/scripts
└── runs/                    # Generated ledgers/results; git-ignored
```

For the rationale behind these boundaries, see `docs/architecture.md`.

## Setup

```bash
uv sync
```

For local ROCm/PyTorch/Triton execution:

```bash
uv sync --extra rocm
```

If you use Nix/NixOS:

```bash
nix develop .#default
# or, for local ROCm execution
nix develop .#rocm
```

## Common commands

```bash
# List configured inference models
uv run python -m kernelforge.benchmark.llm_inference --list-models

# One-row dry run against the configured benchmark prompt builder
uv run python -m kernelforge.benchmark.llm_inference \
  --model lightning-ai/gemma-4-31B-it \
  --limit 1 \
  --dry-run \
  --dry-run-preview 1

# Modal Gemma backend
uv run modal serve scripts/modal_vllm.py

# Interactive notebooks
uv run marimo edit notebooks/grammar.py
uv run marimo edit notebooks/tritonbench.py

# Static checks
uv run ruff check .
uv run basedpyright
```

Use `kernelforge.benchmark` for reusable benchmark imports and CLI entry points.

## Credentials

Set only the provider credentials needed for the workflow you are running:

```bash
export LIGHTNING_API_KEY="..."
export MODAL_API_KEY="..."      # or MODAL_API_TOKEN
export MODAL_API_SECRET="..."
export GOOGLE_API_KEY="..."     # older Google AI Studio notebook path only
```

Keep local `.env*` files uncommitted.

## Current stable entry points

- `kernelforge.benchmark.tritonbench`: TritonBench-T loading, prompt construction,
  generated-code cleanup, and lightweight local evaluation helpers.
- `kernelforge.benchmark.llm_inference`: resumable OpenAI-compatible batch
  inference CLI over TritonBench-T simple tasks.
- `kernelforge.benchmark.llm_results`: JSONL loading, syntax metrics, cost
  estimates, and notebook-friendly tables.
- `scripts/modal_vllm.py`: Modal deployment for the Gemma 4 E4B vLLM backend.
- `apps/agent/`: Pi Agent tools for generating, checking, validating, and logging
  kernels.
- `grammar/triton.gbnf`: current XGrammar/vLLM constrained-decoding grammar
  experiment.

Generated inference and evaluation ledgers should go under `runs/`.

## More documentation

- `docs/onboarding.md`: detailed setup, provider, inference, and troubleshooting
  workflow.
- `docs/evaluate.md`: Colab/local evaluation notes for generated kernels.
- `docs/architecture.md`: monorepo boundaries and migration rule.
