# KernelForge agent notes

Keep this file short. Prefer linking to the canonical docs below instead of duplicating project details.

## Language

- Write all code, comments, commit-facing text, and documentation in English.

## Canonical docs
Only read these docs when relevant to the task.

- Project overview, repository map, credentials, and common commands: `README.md`.
- Detailed setup, provider workflows, Modal notes, constrained-decoding notes, and troubleshooting: `docs/onboarding.md`.
- Ownership boundaries and migration rule: `docs/architecture.md`.
- Dataset, inference-ledger, evaluation-result, and filtering guide: `docs/data-and-runs.md`.
- Generated-kernel evaluation notes: `docs/evaluate.md`.
- Current work checklist and near-term priorities: `docs/checklist.md`.

## Environment

- Use `uv` as the Python environment and command runner. Run commands from the repository root unless a doc says otherwise.
- Python requirement: `>=3.11` from `pyproject.toml`.
- Base setup: `uv sync`.
- Local ROCm/Triton execution setup: `uv sync --extra rocm`.
- Optional Nix shells: `nix develop .#default` or `nix develop .#rocm`; the flake provides `uv` and sets `UV_PROJECT_ENVIRONMENT="$PWD/.venv"`.
- Keep secrets in local environment variables or git-ignored `.env*` files only. Relevant variables are documented in `README.md` and `docs/onboarding.md`.

## Common commands

```bash
# List configured models and provider wiring before starting inference work.
uv run python -m kernelforge.benchmark.llm_inference --list-models

# Build one TritonBench-T prompt without calling a model; use this to check credentials and prompt shape.
uv run python -m kernelforge.benchmark.llm_inference --model lightning-ai/gemma-4-31B-it --limit 1 --dry-run --dry-run-preview 1

# Serve the Modal vLLM backend locally while iterating on the Gemma 4 Modal provider.
uv run modal serve scripts/modal_vllm.py

# Run the project linter.
uv run ruff check .

# Run static type checks for the reusable package under src/.
uv run basedpyright
```

For Marimo notebooks, use `uv run marimo edit notebooks/<name>.py` for the interactive browser UI. **this command is blocking**.
For agent-readable execution, use `python3 notebooks/<name>.py`; this prints stdout and runs side effects, but charts, tables, and other visual UI elements are not shown.

## Repo-specific cautions

- Treat `vendor/TritonBench/` as read-mostly upstream code.
- Store generated inference/evaluation ledgers under `runs/`.
- Move reusable notebook code into `src/kernelforge/`; keep notebooks as interfaces.
- The current target is a lightweight validated Triton kernel loop, not a broad application shell. See `docs/checklist.md` before expanding scope.
