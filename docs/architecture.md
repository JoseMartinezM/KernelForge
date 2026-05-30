# KernelForge architecture

KernelForge is a monorepo for a research project, a course compiler component,
and a prototype differentiator around validated Triton kernel generation.

## Current shape

```diagram
╭────────────────────╮     ╭─────────────────────╮
│ notebooks/         │────▶│ src/kernelforge/    │
│ marimo exploration │     │ reusable utilities  │
╰─────────┬──────────╯     ╰──────────┬──────────╯
          │                           │
          ▼                           ▼
╭────────────────────╮     ╭─────────────────────╮
│ grammar/           │     │ vendor/TritonBench  │
│ GBNF experiments   │     │ benchmark data      │
╰────────────────────╯     ╰─────────────────────╯
          │                           ▲
          ▼                           │
╭────────────────────╮     ╭─────────────────────╮
│ apps/              │────▶│ runs/               │
│ agent-loop adapters│     │ generated ledgers   │
╰────────────────────╯     ╰─────────────────────╯
          ▲
          │
╭────────────────────╮
│ scripts/           │
│ Modal/inference ops│
╰────────────────────╯
```

## Ownership boundaries

- `src/kernelforge/benchmark/`: reusable Python utilities for TritonBench data,
  prompts, inference ledgers, lightweight evaluation, and model/provider config.
- `notebooks/`: marimo notebooks for exploration, visualization, and manual
  analysis. Notebooks may be messy; reusable functions should move into `src/`.
- `grammar/`: grammar assets, generated export lists, viewers, and fixtures for
  constrained generation experiments.
- `compiler/`: the LEX/YACC syntax validator for the course requirement.
- `scripts/`: operational one-file entry points, currently the Modal vLLM app.
- `apps/`: future multi-file agent-loop adapters. The first candidate to study is
  Pi Agent and its plugin system.
- `vendor/TritonBench/`: upstream benchmark data and scripts. Treat as read-mostly.
- `runs/`: generated inference/evaluation ledgers. This is intentionally ignored.

## Near-term direction

The current target is a validation loop, not a broad application shell. The loop
should accept a generated Triton kernel, run a basic semantic checker, validate it
against known-good kernels and values, and record reproducible metrics. Pi Agent
is the main candidate for orchestrating this loop once its plugin architecture is
understood.

The constrained-decoding backend remains a design dependency for the grammar.
LLGuidance is the current preferred option because it supports grammar testing
against strings and can be integrated through llama.cpp. The caveat is deployment:
llama.cpp needs a custom build with `-DLLAMA_LLGUIDANCE=ON` and Rust available,
which must be made reproducible and cost-conscious on Modal.

## Near-term migration rule

When code in a notebook becomes useful from a CLI, a test, or another notebook,
move it into `src/kernelforge/` and leave the notebook as an interface. That keeps
the research flexible while making the reusable package stable enough for tests,
validation, and the eventual agent-loop demo.
