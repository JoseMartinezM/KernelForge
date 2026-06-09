# KernelForge checklist

Each item should correspond to a GitHub issue. Before starting work, open the
issue, assign it to yourself, and reference it in the PR. When the work lands,
mark the item as `[x]` here and close the issue.

---

## Kernel-development agent loop

- [x] **Evaluate Pi Agent as the main agent-loop candidate**
  - Architecture: minimal TypeScript/Node.js harness, extensible via `.pi/extensions/*.ts`.
  - Plugin system: `pi.registerTool()` registers tools the LLM can call; tools invoke
    subprocesses via `pi.exec()`, so all existing Python CLIs are callable directly.
  - SDK mode: `createAgentSession()` supports fully programmatic loops with custom tools,
    no terminal UI required.
  - Compatibility verdict: no blockers. Pi calls KernelForge Python commands as subprocesses.
    Modal stays unchanged — Pi orchestrates locally and calls Modal via the existing CLI.
  - Extra credential needed: `ANTHROPIC_API_KEY` for the Pi agent brain (separate from
    Lightning and Modal keys already in use).
  - Agent code goes under `apps/agent/` as a Pi package.

- [x] **Design the MVP validation loop**
  - Workflow: generate kernel → semantic check → validate with GPU → write ledger.
  - Input: a TritonBench task (entry_file + prompt).
  - Steps: `generate_kernel` tool → `run_semantic_check` tool → `validate_kernel` tool →
    `write_ledger` tool.
  - Output: a JSONL row written to `runs/` with call@1, exe@1, semantic_warnings, and
    model/provider metadata.
  - Each step maps to one Pi tool in `apps/agent/extensions/`.

- [x] **Implement the agent tools in `apps/agent/`**
  - [x] `search_triton_docs` — returns concise local Triton guidance before generation.
  - [x] `generate_kernel` — calls `scripts/generate_kernel.py` with the task entry
    and returns the generated code. Supports optional `grammar/triton.gbnf`
    constrained generation through XGrammar.
  - [x] `run_semantic_check` — wraps
    `src/kernelforge/benchmark/semantic_checker.py` as a Pi tool.
  - [x] `validate_kernel` — runs the generated kernel against the TritonBench reference
    and returns call@1, exe@1, mismatches, and semantic_warnings. Requires GPU.
    - [x] Create `scripts/modal_eval.py` with a Modal T4 function that accepts
      kernel code and entry_file and runs the evaluation.
    - [x] Test the Modal eval function standalone with one known-good kernel.
    - [x] Wire the Modal eval function as the backend for the `validate_kernel` Pi tool
      in `apps/agent/extensions/validate-kernel.ts`.
  - [x] `write_ledger` — appends the result object as a JSONL row under
    `runs/agent/`.

---

## Validated benchmarks

- [ ] **Map known-good kernels and values**
  - Identify the reference TritonBench kernels and golden values used for the
    first validation set.
  - Keep GPU-dependent execution opt-in, but make metadata and fixture loading
    testable on CPU.

- [ ] **Define the validation ledger schema**
  - Record generated code, model/provider metadata, semantic warnings, call
    accuracy, execution accuracy, speedup when available, and validation errors.
  - Store generated ledgers under `runs/`.

---

## Constrained-decoding backend selection

- [ ] **Finalize backend requirements**
  - Must support unit tests against plain test strings containing Triton kernels.
  - Must provide constrained-decoding capabilities close enough to XGrammar for
    the planned grammar.
  - Must be deployable for inference on Modal with minimal startup and usage
    cost.

- [ ] **Document the backend trade study**
  - XGrammar: current path through the Modal vLLM structured-output backend.
  - Plain llama.cpp GBNF: useful for local experiments, but lacks a clean Python
    reference parser library for testing the grammar against example kernels.

---

## Grammar implementation and tests

- [ ] **Update the Triton grammar for the selected backend**
  - Include `@triton.jit` decorators.
  - Include binary operators in expressions (`+`, `*`, `<`, `==`, `//`).
  - Include `for` loops for tile loops.
  - Include `tl.constexpr` parameter annotations (`BLOCK_SIZE: tl.constexpr`).
  - Include qualified calls such as `tl.load`, `tl.store`, `tl.arange`,
    `tl.program_id`, and `tl.zeros`.

- [ ] **Validate the grammar against real TritonBench kernels**
  - Add CPU-only grammar fixtures under `tests/grammar/`.
  - Accept real Triton snippets and reject unrelated generic Python.
  - Document every rejected reference kernel and decide whether the failure is a
    grammar gap or an out-of-scope kernel pattern.
  - Represent indentation with explicit 4-space literals and bounded nesting;
    do not depend on virtual `INDENT`/`DEDENT` tokens.

---

## Semantic checker

- [ ] **Implement a basic semantic checker in `src/kernelforge/`**
  - Warn on `tl.load(...)` without `mask=`.
  - Warn on `tl.store(...)` without `mask=`.
  - Warn on kernel functions without `@triton.jit`.
  - Warn on `BLOCK_SIZE` usage without a `tl.constexpr` annotation.
  - Warn when no `tl.program_id` appears in the kernel.
  - Return warnings; do not reject the kernel at this stage.

- [ ] **Integrate semantic warnings into evaluation ledgers**
  - Run the checker before TritonBench evaluation.
  - Save `semantic_warnings: list[str]` in each JSONL result row.

---

## Metrics and paper tables

- [ ] **Compare constrained and unconstrained generation**
  - For each selected model, run batches with and without the selected grammar
    backend.
  - Compare `call@1`, `exe@1`, and speedup when execution data is available.

- [ ] **Summarize frequent semantic warnings by model**
  - Identify which Triton anti-patterns each LLM generates most often.
  - Add a notebook/table path for warning counts.

- [ ] **Generate paper-ready result tables automatically**
  - Input: `runs/tritonbench/*.jsonl`.
  - Output: Markdown and CSV tables ready to copy into the paper.

---

## Tests

- [ ] **`tests/grammar/` fixtures** — selected grammar accepts valid Triton kernels
  and rejects unrelated generic Python.
- [ ] **`tests/semantic_checker/` fixtures** — checker detects each documented
  anti-pattern.
- [x] **`tests/benchmark/` fixtures** — benchmark loading, prompt construction,
  result parsing, and validation metadata remain CPU-testable.

---

## Team workflow

1. Pick an item from this list that nobody has assigned.
2. Open a GitHub issue with the same item name.
3. Mark the item as `[x]` in this file in the same PR that completes the work.
