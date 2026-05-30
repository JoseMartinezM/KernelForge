# KernelForge checklist

Each item should correspond to a GitHub issue. Before starting work, open the
issue, assign it to yourself, and reference it in the PR. When the work lands,
mark the item as `[x]` here and close the issue.

---

## Kernel-development agent loop

- [ ] **Evaluate Pi Agent as the main agent-loop candidate**
  - Study its architecture, extension points, and plugin system.
  - Confirm whether plugins can call KernelForge benchmark loaders, semantic
    checks, constrained decoding, and validation commands.
  - Document compatibility risks before adding adapters under `apps/`.

- [ ] **Design the MVP validation loop**
  - Input: a TritonBench task and a generated candidate kernel.
  - Steps: semantic check, known-good value validation, benchmark metrics, ledger
    write.
  - Output: a compact result object that an agent plugin can consume.

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
  - XGrammar: strong constrained decoding, but the practical Python path targets
    Hugging Face models and production engines such as vLLM or SGLang add Modal
    complexity and cost.
  - Plain llama.cpp GBNF: useful for local experiments, but lacks a clean Python
    reference parser library for testing the grammar against example kernels.
  - LLGuidance: current preferred option because grammars can be unit-tested
    against strings and integrated through llama.cpp, with the caveat below.

- [ ] **Investigate reproducible LLGuidance deployment on Modal**
  - llama.cpp must be compiled with `-DLLAMA_LLGUIDANCE=ON` and a Rust toolchain.
  - Check whether a maintained prebuilt binary, image, or community Modal recipe
    already exists.
  - If not, prototype a custom Modal image and measure cold-start/build impact
    before adopting LLGuidance as the default path.

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
- [ ] **`tests/benchmark/` fixtures** — benchmark loading, prompt construction,
  result parsing, and validation metadata remain CPU-testable.

---

## Team workflow

1. Pick an item from this list that nobody has assigned.
2. Open a GitHub issue with the same item name.
3. Mark the item as `[x]` in this file in the same PR that completes the work.
