# Tests

Repository-level tests should live here once behavior graduates from notebooks.

Suggested layout:

- `tests/benchmark/`: dataset loading, prompt construction, result parsing (CPU-only;
  see `tests/benchmark/fixtures/sample_inference.jsonl`).
- `tests/grammar/`: constrained grammar fixtures against real Triton snippets and
  plain accept/reject strings.
- `tests/compiler/`: LEX/YACC accept/reject syntax fixtures.
- `tests/semantic_checker/`: warnings for generated Triton anti-patterns.

Prefer small CPU-only tests by default. GPU/Triton execution tests should be
opt-in because they require a compatible local or Colab runtime.
