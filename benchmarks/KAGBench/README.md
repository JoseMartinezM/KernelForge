# KAGBench

KAGBench (Kernel Agentic Bench) is a small, self-contained benchmark dataset for
agentic Triton kernel synthesis experiments.

Each case is stored as a directory under `cases/`. Multiline Python fields are
kept as standalone `.py` files for readability and to avoid JSON escaping issues.
The fused machine-readable ledger is `kagbench.jsonl`.

## Case source layout

```text
cases/<case_id>/
├── manifest.json          # small metadata fields
├── prompt.md              # user-facing task prompt
├── pytorch_reference.py   # reference implementation available to the model
├── public_tests.py        # visible/basic tests
├── unit_tests.py          # hidden/scoring tests
└── source.py              # vendored original TritonBench-G source
```

## Manifest shape

```json
{
  "id": "tritonbench_g/swiglu_fwd",
  "source_file": "vendor/TritonBench/data/TritonBench_G_v1/swiglu_fwd.py",
  "entry_file": "swiglu_fwd.py",
  "tags": ["elementwise", "activation"]
}
```

## Rebuild the ledger

```bash
uv run python benchmarks/KAGBench/scripts/fuse_all.py
```

To import a newly generated temporary case directory:

```bash
uv run python benchmarks/KAGBench/scripts/import_case.py /tmp/kagbench_case_xxx
```

The import script copies the case files into `cases/`, vendors the source file as
`source.py`, and rebuilds `kagbench.jsonl`.
