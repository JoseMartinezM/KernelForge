# Dataset and run navigation

This guide is for quickly answering questions about TritonBench tasks, inference
ledgers, and generated-kernel evaluation results. Prefer these paths and helper
APIs over ad-hoc parsing.

## Canonical paths

- TritonBench-T task metadata: `vendor/TritonBench/data/TritonBench_T_v1.jsonl`.
  Despite the suffix, this is one JSON list with 166 task objects.
- TritonBench-T simple Alpaca prompts: `vendor/TritonBench/data/TritonBench_T_simp_alpac_v1.json`.
- TritonBench-T reference/test files: `vendor/TritonBench/data/TritonBench_T_v1/<entry_file>`.
- Current inference ledgers: `runs/tritonbench/*.jsonl`.
- Historical notebook inference ledgers used by `notebooks/tritonbench.py`:
  `notebooks/data/*.jsonl`.
- Historical aggregate evaluation results: `notebooks/results/eval_results.json`,
  plus CSV exports in `notebooks/results/`.

## Loader APIs

Use `kernelforge.benchmark` from the repository root:

```python
from kernelforge.benchmark import load_analysis_table, load_jsonl, load_t_simple_entries

entries, errors, t_json, simple_alpaca = load_t_simple_entries("vendor/TritonBench")
assert not errors

rows = load_jsonl("runs/tritonbench/modal-gemma4-e4b-t-simple-max8000-temp0.jsonl")
table = load_analysis_table("runs/tritonbench/modal-gemma4-e4b-t-simple-max8000-temp0.jsonl")
```

`load_t_simple_entries()` returns prepared entries with `source_path`, `ref_code`,
and `test_code`, split from the reference file at the TritonBench separator.
`load_analysis_table()` flattens inference ledgers into filter-friendly rows with
token counts, cost estimates, syntax flags, truncation flags, and static code
metrics.

## Dataset entry shape

Each TritonBench-T metadata entry has fields such as:

- `file`: join key and reference filename, for example `softmax.py`.
- `name`: function/task name.
- `func_inputs`: wrapper signature and argument documentation.
- `description`, `math`, `other`: prompt context.
- `torch_code`: PyTorch reference implementation shown to the model.
- `difficulty`, `params_cnt`, `torch_cnt`, `example`: task metadata.

The simple task set currently has 166 entries. `entry_index` in inference ledgers
is the 0-based index into the prepared simple entries; `entry_file` is the stable
join key and usually the safer field to use across ledgers and evaluations.

## Inference ledger shape

Inference JSONL rows in `runs/tritonbench/` and `notebooks/data/` usually include:

- task identity: `entry_index`, `entry_file`, `request_hash`;
- model/provider: `model`, `model_label`, `provider`, `provider_url`;
- status/timing: `status`, `attempt`, `max_attempts`, `started_at`, `finished_at`,
  `latency_s`;
- prompt/output: `messages`, `content`, `generation`;
- OpenAI-compatible response: `response.choices[0].finish_reason` and
  `response.usage.prompt_tokens`, `completion_tokens`, `total_tokens`.

Use `finish_reason == "length"` or the flattened `truncated` field from
`load_analysis_table()` to find token-cap truncation. Failed rows can lack
`response.usage`; cost helpers therefore report zero or `None` for missing usage,
even though a provider may still have incurred overhead.

## Evaluation result shape and semantics

`notebooks/results/eval_results.json` has:

- `model_stats`: per-model totals. The historical file contains 166 rows each for
  DeepSeek V4 Pro, GPT 5.4, and Modal Gemma 4 E4B vLLM.
- `per_kernel`: 498 evaluated model/task rows with `file`, `model`,
  `model_label`, `call@1`, `exe@1`, `mismatches`, `pred`, and `ref`.

`call@1` means the generated script subprocess exited successfully. `exe@1` means
the generated outputs matched the reference outputs. `mismatches` records tensor,
shape, dtype, structure, or scalar differences. `mismatches` containing
`"reference execution failed"` means the TritonBench reference failed in the
current harness/environment; do not count that as a clean model failure without
inspection.

The evaluator runs the generated code and reference code in separate subprocesses
and saves `test_results` via `torch.save`. Avoid returning custom classes or
custom `NamedTuple`s from generated code because parent-process `torch.load` can
fail to unpickle `__main__` classes. Prefer tensors, dicts, tuples, lists, and
plain scalars.

## Common filters and joins

Join inference rows to evaluations by `entry_file`/`file` and `model`:

```python
import json
from collections import defaultdict
from pathlib import Path

from kernelforge.benchmark import load_jsonl

eval_results = json.loads(Path("notebooks/results/eval_results.json").read_text())
inference_runs = []
for path in sorted(Path("notebooks/data").glob("*.jsonl")):
    inference_runs.extend(load_jsonl(path))

runs_by_file = defaultdict(dict)
for run in inference_runs:
    runs_by_file[run["entry_file"]][run["model"]] = run

for eval_row in eval_results["per_kernel"]:
    run = runs_by_file[eval_row["file"]].get(eval_row["model"])
```

Useful one-off filters:

```python
from pathlib import Path

from kernelforge.benchmark import load_analysis_table

table = load_analysis_table(sorted(Path("runs/tritonbench").glob("*.jsonl")))

truncated = [row for row in table if row["truncated"]]
syntax_errors = [row for row in table if not row["syntax_ok"]]
torch_fallbacks = [row for row in table if row["torch_call_count"] > 0]
by_case = [row for row in table if row["entry_file"] == "softmax.py"]
by_model = [row for row in table if row["model"] == "google/gemma-4-E4B-it"]
```

For result triage, group filenames by families before reading individual rows:
`softmax`/`cross_entropy`, `matmul`/`bmm`/`dot`, `conv`, `svd`/`lu`/`qr`/`eig`,
reductions, elementwise ops, and random/dropout tasks. Known frequent failure
buckets include wrapper/signature mismatch, launch-grid shape mistakes, Triton
compile errors, numerical mismatch, reference execution failures, and PyTorch
fallback/benchmark-cheating patterns.

## Prior analysis facts worth preserving

- Historical aggregate results cover 498 evaluated generations: 166 tasks across
  three models. Summary: DeepSeek V4 Pro `call@1=58`, `exe@1=32`; GPT 5.4
  `call@1=96`, `exe@1=56`; Modal Gemma 4 E4B `call@1=40`, `exe@1=21`.
- `max_tokens=600` caused frequent Gemma truncation; `max_tokens=3000` still
  truncated some calibration rows; the Modal full run used `max_tokens=8000` and
  completed all 166 rows with `finish_reason="stop"`.
- Models sometimes pass execution by calling PyTorch operations inside the wrapper
  instead of doing the core work in Triton. Look for `torch_call_count > 0`,
  especially on complex linalg/broadcasting tasks such as SVD/LU/eig/pseudoinverse
  and fused high-dimensional ops.
- Some mismatches are harness artifacts: unseeded random inputs can differ between
  isolated subprocesses, and some reference implementations fail under the local
  environment. Inspect `pred.stderr`, `ref.stderr`, and `mismatches` before drawing
  benchmark conclusions.
