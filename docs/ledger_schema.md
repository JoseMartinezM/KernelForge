# KernelForge Validation Ledger Schema

This document defines the JSONL schema for KernelForge validation results stored in `runs/`.
Each row is a JSON object representing one generation/evaluation result.

## Schema overview

Each record has the following fields:

- `schema_version` (int)
  - Ledger schema version. Current value: `1`.

- `request_hash` (str)
  - Stable hash of the request payload used for deduplication and resume.

- `status` (str)
  - Either `"success"` or `"failed"`.

- `entry_index` (int)
  - Index of the dataset entry used for this task.

- `entry_file` (str)
  - Filename or path of the entry from the benchmark dataset.

- `model` (str)
  - Model identifier from the LLM config.

- `model_label` (str)
  - Human-friendly model label.

- `provider` (str)
  - Provider identifier from the LLM config.

- `provider_url` (str)
  - Provider endpoint URL.

- `generation` (object)
  - Generation hyperparameters / request settings used for the model.

- `messages` (list[object])
  - Chat messages sent to the model.

- `content` (str | null)
  - Generated code returned by the model.
  - For failed generation rows this field is `null`.

- `semantic_warnings` (list[str])
  - Semantic checker warnings produced from the generated code.
  - Always present; empty list means no warnings.

- `call@1` (bool | null)
  - Whether the generated kernel passed the functional call test.
  - `null` means the evaluation step has not yet populated this field.

- `exe@1` (bool | null)
  - Whether the generated kernel executed successfully.
  - `null` means the evaluation step has not yet populated this field.

- `speedup` (number | null)
  - Measured speedup of the generated kernel versus the CPU baseline.
  - `null` means the evaluation step has not yet populated this field.

- `validation_errors` (list[str])
  - Validation or evaluation error messages.
  - Always present; empty list means no validation errors were recorded.

- `error_type` (str | null)
  - Type of exception when generation failed.
  - `null` when the request completed successfully.

- `error_message` (str | null)
  - Error message from the generation or evaluation step.
  - `null` when there is no error.

- `retryable` (bool)
  - Whether a failed request may be retried.

- `attempt` (int)
  - Current attempt count for the request.

- `max_attempts` (int)
  - Maximum number of retry attempts configured for the request.

- `started_at` (str)
  - ISO 8601 UTC timestamp when the request started.

- `finished_at` (str)
  - ISO 8601 UTC timestamp when the request finished.

- `latency_s` (number)
  - Round-trip latency in seconds.

## Notes

- `content` is the generated code for the task; this is the main payload consumed by evaluation.
- `semantic_warnings` is always stored as a list, even when empty.
- `call@1`, `exe@1`, and `speedup` are populated during evaluation and may remain `null` for pure generation rows.
- `validation_errors` is intended to record any kernel validation or evaluation failures beyond generation errors.

## Example row

```json
{
  "schema_version": 1,
  "request_hash": "...",
  "status": "success",
  "entry_index": 0,
  "entry_file": "example.py",
  "model": "gpt-4o",
  "model_label": "GPT-4o",
  "provider": "openai",
  "provider_url": "https://api.openai.com",
  "generation": {"max_tokens": 2048, "temperature": 0.0},
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "content": "import triton\n...",
  "semantic_warnings": [],
  "call@1": null,
  "exe@1": null,
  "speedup": null,
  "validation_errors": [],
  "error_type": null,
  "error_message": null,
  "retryable": false,
  "attempt": 1,
  "max_attempts": 3,
  "started_at": "2026-05-29T00:00:00+00:00",
  "finished_at": "2026-05-29T00:00:03+00:00",
  "latency_s": 3.0
}
```
