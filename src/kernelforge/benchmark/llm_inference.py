from __future__ import annotations

import argparse
import email.utils
import hashlib
import json
import math
import os
import random
import textwrap
import time
from collections import deque
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from .tritonbench import DEFAULT_TRITONBENCH_ROOT, load_t_simple_entries, make_prompt
from .semantic_checker import check_kernel

DEFAULT_LLM_CONFIG_PATH = Path(__file__).with_name("llm_models.json")
RATE_LIMIT_WINDOW_SECONDS = 60.0
DEFAULT_TOKEN_ESTIMATE_CHARS_PER_TOKEN = 3
TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
NON_RETRYABLE_ERROR_MARKERS = (
    "unsupportedparameter",
    "unsupportedrequestparameter",
    "unknownparameter",
    "unknownrequestparameter",
    "unrecognizedrequestargument",
    "invalidparameter",
    "notsupportedmaxtokens",
    "usemaxcompletiontokens",
)

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert GPU kernel engineer specializing in Triton, the open-source DSL \
    and compiler for writing high-performance, platform-agnostic GPU kernels.

    Your task: given a PyTorch reference implementation and a functional specification, \
    write an equivalent, self-contained Triton implementation.

    ## Triton Constraints
    - All kernel functions must use the `@triton.jit` decorator.
    - Access global memory exclusively through `tl.load` / `tl.store` with pointer \
    arithmetic. Never use Python-level indexing inside a `@triton.jit` kernel.
    - Tile computations using `tl.program_id` and `tl.arange`; expose block sizes as \
    `tl.constexpr` parameters.
    - Prefer native Triton ops (`tl.dot`, `tl.sum`, `tl.sqrt`, `tl.exp`, `tl.maximum`, \
    `tl.sigmoid`, etc.) over manual emulation.
    - Batch dimensions (e.g., B in BMM) must be handled via strides or grid flattening — \
    not by looping in Python.
    - Dtypes, broadcast semantics, and numerical defaults must match the reference exactly.

    ## Output Requirements
    - Output **only valid Python source code**. No Markdown fences, no prose, \
    no tests, no benchmarking code.
    - Begin your response with the import block: `import triton`, \
    `import triton.language as tl`, and any other required imports.
    - Define the public wrapper with **exactly** the name, parameters, defaults, \
    keyword-only arguments, and return structure given in the specification.
    - Include every `@triton.jit` kernel and helper function the wrapper depends on.
    - No core computation may silently fall back to PyTorch.
    """
)


def load_llm_config(config_path: str | Path = DEFAULT_LLM_CONFIG_PATH) -> dict[str, Any]:
    """Load provider/model metadata for LLM inference."""
    path = Path(config_path)
    with path.open() as f:
        config = json.load(f)

    providers = config.get("providers")
    models = config.get("models")
    if not isinstance(providers, dict) or not isinstance(models, dict):
        raise ValueError(f"LLM config must define object fields 'providers' and 'models': {path}")

    for provider_name, provider in providers.items():
        if not isinstance(provider, dict) or not provider.get("url"):
            raise ValueError(f"Provider {provider_name!r} must define a 'url'")
        headers = provider.get("headers", {})
        if not isinstance(headers, dict):
            raise ValueError(f"Provider {provider_name!r} headers must be an object")
        for header_name, env_names in headers.items():
            if not isinstance(header_name, str) or not header_name:
                raise ValueError(f"Provider {provider_name!r} header names must be non-empty strings")
            _provider_header_env_names(env_names)

    for model_name, model in models.items():
        provider_name = model.get("provider") if isinstance(model, dict) else None
        if provider_name not in providers:
            raise ValueError(f"Model {model_name!r} references unknown provider {provider_name!r}")

    return config


def available_models(config: dict[str, Any] | None = None) -> list[str]:
    """Return configured model names."""
    config = load_llm_config() if config is None else config
    return sorted(config["models"])


def model_generation_defaults(config: dict[str, Any], model: str) -> dict[str, Any]:
    """Return the generation params configured for a model."""
    if model not in config["models"]:
        raise ValueError(f"Model {model!r} is not available. Available models: {available_models(config)}")
    generation = config["models"][model].get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError(f"Model {model!r} generation defaults must be an object")
    if "max_tokens" in generation and "max_completion_tokens" in generation:
        raise ValueError(
            f"Model {model!r} generation defaults must set only one of "
            "max_tokens or max_completion_tokens"
        )
    return dict(generation)


def provider_token(provider: dict[str, Any], environ: Mapping[str, str] | None = None) -> str | None:
    """Resolve a provider token from the env var named by the provider's JSON 'token' field."""
    token_env = provider.get("token")
    if token_env is None:
        return None
    if not isinstance(token_env, str) or not token_env:
        raise ValueError("Provider 'token' field must be an environment variable name")
    environ = os.environ if environ is None else environ
    return environ.get(token_env)


def _provider_header_env_names(value: Any) -> list[str]:
    """Normalize a provider header env-var spec from the JSON config."""
    if isinstance(value, str) and value:
        return [value]
    if isinstance(value, list) and value and all(isinstance(item, str) and item for item in value):
        return value
    raise ValueError("Provider header values must be an env var name or non-empty list of names")


def provider_headers(
    provider: dict[str, Any], environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """Resolve provider-specific HTTP headers from environment variables."""
    headers = provider.get("headers") or {}
    if not isinstance(headers, dict):
        raise ValueError("Provider 'headers' field must be an object")

    environ = os.environ if environ is None else environ
    resolved = {}
    for header_name, env_spec in headers.items():
        env_names = _provider_header_env_names(env_spec)
        for env_name in env_names:
            value = environ.get(env_name)
            if value:
                resolved[header_name] = value
                break
        else:
            raise ValueError(
                f"Missing required header environment variable for {header_name}: "
                f"one of {', '.join(env_names)}"
            )
    return resolved


def messages_for_entry(entry: dict[str, Any], system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Build the chat messages for a TritonBench entry."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": make_prompt(entry)},
    ]


def stable_hash(value: Any) -> str:
    """Hash JSON-compatible data in a stable way for resume/deduplication."""
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def text_hash(value: str) -> str:
    """Hash text exactly as it will be sent to an inference provider."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def manifest_path_for_output(out_path: str | Path) -> Path:
    """Return the sidecar manifest path for an inference JSONL ledger."""
    return Path(f"{out_path}.manifest.json")


def _grammar_manifest_entries(
    generation: dict[str, Any],
    *,
    grammar_source_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    extra_body = generation.get("extra_body")
    if not isinstance(extra_body, dict):
        return []

    location = "generation.extra_body.guided_grammar"
    grammar = extra_body.get("guided_grammar")
    if not isinstance(grammar, str):
        return []

    entry: dict[str, Any] = {
        "location": location,
        "sha256": text_hash(grammar),
        "bytes": len(grammar.encode("utf-8")),
        "content": grammar,
    }
    if grammar_source_path is not None:
        entry["source_path"] = str(Path(grammar_source_path))
    return [entry]


def _generation_manifest_view(
    generation: dict[str, Any], grammar_entries: list[dict[str, Any]]
) -> dict[str, Any]:
    """Return generation params with large grammar strings replaced by manifest refs."""
    view = json.loads(json.dumps(generation, ensure_ascii=False))
    extra_body = view.get("extra_body")
    if isinstance(extra_body, dict):
        if isinstance(extra_body.get("guided_grammar"), str):
            grammar_entry = grammar_entries[0] if grammar_entries else None
            extra_body["guided_grammar"] = {
                "manifest_ref": "grammars[0]",
                "sha256": grammar_entry.get("sha256") if grammar_entry else None,
                "bytes": grammar_entry.get("bytes") if grammar_entry else None,
            }
    return view


def _redacted_generation_for_log(generation: dict[str, Any]) -> dict[str, Any]:
    """Return generation params safe for progress logs without dumping full grammars."""
    view = json.loads(json.dumps(generation, ensure_ascii=False))
    extra_body = view.get("extra_body")
    if isinstance(extra_body, dict):
        if isinstance(extra_body.get("guided_grammar"), str):
            grammar = extra_body["guided_grammar"]
            extra_body["guided_grammar"] = {
                "redacted": True,
                "sha256": text_hash(grammar),
                "bytes": len(grammar.encode("utf-8")),
            }
    return view


def write_run_manifest(
    out_path: str | Path,
    tasks: list[dict[str, Any]],
    *,
    generation: dict[str, Any],
    grammar_source_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    """Write a run-level sidecar manifest for an inference ledger."""
    out_path = Path(out_path)
    path = Path(manifest_path) if manifest_path is not None else manifest_path_for_output(out_path)
    grammar_entries = _grammar_manifest_entries(
        generation, grammar_source_path=grammar_source_path
    )
    manifest = {
        "schema_version": 1,
        "created_at": _now_iso(),
        "ledger_path": str(out_path),
        "task_count": len(tasks),
        "models": sorted({task["model"] for task in tasks}),
        "providers": sorted({task["provider"] for task in tasks}),
        "generation": _generation_manifest_view(generation, grammar_entries),
        "grammars": grammar_entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def build_tasks(
    entries: list[dict[str, Any]],
    model: str,
    generation: dict[str, Any],
    config: dict[str, Any],
    *,
    limit: int | None = None,
    entry_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Build plain request dictionaries for a batch run."""
    if model not in config["models"]:
        raise ValueError(f"Model {model!r} is not available. Available models: {available_models(config)}")

    model_config = config["models"][model]
    provider_name = model_config["provider"]
    provider = config["providers"][provider_name]
    indexed_entries = list(enumerate(entries))
    if entry_indices is not None:
        indexed_entries = [
            (entry_index, entry)
            for entry_index, entry in indexed_entries
            if entry_index in entry_indices
        ]
    if limit is not None:
        indexed_entries = indexed_entries[:limit]
    tasks = []

    for entry_index, entry in indexed_entries:
        messages = messages_for_entry(entry)
        task = {
            "entry_index": entry_index,
            "entry_file": entry["file"],
            "model": model,
            "model_label": model_config.get("label", model),
            "provider": provider_name,
            "provider_url": provider["url"],
            "messages": messages,
            "generation": dict(generation),
        }
        task["request_hash"] = stable_hash(
            {
                "entry_index": task["entry_index"],
                "entry_file": task["entry_file"],
                "model": task["model"],
                "provider": task["provider"],
                "messages": task["messages"],
                "generation": task["generation"],
            }
        )
        tasks.append(task)

    return tasks


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    """Append one JSON row to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_existing_result_hashes(path: str | Path) -> dict[str, set[str]]:
    """Read an existing JSONL ledger and return request hashes by terminal status."""
    path = Path(path)
    successful: set[str] = set()
    failed: set[str] = set()
    if not path.exists():
        return {"success": successful, "failed": failed}

    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc

            request_hash = row.get("request_hash")
            status = row.get("status")
            if not isinstance(request_hash, str):
                continue
            if status == "success":
                successful.add(request_hash)
                failed.discard(request_hash)
            elif status == "failed" and request_hash not in successful:
                failed.add(request_hash)

    return {"success": successful, "failed": failed}


def row_finish_reason(row: dict[str, Any]) -> str | None:
    """Return the first choice finish_reason for a result row, if present."""
    response = row.get("response")
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    finish_reason = choice.get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None


def latest_truncated_entry_indices(path: str | Path, model: str) -> set[int]:
    """Return entry indices whose latest row for a model ended due to length."""
    path = Path(path)
    latest_rows: dict[tuple[int, str], dict[str, Any]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Cannot select truncated rows from missing ledger: {path}")

    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if row.get("model") != model:
                continue
            entry_index = row.get("entry_index")
            entry_file = row.get("entry_file")
            if not isinstance(entry_index, int) or not isinstance(entry_file, str):
                continue
            latest_rows[(entry_index, entry_file)] = row

    return {
        entry_index
        for (entry_index, _entry_file), row in latest_rows.items()
        if row.get("status") == "success" and row_finish_reason(row) == "length"
    }


def filter_tasks_for_resume(
    tasks: list[dict[str, Any]],
    out_path: str | Path,
    *,
    resume: bool = True,
    retry_failed: bool = False,
    force: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Filter tasks against an existing output ledger for resumable batch runs."""
    if force or not resume:
        return tasks, {
            "planned": len(tasks),
            "remaining": len(tasks),
            "skipped_success": 0,
            "skipped_failed": 0,
        }

    existing = read_existing_result_hashes(out_path)
    successful = existing["success"]
    failed = existing["failed"]
    filtered = []
    skipped_success = 0
    skipped_failed = 0

    for task in tasks:
        request_hash = task["request_hash"]
        if request_hash in successful:
            skipped_success += 1
        elif request_hash in failed and not retry_failed:
            skipped_failed += 1
        else:
            filtered.append(task)

    return filtered, {
        "planned": len(tasks),
        "remaining": len(filtered),
        "skipped_success": skipped_success,
        "skipped_failed": skipped_failed,
    }


def compact_successful_replacements(path: str | Path, tasks: list[dict[str, Any]]) -> int:
    """Keep only the latest non-truncated successful row for retried task entries."""
    path = Path(path)
    if not path.exists() or not tasks:
        return 0

    replacement_keys = {
        (task["model"], task["provider"], task["entry_index"], task["entry_file"])
        for task in tasks
    }
    rows = []
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc

    last_index_by_key: dict[tuple[str, str, int, str], int] = {}
    for index, row in enumerate(rows):
        key = (
            row.get("model"),
            row.get("provider"),
            row.get("entry_index"),
            row.get("entry_file"),
        )
        if key in replacement_keys:
            last_index_by_key[key] = index

    replaceable_keys = {
        key
        for key, index in last_index_by_key.items()
        if rows[index].get("status") == "success" and row_finish_reason(rows[index]) != "length"
    }
    if not replaceable_keys:
        return 0

    compacted = []
    removed = 0
    for index, row in enumerate(rows):
        key = (
            row.get("model"),
            row.get("provider"),
            row.get("entry_index"),
            row.get("entry_file"),
        )
        if key in replaceable_keys and index != last_index_by_key[key]:
            removed += 1
            continue
        compacted.append(row)

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        for row in compacted:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp_path.replace(path)
    return removed


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProviderRateLimiter:
    """Thread-safe rolling-window limiter for one provider."""

    def __init__(
        self,
        *,
        requests_per_minute: int | None = None,
        tokens_per_minute: int | None = None,
        token_estimate_chars_per_token: float = DEFAULT_TOKEN_ESTIMATE_CHARS_PER_TOKEN,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self.token_estimate_chars_per_token = token_estimate_chars_per_token
        self._request_times: deque[float] = deque()
        self._token_reservations: deque[tuple[float, int]] = deque()
        self._lock = Lock()

    @property
    def enabled(self) -> bool:
        return self.requests_per_minute is not None or self.tokens_per_minute is not None

    def acquire(self, estimated_tokens: int = 0) -> None:
        """Reserve one request and its estimated tokens before making an API call."""
        if not self.enabled:
            return

        estimated_tokens = max(0, estimated_tokens)
        if self.tokens_per_minute is not None and estimated_tokens > self.tokens_per_minute:
            raise ValueError(
                f"Estimated request size ({estimated_tokens} tokens) exceeds provider "
                f"tokens_per_minute limit ({self.tokens_per_minute})"
            )

        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                delay = self._delay_seconds(now, estimated_tokens)
                if delay <= 0:
                    self._request_times.append(now)
                    if self.tokens_per_minute is not None:
                        self._token_reservations.append((now, estimated_tokens))
                    return

            time.sleep(delay)

    def _prune(self, now: float) -> None:
        cutoff = now - RATE_LIMIT_WINDOW_SECONDS
        while self._request_times and self._request_times[0] <= cutoff:
            self._request_times.popleft()
        while self._token_reservations and self._token_reservations[0][0] <= cutoff:
            self._token_reservations.popleft()

    def _delay_seconds(self, now: float, estimated_tokens: int) -> float:
        delays = []
        if (
            self.requests_per_minute is not None
            and len(self._request_times) >= self.requests_per_minute
        ):
            delays.append(self._request_times[0] + RATE_LIMIT_WINDOW_SECONDS - now)

        if self.tokens_per_minute is not None:
            reserved_tokens = sum(tokens for _, tokens in self._token_reservations)
            overflow = reserved_tokens + estimated_tokens - self.tokens_per_minute
            if overflow > 0:
                freed_tokens = 0
                for timestamp, tokens in self._token_reservations:
                    freed_tokens += tokens
                    if freed_tokens >= overflow:
                        delays.append(timestamp + RATE_LIMIT_WINDOW_SECONDS - now)
                        break

        return max(delays, default=0.0)


def _positive_int_or_none(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Provider rate_limits.{field_name} must be a positive integer")
    return value


def make_provider_rate_limiter(provider: dict[str, Any]) -> ProviderRateLimiter | None:
    """Create a limiter from provider metadata, or None for unthrottled providers."""
    limits = provider.get("rate_limits")
    if limits is None:
        return None
    if not isinstance(limits, dict):
        raise ValueError("Provider rate_limits must be an object")

    chars_per_token = limits.get(
        "token_estimate_chars_per_token", DEFAULT_TOKEN_ESTIMATE_CHARS_PER_TOKEN
    )
    if not isinstance(chars_per_token, int | float) or chars_per_token <= 0:
        raise ValueError("Provider rate_limits.token_estimate_chars_per_token must be positive")

    limiter = ProviderRateLimiter(
        requests_per_minute=_positive_int_or_none(
            limits.get("requests_per_minute"), "requests_per_minute"
        ),
        tokens_per_minute=_positive_int_or_none(
            limits.get("tokens_per_minute"), "tokens_per_minute"
        ),
        token_estimate_chars_per_token=float(chars_per_token),
    )
    return limiter if limiter.enabled else None


def estimate_task_tokens(task: dict[str, Any], chars_per_token: float) -> int:
    """Conservatively estimate request tokens for provider TPM throttling."""
    prompt_chars = sum(len(message.get("content", "")) for message in task["messages"])
    prompt_tokens = math.ceil(prompt_chars / chars_per_token)

    generation = task.get("generation", {})
    output_tokens = generation.get("max_completion_tokens", generation.get("max_tokens"))
    if output_tokens is None:
        raise ValueError(
            "Token-per-minute rate limiting requires generation max_tokens or "
            "max_completion_tokens"
        )
    if not isinstance(output_tokens, int) or output_tokens < 0:
        raise ValueError("Generation token budget must be a non-negative integer")

    return prompt_tokens + output_tokens


def llm_request(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    token: str | None = None,
    *,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    top_p: float | None = None,
    timeout: int = 300,
    extra_body: dict[str, Any] | None = None,
    default_headers: dict[str, str] | None = None,
) -> Any:
    """Make one OpenAI-compatible chat completion request."""
    if max_tokens is not None and max_completion_tokens is not None:
        raise ValueError("Set only one of max_tokens or max_completion_tokens")

    client = OpenAI(
        base_url=base_url.rstrip("/"),
        api_key=token or "EMPTY",
        timeout=timeout,
        default_headers=default_headers,
    )

    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if temperature is not None:
        kwargs["temperature"] = temperature
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if top_p is not None:
        kwargs["top_p"] = top_p
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    result = client.chat.completions.create(**kwargs)
    if len(result.choices) != 1:
        raise ValueError(
            f"Expected exactly one choice in LLM response, got: {len(result.choices)}"
        )
    return result


def response_to_dict(response: Any) -> dict[str, Any]:
    """Convert an OpenAI SDK response object to a plain dict when possible."""
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    raise TypeError(f"Cannot convert response of type {type(response).__name__} to dict")


def response_content(response: Any) -> str:
    """Extract the first chat completion message content from an SDK object or dict."""
    if isinstance(response, dict):
        return response["choices"][0]["message"]["content"]
    return response.choices[0].message.content


def make_provider_client(provider: dict[str, Any], *, timeout: int = 300) -> OpenAI:
    """Create an OpenAI-compatible client for a configured provider."""
    token = provider_token(provider)
    if provider.get("token") and token is None:
        raise ValueError(f"Missing required token environment variable: {provider['token']}")
    headers = provider_headers(provider)
    return OpenAI(
        base_url=provider["url"].rstrip("/"),
        api_key=token or "EMPTY",
        timeout=timeout,
        default_headers=headers or None,
    )


def _retry_after_seconds(exc: APIStatusError) -> float | None:
    retry_after = exc.response.headers.get("retry-after")
    if retry_after is None:
        return None

    try:
        return max(0.0, float(retry_after))
    except ValueError:
        try:
            retry_at = email.utils.parsedate_to_datetime(retry_after)
        except (TypeError, ValueError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def _error_text(exc: APIStatusError) -> str:
    parts = [str(exc)]
    response_text = getattr(exc.response, "text", None)
    if response_text:
        parts.append(response_text)
    return "\n".join(parts).replace("_", "").replace(" ", "").lower()


def _is_deterministic_request_error(exc: APIStatusError) -> bool:
    if exc.status_code in {400, 401, 403, 404, 422}:
        return True
    error_text = _error_text(exc)
    return any(marker in error_text for marker in NON_RETRYABLE_ERROR_MARKERS)


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(exc, APIStatusError):
        if _is_deterministic_request_error(exc):
            return False
        return exc.status_code in TRANSIENT_STATUS_CODES
    return False


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    if isinstance(exc, APIStatusError):
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return retry_after
    return (2 ** (attempt - 1)) + random.uniform(0, 1.5)


def call_task(
    task: dict[str, Any],
    client: OpenAI,
    *,
    max_attempts: int = 3,
    rate_limiter: ProviderRateLimiter | None = None,
) -> dict[str, Any]:
    """Execute one task and return a JSON-serializable result row."""
    started_at = _now_iso()
    start = time.monotonic()
    max_attempts = max(1, max_attempts)
    estimated_tokens = (
        estimate_task_tokens(task, rate_limiter.token_estimate_chars_per_token)
        if rate_limiter is not None and rate_limiter.tokens_per_minute is not None
        else 0
    )

    for attempt in range(1, max_attempts + 1):
        try:
            if rate_limiter is not None:
                rate_limiter.acquire(estimated_tokens)
            response = client.chat.completions.create(
                model=task["model"],
                messages=task["messages"],
                **task["generation"],
            )
            if len(response.choices) != 1:
                raise ValueError(
                    f"Expected exactly one choice in LLM response, got: {len(response.choices)}"
                )

            return {
                "schema_version": 1,
                "request_hash": task["request_hash"],
                "status": "success",
                "entry_index": task["entry_index"],
                "entry_file": task["entry_file"],
                "model": task["model"],
                "model_label": task["model_label"],
                "provider": task["provider"],
                "provider_url": task["provider_url"],
                "generation": task["generation"],
                "messages": task["messages"],
                "content": response_content(response),
                "semantic_warnings": check_kernel(response_content(response)),
                "response": response_to_dict(response),
                "attempt": attempt,
                "max_attempts": max_attempts,
                "started_at": started_at,
                "finished_at": _now_iso(),
                "latency_s": time.monotonic() - start,
            }
        except Exception as exc:
            retryable = _is_retryable_error(exc)
            if retryable and attempt < max_attempts:
                delay = _retry_delay_seconds(exc, attempt)
                print(
                    f"retrying {task['entry_file']} {task['model']} after "
                    f"{type(exc).__name__} attempt={attempt}/{max_attempts} "
                    f"delay={delay:.1f}s",
                    flush=True,
                )
                time.sleep(delay)
                continue

            return {
                "schema_version": 1,
                "request_hash": task["request_hash"],
                "status": "failed",
                "entry_index": task["entry_index"],
                "entry_file": task["entry_file"],
                "model": task["model"],
                "model_label": task["model_label"],
                "provider": task["provider"],
                "provider_url": task["provider_url"],
                "generation": task["generation"],
                "messages": task["messages"],
                "semantic_warnings": [],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "retryable": retryable,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "started_at": started_at,
                "finished_at": _now_iso(),
                "latency_s": time.monotonic() - start,
            }

    raise RuntimeError("unreachable: call_task loop exhausted without returning")


def run_batch(
    tasks: list[dict[str, Any]],
    config: dict[str, Any],
    out_path: str | Path,
    *,
    max_workers: int = 2,
    stagger_seconds: float = 2.0,
    timeout: int = 300,
    max_attempts: int = 3,
) -> None:
    """Run tasks with bounded, staggered thread-pool concurrency and append JSONL rows."""
    provider_names = sorted({task["provider"] for task in tasks})
    clients = {
        name: make_provider_client(config["providers"][name], timeout=timeout)
        for name in provider_names
    }
    rate_limiters = {
        name: make_provider_rate_limiter(config["providers"][name]) for name in provider_names
    }
    completed = 0
    succeeded = 0
    failed = 0

    def record_done(done_futures: set[Any]) -> None:
        nonlocal completed, succeeded, failed
        for future in done_futures:
            row = future.result()
            append_jsonl(out_path, row)
            completed += 1
            if row["status"] == "success":
                succeeded += 1
            else:
                failed += 1
            print(
                f"[{completed}/{len(tasks)}] {row['status']} "
                f"{row['entry_file']} {row['model']} "
                f"latency={row['latency_s']:.1f}s success={succeeded} failed={failed}",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending = set()
        last_submit = 0.0
        for task_index, task in enumerate(tasks):
            if task_index > 0 and stagger_seconds > 0:
                elapsed = time.monotonic() - last_submit
                if elapsed < stagger_seconds:
                    time.sleep(stagger_seconds - elapsed)
            print(
                f"submitting [{task_index + 1}/{len(tasks)}] {task['entry_file']} "
                f"{task['model']} generation={_redacted_generation_for_log(task['generation'])}",
                flush=True,
            )
            pending.add(
                pool.submit(
                    call_task,
                    task,
                    clients[task["provider"]],
                    max_attempts=max_attempts,
                    rate_limiter=rate_limiters[task["provider"]],
                )
            )
            last_submit = time.monotonic()
            if len(pending) >= max_workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                record_done(done)

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            record_done(done)


def call_llm(
    entry: dict[str, Any],
    model: str,
    *,
    config: dict[str, Any] | None = None,
    config_path: str | Path = DEFAULT_LLM_CONFIG_PATH,
    max_tokens: int | None = None,
    max_completion_tokens: int | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    top_p: float | None = None,
    timeout: int = 300,
    extra_body: dict[str, Any] | None = None,
) -> Any:
    """Call a configured model for a TritonBench entry."""
    config = load_llm_config(config_path) if config is None else config
    if model not in config["models"]:
        raise ValueError(f"Model {model!r} is not available. Available models: {available_models(config)}")

    model_config = config["models"][model]
    provider = config["providers"][model_config["provider"]]
    token = provider_token(provider)
    headers = provider_headers(provider)
    generation = model_generation_defaults(config, model)

    if max_tokens is not None:
        generation.pop("max_completion_tokens", None)
        generation["max_tokens"] = max_tokens
    if max_completion_tokens is not None:
        generation.pop("max_tokens", None)
        generation["max_completion_tokens"] = max_completion_tokens
    if temperature is not None:
        generation["temperature"] = temperature
    if reasoning_effort is not None:
        generation["reasoning_effort"] = reasoning_effort
    if top_p is not None:
        generation["top_p"] = top_p
    if extra_body is not None:
        generation["extra_body"] = extra_body

    return llm_request(
        provider["url"],
        model,
        messages_for_entry(entry),
        token=token,
        max_tokens=generation.get("max_tokens"),
        max_completion_tokens=generation.get("max_completion_tokens"),
        temperature=generation.get("temperature"),
        reasoning_effort=generation.get("reasoning_effort"),
        top_p=generation.get("top_p"),
        timeout=timeout,
        extra_body=generation.get("extra_body"),
        default_headers=headers or None,
    )


def _parse_extra_body(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--extra-body must be a JSON object")
    return parsed


def _print_dry_run(tasks: list[dict[str, Any]], config: dict[str, Any], preview: int) -> None:
    print(f"planned_requests: {len(tasks)}")
    if not tasks:
        return

    provider_names = sorted({task["provider"] for task in tasks})
    for name in provider_names:
        provider = config["providers"][name]
        token_env = provider.get("token")
        header_envs = {
            header_name: _provider_header_env_names(env_names)
            for header_name, env_names in (provider.get("headers") or {}).items()
        }
        print(
            json.dumps(
                {
                    "provider": name,
                    "url": provider["url"],
                    "token_env": token_env,
                    "token_present": provider_token(provider) is not None if token_env else None,
                    "header_envs": header_envs or None,
                    "headers_present": {
                        header_name: any(os.environ.get(env_name) for env_name in env_names)
                        for header_name, env_names in header_envs.items()
                    } or None,
                    "rate_limits": provider.get("rate_limits"),
                },
                indent=2,
            )
        )

    for task in tasks[:preview]:
        rate_limiter = make_provider_rate_limiter(config["providers"][task["provider"]])
        estimated_tokens = (
            estimate_task_tokens(task, rate_limiter.token_estimate_chars_per_token)
            if rate_limiter is not None and rate_limiter.tokens_per_minute is not None
            else None
        )
        print(
            json.dumps(
                {
                    "request_hash": task["request_hash"],
                    "entry_index": task["entry_index"],
                    "entry_file": task["entry_file"],
                    "model": task["model"],
                    "provider": task["provider"],
                    "generation": _redacted_generation_for_log(task["generation"]),
                    "estimated_rate_limit_tokens": estimated_tokens,
                    "message_chars": [len(message["content"]) for message in task["messages"]],
                    "user_prompt_preview": task["messages"][1]["content"][:500],
                },
                indent=2,
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run TritonBench LLM inference batches.")
    parser.add_argument("--config", type=Path, default=DEFAULT_LLM_CONFIG_PATH)
    parser.add_argument("--tritonbench-root", type=Path, default=DEFAULT_TRITONBENCH_ROOT)
    parser.add_argument("--model", default="lightning-ai/gemma-4-31B-it")
    parser.add_argument("--out", type=Path, default=Path("runs/tritonbench/llm-inference.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-preview", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--stagger-seconds", type=float, default=2.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--only-truncated-from",
        type=Path,
        help="Run only entries whose latest row in this ledger has finish_reason=length.",
    )
    parser.add_argument(
        "--replace-successful-retries",
        action="store_true",
        help=(
            "After the run, remove older rows for retried entries that now have a "
            "successful non-truncated replacement."
        ),
    )
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-completion-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument("--extra-body", type=_parse_extra_body, default={})
    parser.add_argument(
        "--grammar-file",
        type=Path,
        help=(
            "Read a GBNF grammar into extra_body.guided_grammar for vLLM/XGrammar "
            "and snapshot the exact content in the run manifest."
        ),
    )
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args(argv)

    if args.max_tokens is not None and args.max_completion_tokens is not None:
        parser.error("Set only one of --max-tokens or --max-completion-tokens")

    config = load_llm_config(args.config)
    if args.list_models:
        print("\n".join(available_models(config)))
        return 0

    entries, errors, *_ = load_t_simple_entries(args.tritonbench_root)
    if errors:
        raise RuntimeError(f"Errors assembling T dataset: {errors}")

    generation = model_generation_defaults(config, args.model)
    if args.max_tokens is not None:
        generation.pop("max_completion_tokens", None)
        generation["max_tokens"] = args.max_tokens
    if args.max_completion_tokens is not None:
        generation.pop("max_tokens", None)
        generation["max_completion_tokens"] = args.max_completion_tokens
    if args.temperature is not None:
        generation["temperature"] = args.temperature
    if args.top_p is not None:
        generation["top_p"] = args.top_p
    if args.reasoning_effort is not None:
        generation["reasoning_effort"] = args.reasoning_effort
    if args.extra_body:
        generation["extra_body"] = args.extra_body
    if args.grammar_file is not None:
        extra_body = generation.setdefault("extra_body", {})
        if not isinstance(extra_body, dict):
            raise ValueError("generation extra_body must be an object to use --grammar-file")
        if "guided_grammar" in extra_body:
            parser.error("Set only one of --extra-body guided_grammar or --grammar-file")
        extra_body["guided_grammar"] = args.grammar_file.read_text(encoding="utf-8")
        extra_body.setdefault("guided_decoding_backend", "xgrammar")

    entry_indices = None
    if args.only_truncated_from is not None:
        entry_indices = latest_truncated_entry_indices(args.only_truncated_from, args.model)
        print(
            f"only-truncated-from: selected {len(entry_indices)} entries "
            f"from {args.only_truncated_from}",
            flush=True,
        )

    tasks = build_tasks(
        entries,
        args.model,
        generation,
        config,
        limit=args.limit,
        entry_indices=entry_indices,
    )
    tasks, resume_stats = filter_tasks_for_resume(
        tasks,
        args.out,
        resume=args.resume,
        retry_failed=args.retry_failed,
        force=args.force,
    )

    if resume_stats["skipped_success"] or resume_stats["skipped_failed"]:
        print(
            f"resume: planned={resume_stats['planned']} remaining={resume_stats['remaining']} "
            f"skipped_success={resume_stats['skipped_success']} "
            f"skipped_failed={resume_stats['skipped_failed']} retry_failed={args.retry_failed}",
            flush=True,
        )

    if args.dry_run:
        _print_dry_run(tasks, config, args.dry_run_preview)
        return 0

    if not tasks:
        print("no requests to run", flush=True)
        return 0

    manifest_path = write_run_manifest(
        args.out,
        tasks,
        generation=generation,
        grammar_source_path=args.grammar_file,
    )

    print(
        f"running {len(tasks)} requests model={args.model} "
        f"max_workers={args.max_workers} stagger_seconds={args.stagger_seconds} "
        f"max_attempts={args.max_attempts} out={args.out} manifest={manifest_path}",
        flush=True,
    )
    run_batch(
        tasks,
        config,
        args.out,
        max_workers=args.max_workers,
        stagger_seconds=args.stagger_seconds,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
    )
    if args.replace_successful_retries:
        removed = compact_successful_replacements(args.out, tasks)
        print(f"replace-successful-retries: removed {removed} older rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
