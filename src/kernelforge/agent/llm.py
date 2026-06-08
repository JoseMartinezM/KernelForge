from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, cast

from .schemas import LLMCallResult
from kernelforge.benchmark.llm_inference import (
    DEFAULT_LLM_CONFIG_PATH,
    load_llm_config,
    make_provider_client,
    model_generation_defaults,
    response_content,
    response_to_dict,
)


def strip_reasoning_artifacts(content: str) -> str:
    """Remove common hidden-thought tags that reasoning models may return."""
    text = content or ""
    text = re.sub(r"<thought>.*?</thought>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def merge_generation(
    base: dict[str, Any],
    overrides: dict[str, Any] | None = None,
    *,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge generation defaults, explicit overrides, and optional extra_body."""
    generation = dict(base)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                generation[key] = value
    if extra_body:
        merged_extra = dict(generation.get("extra_body") or {})
        merged_extra.update(extra_body)
        generation["extra_body"] = merged_extra
    if "max_tokens" in generation and "max_completion_tokens" in generation:
        raise ValueError("Set only one of max_tokens or max_completion_tokens")
    return generation


def api_model_name(config: dict[str, Any], model: str) -> str:
    """Return provider-facing model id, honoring optional llm_models.json aliases."""
    model_config = config["models"][model]
    model_id = model_config.get("model_id")
    return model_id if isinstance(model_id, str) and model_id else model


def call_model(
    model: str,
    messages: list[dict[str, str]],
    *,
    config_path: str | Path = DEFAULT_LLM_CONFIG_PATH,
    generation_overrides: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    timeout: int = 300,
) -> tuple[LLMCallResult, dict[str, Any]]:
    """Call one configured OpenAI-compatible chat-completions model."""
    config = load_llm_config(config_path)
    if model not in config["models"]:
        raise ValueError(f"Unknown model {model!r}; available models: {sorted(config['models'])}")
    model_config = config["models"][model]
    provider = config["providers"][model_config["provider"]]
    client = make_provider_client(provider, timeout=timeout)
    generation = merge_generation(
        model_generation_defaults(config, model),
        generation_overrides,
        extra_body=extra_body,
    )

    request_kwargs: dict[str, Any] = {
        "model": api_model_name(config, model),
        "messages": messages,
        **generation,
    }
    start = time.monotonic()
    response = client.chat.completions.create(**cast(Any, request_kwargs))
    latency_s = time.monotonic() - start
    response_dict = response_to_dict(response)
    usage_value = response_dict.get("usage")
    usage = cast(dict[str, Any], usage_value) if isinstance(usage_value, dict) else {}
    choices = response_dict.get("choices")
    finish_reason = None
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
            finish_reason = choice["finish_reason"]
    return (
        LLMCallResult(
            model=model,
            content=strip_reasoning_artifacts(response_content(response)),
            response=response_dict,
            usage=usage,
            finish_reason=finish_reason,
            latency_s=latency_s,
        ),
        generation,
    )
