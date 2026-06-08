from __future__ import annotations

from pathlib import Path
from typing import Any

from .schemas import text_hash

DEFAULT_TRITON_GRAMMAR = Path("grammar/triton.gbnf")
GrammarBackend = str


def read_grammar(path: str | Path = DEFAULT_TRITON_GRAMMAR) -> str:
    """Read the active GBNF grammar for vLLM/XGrammar structured outputs."""
    return Path(path).read_text(encoding="utf-8")


def grammar_metadata(path: str | Path | None) -> dict[str, Any] | None:
    """Return a manifest-friendly grammar hash block."""
    if path is None:
        return None
    grammar_path = Path(path)
    content = read_grammar(grammar_path)
    return {
        "source_path": str(grammar_path),
        "sha256": text_hash(content),
        "bytes": len(content.encode("utf-8")),
    }


def xgrammar_extra_body(path: str | Path) -> dict[str, Any]:
    """Build OpenAI-compatible vLLM extra_body for xgrammar-constrained decoding."""
    return {
        "guided_grammar": read_grammar(path),
        "guided_decoding_backend": "xgrammar",
    }


def llama_cpp_extra_body(path: str | Path) -> dict[str, Any]:
    """Build OpenAI-compatible llama.cpp request extras for native GBNF decoding."""
    return {
        "grammar": read_grammar(path),
        "chat_template_kwargs": {"enable_thinking": False},
    }


def grammar_extra_body(path: str | Path, backend: GrammarBackend = "xgrammar") -> dict[str, Any]:
    """Build provider-specific grammar request extras."""
    if backend == "xgrammar":
        return xgrammar_extra_body(path)
    if backend == "llama-cpp":
        return llama_cpp_extra_body(path)
    raise ValueError(f"unsupported grammar backend: {backend}")


def redacted_grammar_extra_body(
    path: str | Path | None,
    backend: GrammarBackend = "xgrammar",
) -> dict[str, Any] | None:
    """Return generation metadata with the full grammar replaced by its hash."""
    metadata = grammar_metadata(path)
    if metadata is None:
        return None
    if backend == "llama-cpp":
        return {
            "grammar": {
                "sha256": metadata["sha256"],
                "bytes": metadata["bytes"],
                "source_path": metadata["source_path"],
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
    return {
        "guided_decoding_backend": "xgrammar",
        "guided_grammar": {
            "sha256": metadata["sha256"],
            "bytes": metadata["bytes"],
            "source_path": metadata["source_path"],
        },
    }


def redacted_xgrammar_extra_body(path: str | Path | None) -> dict[str, Any] | None:
    """Backward-compatible alias for xgrammar redaction."""
    return redacted_grammar_extra_body(path, backend="xgrammar")
