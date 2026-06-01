from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from llguidance import LLMatcher, LLTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRITON_LARK_PATH = PROJECT_ROOT / "grammar" / "triton.lark"


@dataclass(frozen=True)
class GrammarMatchResult:
    accepted: bool
    consumed_all: bool
    accepting: bool
    error: str


class LLGuidanceHarness:
    def __init__(self, grammar_path: Path) -> None:
        self.grammar_path = grammar_path
        self.grammar_text = grammar_path.read_text(encoding="utf-8")
        self.tokenizer = LLTokenizer("byte")
        self.grammar = LLMatcher.grammar_from_lark(self.grammar_text)
        self.validation_error = LLMatcher.validate_grammar(self.grammar, self.tokenizer)
        _, self.validation_messages = LLMatcher.validate_grammar_with_warnings(
            self.grammar,
            self.tokenizer,
        )

    def match(self, text: str) -> GrammarMatchResult:
        matcher = LLMatcher(self.tokenizer, self.grammar, log_level=0)
        if matcher.is_error():
            return GrammarMatchResult(
                accepted=False,
                consumed_all=False,
                accepting=False,
                error=matcher.get_error(),
            )

        consumed_all = matcher.consume_tokens(self.tokenizer.tokenize_str(text))
        error = matcher.get_error() if matcher.is_error() else ""
        accepting = matcher.is_accepting()
        return GrammarMatchResult(
            accepted=consumed_all and accepting and not error,
            consumed_all=consumed_all,
            accepting=accepting,
            error=error,
        )


@pytest.fixture(scope="session")
def triton_llguidance() -> LLGuidanceHarness:
    harness = LLGuidanceHarness(TRITON_LARK_PATH)
    if harness.validation_error:
        pytest.fail(f"{TRITON_LARK_PATH} is not valid LLGuidance Lark:\n{harness.validation_error}")
    return harness
