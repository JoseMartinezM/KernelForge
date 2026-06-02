from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import xgrammar as xgr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRITON_GBNF_PATH = PROJECT_ROOT / "grammar" / "triton.gbnf"


@dataclass(frozen=True)
class GrammarMatchResult:
    accepted: bool
    consumed_all: bool
    accepting: bool
    error: str


class XGrammarHarness:
    def __init__(self, grammar_path: Path, *, root_rule_name: str = "root") -> None:
        self.grammar_path = grammar_path
        self.grammar_text = grammar_path.read_text(encoding="utf-8")
        self.root_rule_name = root_rule_name
        # A byte-sized raw vocabulary keeps corpus tests tokenizer-independent while
        # still exercising xgrammar's GBNF parser and string matcher directly.
        encoded_vocab = [bytes([value]) for value in range(256)] + [b"<eos>"]
        self.tokenizer_info = xgr.TokenizerInfo(
            encoded_vocab,
            xgr.VocabType.RAW,
            stop_token_ids=[256],
        )
        self.compiler = xgr.GrammarCompiler(self.tokenizer_info)
        self.compiled_grammar = self.compiler.compile_grammar(
            self.grammar_text,
            root_rule_name=root_rule_name,
        )

    def match(self, text: str) -> GrammarMatchResult:
        matcher = xgr.GrammarMatcher(
            self.compiled_grammar,
            terminate_without_stop_token=True,
        )
        accepted = matcher.accept_string(text)
        completed = matcher.is_completed()
        terminated = matcher.is_terminated()
        return GrammarMatchResult(
            accepted=accepted and completed and terminated,
            consumed_all=accepted,
            accepting=completed,
            error="" if accepted else f"xgrammar rejected input for {self.root_rule_name!r}",
        )


@pytest.fixture(scope="session")
def triton_xgrammar_jit_block() -> XGrammarHarness:
    return XGrammarHarness(TRITON_GBNF_PATH, root_rule_name="jit-block")


@pytest.fixture(scope="session")
def triton_xgrammar_root() -> XGrammarHarness:
    return XGrammarHarness(TRITON_GBNF_PATH, root_rule_name="root")
