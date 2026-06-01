from __future__ import annotations


def test_grammar_loads(triton_llguidance):
    """Smoke-test that LLGuidance can parse and validate the grammar file."""

    assert triton_llguidance.validation_error == ""


def test_no_warnings(triton_llguidance):
    assert triton_llguidance.validation_messages == []
