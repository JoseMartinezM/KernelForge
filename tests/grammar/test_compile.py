from __future__ import annotations


def test_grammar_loads(triton_xgrammar_jit_block):
    """Smoke-test that XGrammar can parse and compile the JIT-block grammar."""

    assert triton_xgrammar_jit_block.compiled_grammar is not None


def test_root_rejects_invalid_indented_jit_statement(triton_xgrammar_root):
    result = triton_xgrammar_root.match(
        "@triton.jit\n"
        "def kernel(x):\n"
        "    y = x\n"
        "    import os\n"
    )

    assert not result.accepted


def test_root_rejects_invalid_indented_jit_statement_after_blank_line(triton_xgrammar_root):
    result = triton_xgrammar_root.match(
        "@triton.jit\n"
        "def kernel(x):\n"
        "    y = x\n"
        "\n"
        "    import os\n"
    )

    assert not result.accepted


def test_root_accepts_dedented_python_after_jit_block(triton_xgrammar_root):
    result = triton_xgrammar_root.match(
        "@triton.jit\n"
        "def kernel(x):\n"
        "    y = x\n"
        "\n"
        "def wrapper():\n"
        "    import os\n"
    )

    assert result.accepted


def test_root_accepts_adjacent_jit_blocks(triton_xgrammar_root):
    result = triton_xgrammar_root.match(
        "@triton.jit\n"
        "def first(x):\n"
        "    y = x\n"
        "@triton.jit()\n"
        "def second(x):\n"
        "    return x\n"
    )

    assert result.accepted
