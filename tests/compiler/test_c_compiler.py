"""
Tests for the C compiler binary (triton_compiler.exe / triton_compiler).

Run the Bison+Flex binary against .py fixtures and verify exit codes,
stdout, and stderr.

Prerequisite: build from compiler/ first:
    cd compiler && make compiler
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

COMPILER_DIR     = Path(__file__).parents[2] / "compiler"
FIXTURES_DIR     = Path(__file__).parent / "fixtures"
FIXTURES_VALID   = FIXTURES_DIR / "valid"
FIXTURES_INVALID = FIXTURES_DIR / "invalid"

# Binary name (Windows uses .exe, Linux/Mac do not)
_BINARY_STEM = "triton_compiler"
_BINARY = COMPILER_DIR / (_BINARY_STEM + (".exe" if sys.platform == "win32" else ""))

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _run(fixture: Path, timeout: int = 10) -> subprocess.CompletedProcess:
    """Run triton_compiler on a fixture and return the result."""
    return subprocess.run(
        [str(_BINARY), str(fixture)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _skip_if_no_binary():
    if not _BINARY.exists():
        pytest.skip(
            f"Binary '{_BINARY.name}' not found in {COMPILER_DIR}. "
            "Build it with: cd compiler && make compiler"
        )


# ---------------------------------------------------------------------------
# Valid fixtures: exit 0 and no "error" in stderr
# ---------------------------------------------------------------------------

class TestValidFixtures:

    @pytest.fixture(autouse=True)
    def require_binary(self):
        _skip_if_no_binary()

    @pytest.mark.parametrize("fixture", sorted(FIXTURES_VALID.glob("*.py")))
    def test_exits_zero(self, fixture: Path):
        result = _run(fixture)
        assert result.returncode == 0, (
            f"{fixture.name}: exit {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    @pytest.mark.parametrize("fixture", sorted(FIXTURES_VALID.glob("*.py")))
    def test_no_parse_error_in_output(self, fixture: Path):
        result = _run(fixture)
        combined = result.stdout.lower() + result.stderr.lower()
        assert "syntax error" not in combined, (
            f"{fixture.name} reported 'syntax error':\n{result.stderr}"
        )

    def test_vector_add_symbol_table(self):
        result = _run(FIXTURES_VALID / "vector_add.py")
        assert result.returncode == 0
        out = result.stdout
        assert "add_kernel" in out, "Kernel name was not found in the output"
        assert "triton.jit" in out.lower() or "jit" in out.lower(), (
            "Did not report @triton.jit"
        )

    def test_softmax_detects_tl_calls(self):
        result = _run(FIXTURES_VALID / "softmax_kernel.py")
        assert result.returncode == 0
        out = result.stdout
        assert any(api in out for api in ["tl.", "program_id", "load", "store"]), (
            "softmax_kernel did not report any Triton API"
        )

    def test_matmul_detects_dot(self):
        result = _run(FIXTURES_VALID / "matmul_kernel.py")
        assert result.returncode == 0
        # tl.dot must appear in the Triton call table.
        assert "dot" in result.stdout, (
            "matmul_kernel did not detect tl.dot in the symbol table"
        )

    def test_dropout_no_jit_warning(self):
        """dropout_kernel has @triton.jit, so no missing-JIT warning is expected."""
        result = _run(FIXTURES_VALID / "dropout_kernel.py")
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "without @triton.jit" not in combined.lower(), (
            "dropout_kernel reported a missing-JIT warning even though it has @triton.jit"
        )

    def test_flash_attention_constexpr_params(self):
        result = _run(FIXTURES_VALID / "flash_attention.py")
        assert result.returncode == 0
        out = result.stdout
        # Must report constexpr parameters.
        assert "constexpr" in out.lower() or "BLOCK_M" in out, (
            "flash_attention did not report tl.constexpr parameters"
        )

    def test_layer_norm_two_kernels(self):
        result = _run(FIXTURES_VALID / "layer_norm.py")
        assert result.returncode == 0
        out = result.stdout
        assert "layer_norm_kernel" in out, "layer_norm_kernel was not found"
        assert "layer_norm_bwd_dx_fused" in out, "layer_norm_bwd_dx_fused was not found"


# ---------------------------------------------------------------------------
# Invalid fixtures must produce errors or warnings
# ---------------------------------------------------------------------------

class TestInvalidFixtures:

    @pytest.fixture(autouse=True)
    def require_binary(self):
        _skip_if_no_binary()

    def test_bad_syntax_nonzero_exit(self):
        result = _run(FIXTURES_INVALID / "bad_syntax.py")
        assert result.returncode != 0, (
            "bad_syntax.py should fail (exit != 0), but it exited with 0"
        )

    def test_bad_syntax_reports_error(self):
        result = _run(FIXTURES_INVALID / "bad_syntax.py")
        combined = result.stdout.lower() + result.stderr.lower()
        assert "error" in combined or "syntax" in combined, (
            "bad_syntax.py did not report any error"
        )

    def test_duplicate_kernel_detected(self):
        result = _run(FIXTURES_INVALID / "duplicate_kernel.py")
        combined = result.stdout + result.stderr
        assert "add_kernel" in combined, (
            "duplicate_kernel.py did not mention the duplicate name 'add_kernel'"
        )
        has_error = (
            result.returncode != 0
            or "error" in combined.lower()
            or "duplicate" in combined.lower()
            or "already declared" in combined.lower()
        )
        assert has_error, (
            "duplicate_kernel.py did not report the kernel redefinition"
        )

    def test_missing_jit_warns(self):
        result = _run(FIXTURES_INVALID / "missing_jit.py")
        combined = result.stdout + result.stderr
        has_warning = (
            "warning" in combined.lower()
            or "without @triton.jit" in combined.lower()
            or "triton.jit" in combined.lower()
        )
        assert has_warning, (
            "missing_jit.py did not generate a warning about missing @triton.jit"
        )


# ---------------------------------------------------------------------------
# Scanner correctness tests (concrete tokens)
# ---------------------------------------------------------------------------

SCANNER_BINARY = COMPILER_DIR / ("triton_scanner" + (".exe" if sys.platform == "win32" else ""))


def _run_scanner(fixture: Path, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(SCANNER_BINARY), str(fixture)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _run_scanner_source(src: str, timeout: int = 10) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(SCANNER_BINARY)],
        input=src,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _token_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("< ")]


def _token_names(output: str) -> list[str]:
    names: list[str] = []
    for line in _token_lines(output):
        parts = line.split(",", maxsplit=2)
        names.append(parts[0].removeprefix("<").strip())
    return names


class TestScanner:

    @pytest.fixture(autouse=True)
    def require_scanner(self):
        if not SCANNER_BINARY.exists():
            pytest.skip(
                f"Scanner '{SCANNER_BINARY.name}' not found. "
                "Build it with: cd compiler && make scanner"
            )

    def test_scanner_vector_add_tokens(self):
        result = _run_scanner(FIXTURES_VALID / "vector_add.py")
        assert result.returncode == 0
        out = result.stdout
        # Tokens that should always appear in a well-formed kernel.
        assert "DEF"     in out, "Token DEF was not found"
        assert "INDENT"  in out, "Token INDENT was not found"
        assert "DEDENT"  in out, "Token DEDENT was not found"
        assert "NEWLINE" in out, "Token NEWLINE was not found"

    def test_scanner_emits_jit_tokens(self):
        result = _run_scanner(FIXTURES_VALID / "vector_add.py")
        out = result.stdout
        assert "triton" in out or "NAME" in out, (
            "No identifiers were found in the token stream"
        )

    def test_scanner_file_and_stdin_token_streams_match(self):
        fixture = FIXTURES_VALID / "vector_add.py"
        file_result = _run_scanner(fixture)
        stdin_result = _run_scanner_source(fixture.read_text(encoding="utf-8"))

        assert file_result.returncode == 0
        assert stdin_result.returncode == 0
        assert _token_lines(stdin_result.stdout) == _token_lines(file_result.stdout)

    def test_scanner_skips_host_code_before_jit_decorator(self):
        src = (
            "import triton\n"
            "HOST_ONLY = 123\n"
            "def helper():\n"
            "    return HOST_ONLY\n\n"
            "@triton.jit\n"
            "def k(x_ptr):\n"
            "    tl.store(x_ptr, 0.0)\n"
        )
        result = _run_scanner_source(src)

        assert result.returncode == 0
        tokens = _token_lines(result.stdout)
        assert tokens[0].startswith("< AT")
        assert "HOST_ONLY" not in "\n".join(tokens)

    def test_scanner_rejects_indented_nested_jit_decorator(self):
        src = (
            "def outer():\n"
            "    @triton.jit\n"
            "    def nested(x_ptr):\n"
            "        tl.store(x_ptr, 0.0)\n"
        )
        result = _run_scanner_source(src)
        combined = result.stdout + result.stderr

        assert result.returncode != 0
        assert "nested @triton.jit" in combined
        assert not any(line.startswith("< AT") for line in _token_lines(result.stdout))

    def test_scanner_longest_match_overlapping_lexemes(self):
        src = (
            "@triton.jit\n"
            "def k(a, b):\n"
            "    a **= 2\n"
            "    b = a ** 2 * 3\n"
            "    c = ...\n"
            "    d = .5\n"
            "    e = a // 2 / 3\n"
            "    f /= 2\n"
            "    g = a -> b\n"
            "    h <<= 1\n"
            "    i = h << 2\n"
        )
        result = _run_scanner_source(src)
        tokens = _token_lines(result.stdout)

        assert result.returncode == 0
        for token_name, lexeme in [
            ("DOUBLESTAREQ", "**="),
            ("DOUBLESTAR", "**"),
            ("STAR", "*"),
            ("ELLIPSIS", "..."),
            ("NUMBER_FLOAT", ".5"),
            ("DOUBLESLASH", "//"),
            ("SLASH", "/"),
            ("SLASHEQ", "/="),
            ("ARROW", "->"),
            ("LSHIFTEQ", "<<="),
            ("LSHIFT", "<<"),
        ]:
            assert any(token_name in line and lexeme in line for line in tokens), lexeme

    def test_scanner_suppresses_layout_inside_delimiters(self):
        src = (
            "@triton.jit\n"
            "def k(x_ptr):\n"
            "    value = tl.load(\n"
            "        x_ptr,\n"
            "        mask=True,\n"
            "    )\n"
            "    tl.store(x_ptr, value)\n"
        )
        result = _run_scanner_source(src)
        names = _token_names(result.stdout)

        assert result.returncode == 0
        assert names.count("INDENT") == 1
        assert names.count("DEDENT") == 1

    def test_scanner_symbol_table_is_unique_and_uses_first_line(self):
        src = (
            "@triton.jit\n"
            "def k(x_ptr):\n"
            "    value = 1\n"
            "    value = value + 1\n"
        )
        result = _run_scanner_source(src)
        symbol_table = result.stdout.split("=== Symbol table ===", maxsplit=1)[1]
        value_rows = [
            line for line in symbol_table.splitlines()
            if " NAME" in line and " value" in line
        ]

        assert result.returncode == 0
        assert len(value_rows) == 1
        assert value_rows[0].rstrip().endswith("3")

    def test_scanner_reports_unsupported_character(self):
        result = _run_scanner_source(
            "@triton.jit\n"
            "def k(x_ptr):\n"
            "    x = $\n"
        )

        assert result.returncode != 0
        assert "unsupported character '$'" in result.stderr

    def test_scanner_reports_inconsistent_indentation_and_recovers(self):
        result = _run_scanner_source(
            "@triton.jit\n"
            "def k(x_ptr):\n"
            "    ok = 1\n"
            "  bad = 2\n"
            "\n"
        )

        assert result.returncode != 0
        assert "inconsistent indentation" in result.stderr
        assert " bad" not in "\n".join(_token_lines(result.stdout))

    def test_scanner_reports_excessive_indentation_depth(self):
        lines = ["@triton.jit", "def k(x_ptr):"]
        for depth in range(65):
            lines.append(f"{'    ' * (depth + 1)}if True:")
        lines.append(f"{'    ' * 66}tl.store(x_ptr, 0.0)")
        result = _run_scanner_source("\n".join(lines) + "\n")

        assert result.returncode != 0
        assert "indentation nesting is too deep" in result.stderr

    def test_scanner_all_valid_fixtures(self):
        """The scanner should not crash on any valid fixture."""
        for f in sorted(FIXTURES_VALID.glob("*.py")):
            r = _run_scanner(f)
            assert r.returncode == 0, (
                f"Scanner failed on {f.name}:\n{r.stderr}"
            )

    def test_scanner_indent_dedent_balance(self):
        """The number of INDENT tokens must equal the number of DEDENT tokens."""
        result = _run_scanner(FIXTURES_VALID / "vector_add.py")
        assert result.returncode == 0
        lines = result.stdout.splitlines()
        n_indent  = sum(1 for l in lines if "INDENT"  in l and "DEDENT" not in l)
        n_dedent  = sum(1 for l in lines if "DEDENT"  in l)
        assert n_indent == n_dedent, (
            f"Unbalanced INDENT/DEDENT tokens: {n_indent} vs {n_dedent}"
        )

    def test_scanner_number_types(self):
        """The scanner recognizes numeric forms inside a Triton JIT block."""
        import tempfile, os
        src = (
            "import triton\n\n"
            "@triton.jit\n"
            "def k(x_ptr):\n"
            "    x = 0xFF\n"
            "    y = 0b1010\n"
            "    z = 1.5e-3\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                         delete=False, encoding="utf-8") as f:
            f.write(src)
            tmp = f.name
        try:
            result = _run_scanner(Path(tmp))
            assert result.returncode == 0
            assert "NUMBER" in result.stdout
        finally:
            os.unlink(tmp)

    def test_scanner_omits_octal_number_type(self):
        """Octal literals are outside the reduced Triton numeric subset."""
        import tempfile, os
        src = (
            "import triton\n\n"
            "@triton.jit\n"
            "def k(x_ptr):\n"
            "    x = 0o755\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                         delete=False, encoding="utf-8") as f:
            f.write(src)
            tmp = f.name
        try:
            result = _run_scanner(Path(tmp))
            assert "NUMBER_OCT" not in result.stdout
        finally:
            os.unlink(tmp)

    def test_scanner_string_types(self):
        """The scanner recognizes string forms inside a Triton JIT block."""
        import tempfile, os
        src = (
            "import triton\n\n"
            "@triton.jit\n"
            "def k(x_ptr):\n"
            "    a = r\"raw\"\n"
            "    b = f\"fmt\"\n"
            "    c = \"\"\"triple\"\"\"\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                         delete=False, encoding="utf-8") as f:
            f.write(src)
            tmp = f.name
        try:
            result = _run_scanner(Path(tmp))
            assert result.returncode == 0
            assert "STRING" in result.stdout
        finally:
            os.unlink(tmp)

    def test_scanner_break_continue_are_keyword_tokens(self):
        src = (
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    for i in range(N):\n"
            "        if i == 0:\n"
            "            continue\n"
            "        break\n"
        )
        result = _run_scanner_source(src)

        assert result.returncode == 0
        assert "< CONTINUE" in result.stdout
        assert "< BREAK" in result.stdout


# ---------------------------------------------------------------------------
# End-to-end integration tests (full compiler)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """
    Generate kernels in memory with tempfile and pass them directly to the C
    binary to validate grammar and the symbol table.
    """

    @pytest.fixture(autouse=True)
    def require_binary(self):
        _skip_if_no_binary()

    def _run_src(self, src: str) -> subprocess.CompletedProcess:
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                         delete=False, encoding="utf-8") as f:
            f.write(src)
            tmp = f.name
        try:
            return _run(Path(tmp))
        finally:
            os.unlink(tmp)

    def test_minimal_kernel(self):
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    pid = tl.program_id(axis=0)\n"
            "    tl.store(x_ptr + pid, 0.0)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, f"stderr: {r.stderr}"

    def test_kernel_with_comment_first_line(self):
        """Bug #1 regression: comment as the first body line."""
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    # first comment\n"
            "    pid = tl.program_id(axis=0)\n"
            "    tl.store(x_ptr + pid, 1.0)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, (
            f"Bug #1 regression, first-line comment:\n{r.stderr}"
        )

    def test_kernel_tl_arange_in_arithmetic(self):
        """Bug #2 regression: tl.arange embedded in an arithmetic expression."""
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    pid = tl.program_id(axis=0)\n"
            "    offs = pid * N + tl.arange(0, N)\n"
            "    tl.store(x_ptr + offs, 0.0)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, f"Bug #2 regression:\n{r.stderr}"
        # tl.arange must appear in the symbol table.
        assert "arange" in r.stdout, "Did not detect tl.arange in the symbol table"

    def test_kernel_tl_dot_in_aug_assign(self):
        """Bug #3 regression: tl.dot in aug-assign."""
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    offs = tl.arange(0, N)\n"
            "    a = tl.load(x_ptr + offs)\n"
            "    acc = tl.zeros((N,), dtype=tl.float32)\n"
            "    acc += tl.dot(a, a)\n"
            "    tl.store(x_ptr + offs, acc)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, f"Bug #3 regression:\n{r.stderr}"
        assert "dot" in r.stdout, "Did not detect tl.dot"

    def test_kernel_return_type_annotation(self):
        """Bug #4 regression: -> None in the kernel signature."""
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr) -> None:\n"
            "    pid = tl.program_id(axis=0)\n"
            "    tl.store(x_ptr + pid, 0.0)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, f"Bug #4 regression:\n{r.stderr}"

    def test_kernel_break_continue(self):
        """Bug #5/#6 regression: break and continue inside for."""
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    for i in range(N):\n"
            "        if i == 0:\n"
            "            continue\n"
            "        if i > 5:\n"
            "            break\n"
            "        tl.store(x_ptr + i, 1.0)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, f"Bug #5/6 regression:\n{r.stderr}"

    def test_kernel_raise_is_out_of_scope(self):
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    if N <= 0:\n"
            "        raise ValueError\n"
            "    tl.store(x_ptr, 0.0)\n"
        )
        r = self._run_src(src)
        assert r.returncode != 0
        assert "syntax error" in (r.stdout + r.stderr).lower()

    def test_two_constexpr_kernels(self):
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k1(a, N: tl.constexpr):\n"
            "    tl.store(a, 1.0)\n"
            "@triton.jit\n"
            "def k2(b, M: tl.constexpr):\n"
            "    tl.store(b, 2.0)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, f"stderr: {r.stderr}"
        assert "k1" in r.stdout and "k2" in r.stdout

    def test_deep_nesting(self):
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    for i in range(N):\n"
            "        for j in range(N):\n"
            "            if i != j:\n"
            "                tl.store(x_ptr + i * N + j, 0.0)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, f"Deep nesting failed:\n{r.stderr}"

    def test_bitwise_and_shift_expressions(self):
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    mask = (N - 1) & 0xFF\n"
            "    shifted = N >> 2\n"
            "    tl.store(x_ptr + mask, shifted)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, f"Bitwise/shift failed:\n{r.stderr}"

    def test_multiple_decorators(self):
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@staticmethod\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    tl.store(x_ptr, 0.0)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, f"Multiple decorators failed:\n{r.stderr}"

    def test_duplicate_kernel_error(self):
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    tl.store(x_ptr, 1.0)\n"
            "@triton.jit\n"
            "def k(y_ptr, M: tl.constexpr):\n"
            "    tl.store(y_ptr, 2.0)\n"
        )
        r = self._run_src(src)
        combined = r.stdout + r.stderr
        has_error = (
            r.returncode != 0
            or "error" in combined.lower()
            or "duplicate" in combined.lower()
            or "already declared" in combined.lower()
        )
        assert has_error, "Did not detect duplicate kernel"
