"""
Tests para el Direct Syntax Translator (Lex + Yacc).

Estructura:
  - TestLexer:   verifica que el lexer produce los tokens correctos.
  - TestParser:  verifica que la gramática acepta código válido.
  - TestInvalid: verifica que la gramática rechaza código inválido.
  - TestSymbolTable: verifica que la tabla de símbolos se llena correctamente.
  - TestFixtures: corre los fixtures del directorio fixtures/.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from compiler.triton_lexer import build_lexer
from compiler.triton_validator import validate, validate_file

FIXTURES_VALID   = Path(__file__).parents[2] / "compiler" / "fixtures" / "valid"
FIXTURES_INVALID = Path(__file__).parents[2] / "compiler" / "fixtures" / "invalid"


# ---------------------------------------------------------------------------
# Lexer tests
# ---------------------------------------------------------------------------

class TestLexer:
    def _tokens(self, source: str) -> list[str]:
        lexer = build_lexer()
        return [tok.type for tok in lexer.tokenize(source)]

    def test_keywords_recognized(self):
        toks = self._tokens("def if else return")
        assert "DEF" in toks
        assert "IF" in toks
        assert "ELSE" in toks
        assert "RETURN" in toks

    def test_identifier(self):
        toks = self._tokens("my_var_123")
        assert toks == ["NAME", "NEWLINE"]

    def test_number_int(self):
        toks = self._tokens("42")
        assert "NUMBER" in toks

    def test_number_float(self):
        toks = self._tokens("3.14")
        assert "NUMBER" in toks

    def test_number_hex(self):
        toks = self._tokens("0xFF")
        assert "NUMBER" in toks

    def test_string_double(self):
        toks = self._tokens('"hello"')
        assert "STRING" in toks

    def test_string_single(self):
        toks = self._tokens("'world'")
        assert "STRING" in toks

    def test_operators(self):
        toks = self._tokens("+ - * / // ** << >> == != <= >=")
        assert "PLUS"        in toks
        assert "MINUS"       in toks
        assert "STAR"        in toks
        assert "SLASH"       in toks
        assert "DOUBLESLASH" in toks
        assert "DOUBLESTAR"  in toks
        assert "LSHIFT"      in toks
        assert "RSHIFT"      in toks
        assert "EQEQ"        in toks
        assert "NOTEQ"       in toks
        assert "LTEQ"        in toks
        assert "GTEQ"        in toks

    def test_at_decorator(self):
        toks = self._tokens("@triton.jit")
        assert "AT" in toks

    def test_indent_dedent_simple(self):
        """Un bloque de 4 espacios debe producir INDENT + DEDENT."""
        source = "if True:\n    pass\n"
        toks = self._tokens(source)
        assert "INDENT" in toks
        assert "DEDENT" in toks

    def test_indent_dedent_nested(self):
        """Dos niveles de indentación producen dos INDENT y dos DEDENT."""
        source = "if True:\n    if True:\n        pass\n"
        toks = self._tokens(source)
        assert toks.count("INDENT") == 2
        assert toks.count("DEDENT") == 2

    def test_paren_newline_ignored(self):
        """Newlines dentro de paréntesis NO generan INDENT/DEDENT."""
        source = "x = (\n    1 +\n    2\n)\n"
        toks = self._tokens(source)
        assert "INDENT" not in toks
        assert "DEDENT" not in toks

    def test_comment_ignored(self):
        """Los comentarios no producen tokens."""
        toks = self._tokens("# esto es un comentario\nx = 1")
        assert "COMMENT" not in toks
        assert "NAME" in toks


# ---------------------------------------------------------------------------
# Parser / validator tests
# ---------------------------------------------------------------------------

class TestParser:
    def test_empty_program(self):
        result = validate("")
        assert result.is_valid

    def test_simple_assignment(self):
        result = validate("x = 1\n")
        assert result.is_valid

    def test_import(self):
        result = validate("import triton\nimport triton.language as tl\n")
        assert result.is_valid

    def test_function_no_decorator(self):
        result = validate("def foo(x):\n    return x\n")
        assert result.is_valid
        assert "foo" in result.symbol_table
        # Advertencia sobre falta de @triton.jit
        assert any("triton.jit" in w for w in result.warnings)

    def test_triton_kernel_basic(self):
        source = (
            "import triton\n"
            "import triton.language as tl\n"
            "@triton.jit\n"
            "def add_kernel(x_ptr, n, BLOCK_SIZE: tl.constexpr):\n"
            "    pid = tl.program_id(axis=0)\n"
            "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
            "    mask = offsets < n\n"
            "    x = tl.load(x_ptr + offsets, mask=mask)\n"
            "    tl.store(x_ptr + offsets, x, mask=mask)\n"
        )
        result = validate(source)
        assert result.is_valid, f"Errores: {result.errors}"
        assert "add_kernel" in result.symbol_table
        kernel = result.symbol_table["add_kernel"]
        assert kernel["tiene_triton_jit"]
        params = {p["nombre"]: p for p in kernel["parametros"]}
        assert params["BLOCK_SIZE"]["es_constexpr"]

    def test_if_else(self):
        source = (
            "if x > 0:\n"
            "    y = 1\n"
            "else:\n"
            "    y = 0\n"
        )
        result = validate(source)
        assert result.is_valid, f"Errores: {result.errors}"

    def test_if_elif_else(self):
        source = (
            "if x > 0:\n"
            "    y = 1\n"
            "elif x == 0:\n"
            "    y = 0\n"
            "else:\n"
            "    y = -1\n"
        )
        result = validate(source)
        assert result.is_valid, f"Errores: {result.errors}"

    def test_for_loop(self):
        source = (
            "for i in tl.range(BLOCK_SIZE):\n"
            "    x = x + 1\n"
        )
        result = validate(source)
        assert result.is_valid, f"Errores: {result.errors}"

    def test_while_loop(self):
        source = "while x > 0:\n    x = x - 1\n"
        result = validate(source)
        assert result.is_valid, f"Errores: {result.errors}"

    def test_nested_blocks(self):
        source = (
            "@triton.jit\n"
            "def kernel(x, BLOCK_SIZE: tl.constexpr):\n"
            "    for i in tl.range(BLOCK_SIZE):\n"
            "        if i > 0:\n"
            "            x = x + i\n"
        )
        result = validate(source)
        assert result.is_valid, f"Errores: {result.errors}"

    def test_augmented_assignment(self):
        result = validate("x += 1\n")
        assert result.is_valid

    def test_annotated_assignment(self):
        result = validate("acc: tl.float32 = 0.0\n")
        assert result.is_valid

    def test_assert_stmt(self):
        result = validate("assert x > 0, 'x debe ser positivo'\n")
        assert result.is_valid

    def test_pass_stmt(self):
        result = validate("def empty(x):\n    pass\n")
        assert result.is_valid

    def test_multiline_params(self):
        source = (
            "@triton.jit\n"
            "def kernel(\n"
            "    x_ptr,\n"
            "    y_ptr,\n"
            "    n: tl.constexpr,\n"
            "):\n"
            "    pass\n"
        )
        result = validate(source)
        assert result.is_valid, f"Errores: {result.errors}"

    def test_expression_precedence(self):
        result = validate("z = x + y * 2 - 1 // 3\n")
        assert result.is_valid

    def test_bitwise_ops(self):
        result = validate("mask = (offsets < n) & (offsets >= 0)\n")
        assert result.is_valid

    def test_ternary_expr(self):
        result = validate("val = a if cond else b\n")
        assert result.is_valid

    def test_subscript(self):
        result = validate("x = arr[0:BLOCK_SIZE]\n")
        assert result.is_valid


# ---------------------------------------------------------------------------
# Invalid code tests
# ---------------------------------------------------------------------------

class TestInvalid:
    def test_bad_syntax_missing_colon(self):
        """def sin ':' es error sintáctico."""
        source = "def broken(x)\n    pass\n"
        result = validate(source)
        assert not result.is_valid or result.errors, "Debería haber errores"

    def test_duplicate_function(self):
        """Dos funciones con el mismo nombre → error semántico."""
        source = (
            "def foo(x):\n    pass\n"
            "def foo(y):\n    pass\n"
        )
        result = validate(source)
        assert result.errors, "Debería detectar declaración duplicada"


# ---------------------------------------------------------------------------
# Symbol table tests
# ---------------------------------------------------------------------------

class TestSymbolTable:
    def test_params_registered(self):
        source = (
            "@triton.jit\n"
            "def kernel(a_ptr, b_ptr, n, BLOCK_SIZE: tl.constexpr):\n"
            "    pass\n"
        )
        result = validate(source)
        assert result.is_valid
        params = {p["nombre"]: p for p in result.symbol_table["kernel"]["parametros"]}
        assert "a_ptr"      in params
        assert "b_ptr"      in params
        assert "n"          in params
        assert "BLOCK_SIZE" in params
        assert params["BLOCK_SIZE"]["es_constexpr"]
        assert not params["n"]["es_constexpr"]

    def test_triton_jit_flag(self):
        source = "@triton.jit\ndef k(x):\n    pass\n"
        result = validate(source)
        assert result.symbol_table["k"]["tiene_triton_jit"]

    def test_no_triton_jit_flag(self):
        source = "def k(x):\n    pass\n"
        result = validate(source)
        assert not result.symbol_table["k"]["tiene_triton_jit"]


# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------

class TestFixtures:
    @pytest.mark.parametrize("fixture", list(FIXTURES_VALID.glob("*.py")))
    def test_valid_fixture(self, fixture: Path):
        result = validate_file(fixture)
        assert result.is_valid, (
            f"Fixture válido '{fixture.name}' falló:\n" + "\n".join(result.errors)
        )

    @pytest.mark.parametrize("fixture", list(FIXTURES_INVALID.glob("*.py")))
    def test_invalid_fixture_detected(self, fixture: Path):
        result = validate_file(fixture)
        # Debe tener errores O advertencias (algo debe detectar)
        has_issues = bool(result.errors) or bool(result.warnings)
        assert has_issues, (
            f"Fixture inválido '{fixture.name}' pasó sin detectar nada"
        )
