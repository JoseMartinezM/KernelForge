"""
Tests para el compilador C (triton_compiler.exe / triton_compiler).

Corre el binario compilado con bison+flex contra los fixtures de .py
y verifica exit codes, stdout y stderr.

Prerequisito: compilar primero desde compiler/:
    cd compiler && make compiler
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------

COMPILER_DIR     = Path(__file__).parents[2] / "compiler"
FIXTURES_DIR     = Path(__file__).parent / "fixtures"
FIXTURES_VALID   = FIXTURES_DIR / "valid"
FIXTURES_INVALID = FIXTURES_DIR / "invalid"

# Nombre del binario (Windows usa .exe, Linux/Mac sin extensión)
_BINARY_STEM = "triton_compiler"
_BINARY = COMPILER_DIR / (_BINARY_STEM + (".exe" if sys.platform == "win32" else ""))

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _run(fixture: Path, timeout: int = 10) -> subprocess.CompletedProcess:
    """Ejecuta triton_compiler sobre un fixture y retorna el resultado."""
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
            f"Binario '{_BINARY.name}' no encontrado en {COMPILER_DIR}. "
            "Compila con: cd compiler && make compiler"
        )


# ---------------------------------------------------------------------------
# Fixtures válidos — exit 0, no "error" en stderr
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
            f"{fixture.name} reportó 'syntax error':\n{result.stderr}"
        )

    def test_vector_add_symbol_table(self):
        result = _run(FIXTURES_VALID / "vector_add.py")
        assert result.returncode == 0
        out = result.stdout
        assert "add_kernel" in out, "No encontró nombre del kernel en la salida"
        assert "triton.jit" in out.lower() or "jit" in out.lower(), (
            "No reportó @triton.jit"
        )

    def test_softmax_detects_tl_calls(self):
        result = _run(FIXTURES_VALID / "softmax_kernel.py")
        assert result.returncode == 0
        out = result.stdout
        assert any(api in out for api in ["tl.", "program_id", "load", "store"]), (
            "softmax_kernel no reportó ninguna API de Triton"
        )

    def test_matmul_detects_dot(self):
        result = _run(FIXTURES_VALID / "matmul_kernel.py")
        assert result.returncode == 0
        # tl.dot debe aparecer en la tabla de llamadas Triton
        assert "dot" in result.stdout, (
            "matmul_kernel no detectó tl.dot en la tabla de símbolos"
        )

    def test_dropout_no_jit_warning(self):
        """dropout_kernel tiene @triton.jit — no debe haber advertencia de JIT ausente."""
        result = _run(FIXTURES_VALID / "dropout_kernel.py")
        assert result.returncode == 0
        combined = result.stdout + result.stderr
        assert "sin decorador" not in combined and "without" not in combined.lower(), (
            "dropout_kernel reportó advertencia de JIT ausente siendo que sí tiene @triton.jit"
        )

    def test_flash_attention_constexpr_params(self):
        result = _run(FIXTURES_VALID / "flash_attention.py")
        assert result.returncode == 0
        out = result.stdout
        # Debe reportar parámetros constexpr
        assert "constexpr" in out.lower() or "BLOCK_M" in out, (
            "flash_attention no reportó parámetros tl.constexpr"
        )

    def test_layer_norm_two_kernels(self):
        result = _run(FIXTURES_VALID / "layer_norm.py")
        assert result.returncode == 0
        out = result.stdout
        assert "layer_norm_kernel" in out, "No encontró layer_norm_kernel"
        assert "layer_norm_bwd_dx_fused" in out, "No encontró layer_norm_bwd_dx_fused"


# ---------------------------------------------------------------------------
# Fixtures inválidos — deben producir error o advertencia
# ---------------------------------------------------------------------------

class TestInvalidFixtures:

    @pytest.fixture(autouse=True)
    def require_binary(self):
        _skip_if_no_binary()

    def test_bad_syntax_nonzero_exit(self):
        result = _run(FIXTURES_INVALID / "bad_syntax.py")
        assert result.returncode != 0, (
            "bad_syntax.py debería fallar (exit != 0) pero salió con 0"
        )

    def test_bad_syntax_reports_error(self):
        result = _run(FIXTURES_INVALID / "bad_syntax.py")
        combined = result.stdout.lower() + result.stderr.lower()
        assert "error" in combined or "syntax" in combined, (
            "bad_syntax.py no reportó ningún error"
        )

    def test_duplicate_kernel_detected(self):
        result = _run(FIXTURES_INVALID / "duplicate_kernel.py")
        combined = result.stdout + result.stderr
        assert "add_kernel" in combined, (
            "duplicate_kernel.py: no mencionó el nombre duplicado 'add_kernel'"
        )
        has_error = (
            result.returncode != 0
            or "error" in combined.lower()
            or "duplicad" in combined.lower()
            or "ya declarad" in combined.lower()
        )
        assert has_error, (
            "duplicate_kernel.py no reportó la redefinición del kernel"
        )

    def test_missing_jit_warns(self):
        result = _run(FIXTURES_INVALID / "missing_jit.py")
        combined = result.stdout + result.stderr
        has_warning = (
            "advertencia" in combined.lower()
            or "warning" in combined.lower()
            or "sin decorador" in combined.lower()
            or "triton.jit" in combined.lower()
        )
        assert has_warning, (
            "missing_jit.py no generó advertencia sobre @triton.jit ausente"
        )


# ---------------------------------------------------------------------------
# Tests de correctitud del scanner (tokens concretos)
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


class TestScanner:

    @pytest.fixture(autouse=True)
    def require_scanner(self):
        if not SCANNER_BINARY.exists():
            pytest.skip(
                f"Scanner '{SCANNER_BINARY.name}' no encontrado. "
                "Compila con: cd compiler && make scanner"
            )

    def test_scanner_vector_add_tokens(self):
        result = _run_scanner(FIXTURES_VALID / "vector_add.py")
        assert result.returncode == 0
        out = result.stdout
        # Tokens que siempre deben aparecer en un kernel bien formado
        assert "DEF"     in out, "No encontró token DEF"
        assert "INDENT"  in out, "No encontró token INDENT"
        assert "DEDENT"  in out, "No encontró token DEDENT"
        assert "NEWLINE" in out, "No encontró token NEWLINE"

    def test_scanner_emits_jit_tokens(self):
        result = _run_scanner(FIXTURES_VALID / "vector_add.py")
        out = result.stdout
        assert "triton" in out or "NAME" in out, (
            "No encontró identificadores en el stream de tokens"
        )

    def test_scanner_all_valid_fixtures(self):
        """El scanner no debe explotar en ningún fixture válido."""
        for f in sorted(FIXTURES_VALID.glob("*.py")):
            r = _run_scanner(f)
            assert r.returncode == 0, (
                f"Scanner falló en {f.name}:\n{r.stderr}"
            )

    def test_scanner_indent_dedent_balance(self):
        """Número de INDENT debe ser igual a número de DEDENT."""
        result = _run_scanner(FIXTURES_VALID / "vector_add.py")
        assert result.returncode == 0
        lines = result.stdout.splitlines()
        n_indent  = sum(1 for l in lines if "INDENT"  in l and "DEDENT" not in l)
        n_dedent  = sum(1 for l in lines if "DEDENT"  in l)
        assert n_indent == n_dedent, (
            f"INDENT/DEDENT desbalanceados: {n_indent} vs {n_dedent}"
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


# ---------------------------------------------------------------------------
# Tests de integración end-to-end (compiler completo)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """
    Genera kernels en memoria con tempfile y los pasa directamente
    al binario C para validar gramática y tabla de símbolos.
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
        """Bug #1 regresión: comentario como primera línea del cuerpo."""
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    # primer comentario\n"
            "    pid = tl.program_id(axis=0)\n"
            "    tl.store(x_ptr + pid, 1.0)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, (
            f"Bug #1 regresión — comentario primera línea:\n{r.stderr}"
        )

    def test_kernel_tl_arange_in_arithmetic(self):
        """Bug #2 regresión: tl.arange embebido en expresión aritmética."""
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    pid = tl.program_id(axis=0)\n"
            "    offs = pid * N + tl.arange(0, N)\n"
            "    tl.store(x_ptr + offs, 0.0)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, f"Bug #2 regresión:\n{r.stderr}"
        # tl.arange debe aparecer en la tabla de símbolos
        assert "arange" in r.stdout, "No detectó tl.arange en tabla de símbolos"

    def test_kernel_tl_dot_in_aug_assign(self):
        """Bug #3 regresión: tl.dot en aug-assign."""
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
        assert r.returncode == 0, f"Bug #3 regresión:\n{r.stderr}"
        assert "dot" in r.stdout, "No detectó tl.dot"

    def test_kernel_return_type_annotation(self):
        """Bug #4 regresión: -> None en la firma del kernel."""
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr) -> None:\n"
            "    pid = tl.program_id(axis=0)\n"
            "    tl.store(x_ptr + pid, 0.0)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, f"Bug #4 regresión:\n{r.stderr}"

    def test_kernel_break_continue(self):
        """Bug #5/#6 regresión: break y continue dentro de for."""
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
        assert r.returncode == 0, f"Bug #5/6 regresión:\n{r.stderr}"

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
        assert r.returncode == 0, f"Deep nesting falló:\n{r.stderr}"

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
        assert r.returncode == 0, f"Bitwise/shift falló:\n{r.stderr}"

    def test_multiple_decorators(self):
        src = (
            "import triton\nimport triton.language as tl\n\n"
            "@staticmethod\n"
            "@triton.jit\n"
            "def k(x_ptr, N: tl.constexpr):\n"
            "    tl.store(x_ptr, 0.0)\n"
        )
        r = self._run_src(src)
        assert r.returncode == 0, f"Múltiples decoradores fallaron:\n{r.stderr}"

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
            or "duplicad" in combined.lower()
            or "ya declarad" in combined.lower()
        )
        assert has_error, "No detectó kernel duplicado"
