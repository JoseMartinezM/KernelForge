"""
compiler/triton_validator.py
============================
API pública del Direct Syntax Translator.

Uso:
    from compiler.triton_validator import validate

    result = validate(source_code)
    print(result.report)          # reporte de traducción
    print(result.is_valid)        # True / False
    print(result.errors)          # lista de errores con línea
    print(result.symbol_table)    # tabla de símbolos (dict)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .triton_parser import parse, SymbolTable


@dataclass
class ValidationResult:
    is_valid:     bool
    report:       str
    errors:       list[str]       = field(default_factory=list)
    warnings:     list[str]       = field(default_factory=list)
    symbol_table: dict            = field(default_factory=dict)


def validate(source: str) -> ValidationResult:
    """
    Valida código Triton y retorna el resultado de la traducción.

    Este es el punto de entrada del Direct Syntax Translator:
      1. El lexer tokeniza el fuente (tokens + INDENT/DEDENT).
      2. El parser aplica las reglas BNF y ejecuta acciones semánticas.
      3. La tabla de símbolos captura kernels, parámetros y llamadas Triton.
      4. Se genera el reporte de traducción.
    """
    symtab, _ast = parse(source)
    is_valid = len(symtab.errors) == 0

    return ValidationResult(
        is_valid     = is_valid,
        report       = symtab.report(),
        errors       = symtab.errors,
        warnings     = symtab.warnings,
        symbol_table = symtab.functions,
    )


def validate_file(path: str | Path) -> ValidationResult:
    """Valida un archivo de código Triton desde disco."""
    source = Path(path).read_text(encoding="utf-8")
    return validate(source)


# ---------------------------------------------------------------------------
# CLI simple: python -m compiler.triton_validator <archivo.py>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python -m compiler.triton_validator <archivo.py>")
        sys.exit(1)

    result = validate_file(sys.argv[1])
    print(result.report)
    sys.exit(0 if result.is_valid else 1)
