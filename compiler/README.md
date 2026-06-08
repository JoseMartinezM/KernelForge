# Compiler — Direct Syntax Translator para Triton-GPU

Parser Lex + Yacc que valida y traduce código Triton-GPU.
Requerimiento del curso de Compiladores.

## Archivos

| Archivo | Rol |
|---|---|
| `triton_lexer.py`    | Análisis léxico: tokeniza el fuente, maneja INDENT/DEDENT |
| `triton_parser.py`   | Análisis sintáctico + traducción: gramática BNF, acciones semánticas, tabla de símbolos |
| `triton_validator.py`| API pública: `validate(source)` → `ValidationResult` |
| `fixtures/valid/`    | Kernels Triton válidos (fixtures de prueba) |
| `fixtures/invalid/`  | Código con errores (fixtures de prueba) |

## Uso

```bash
# Instalar dependencias
uv sync --group dev

# Validar un archivo
uv run python -m compiler.triton_validator compiler/fixtures/valid/vector_add.py

# Correr tests
uv run pytest tests/compiler/ -v
```

## Tipo de compilador

Este proyecto es un **Direct Syntax Translator** (no un compilador completo):

- **Entrada**: código Python-estilo con construcciones Triton
- **Salida**: reporte de validación estructurado con:
  - Kernels detectados con `@triton.jit`
  - Parámetros y sus anotaciones (`tl.constexpr`)
  - Llamadas a la API Triton (`tl.load`, `tl.store`, etc.)
  - Errores sintácticos con número de línea
- **Diferencia con un compiler tradicional**: la traducción ocurre DURANTE el parsing
  mediante acciones semánticas embebidas en las reglas Yacc (no necesita un AST separado).

## Gramática

### Características clave

1. **Sin recursión izquierda** — usando la técnica estándar:
   ```
   A → A α | β   (recursión izquierda — ELIMINADA)
   se convierte en:
   A  → β A'
   A' → α A' | ε
   ```
   Esto se aplica a todas las expresiones binarias (`sum`, `term`, `bitor`, etc.)

2. **Indentación con INDENT/DEDENT** — el lexer emite tokens especiales:
   ```
   suite → NEWLINE INDENT stmt_seq DEDENT
   ```
   Esto elimina la ambigüedad del "dangling else" presente en otras gramáticas Python.

3. **Acciones semánticas** — cada regla Yacc tiene código Python que:
   - Llena la tabla de símbolos
   - Detecta patrones Triton-específicos
   - Emite el reporte de traducción

### Tabla de símbolos

Por cada función se registra:
- Nombre y línea de declaración
- Si tiene `@triton.jit`
- Parámetros con anotación y bandera `es_constexpr`
- Variables locales asignadas
- Llamadas a la API Triton detectadas (`tl.*`)

### Manejo de errores

- Errores léxicos: carácter ilegal → mensaje con número de línea
- Errores sintácticos: token inesperado → mensaje con línea y tipo de token
- Errores semánticos: función duplicada, `@triton.jit` faltante (advertencia)

## Herramientas usadas

- `ply.lex` — análisis léxico (Python Lex-Yacc)
- `ply.yacc` — análisis sintáctico LALR(1)

PLY implementa los mismos algoritmos que Flex/Bison en C, pero en Python puro,
lo que facilita la integración con el resto del proyecto KernelForge.
