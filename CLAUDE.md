
# KernelForge GPU Kernel Generator

## Visión del proyecto

Sistema que toma **TritonBench como base** y le añade una capa de gramáticas formales
(Lex + Yacc) para guiar y validar la generación de kernels GPU en Triton con tres
niveles de optimización garantizados. El LLM genera el kernel; nuestro parser lo
valida contra la gramática del nivel solicitado; si falla, el sistema hace fallback
automático al nivel inferior.

**Problema que resuelve:** TritonBench ya evalúa si un LLM puede generar código Triton.
Lo que no tiene es ningún mecanismo para *guiar* al LLM ni para garantizar que el código
generado cumple patrones de optimización específicos. Eso es lo que KernelForge añade.

---

## Base: TritonBench (vendor — no modificar)

TritonBench vive en `vendor/TritonBench/` y se usa tal como está.

**Lo que nos da gratis:**

| Componente | Uso en KernelForge |
|---|---|
| `data/TritonBench_G_v1/` | Kernels de referencia con tests incluidos |
| `data/train_crawl.json` + `train_synth.json` | 8k ejemplos para RAG en el prompt |
| `EVAL/eval_G/0_call_acc.py` | Validar que el kernel pasa su test funcional |
| `EVAL/eval_G/1_exe_acc.py` | Validar que el kernel ejecuta sin crash |
| `EVAL/eval_G/2_efficiency.py` | Medir speedup GPU vs CPU |
| `LLM_generated/` | Resultados baseline (GPT-4o, Claude, DeepSeek) sin gramática |

**Versiones requeridas por TritonBench:** `triton = 3.1.0`, `torch >= 2.5.1`

---

## Nuestra contribución: Capa Lex + Yacc

Lo que construimos encima de TritonBench:

```
Código Triton generado por LLM
            │
            ▼
    ┌───────────────┐
    │  Lexer (PLY)  │  ← tokeniza el código Triton
    └───────┬───────┘
            │ tokens
            ▼
    ┌───────────────┐
    │ Parser Yacc   │  ← valida contra gramática L1 / L2 / L3
    └───────┬───────┘
            │
     pasa   │   falla → FallbackHandler → nivel inferior
            ▼
    ┌────────────────────┐
    │ TritonBench EVAL   │  ← reutilizamos call_acc → exe_acc → speedup
    │  + métricas propias│  ← qué regla falló, nivel logrado, tasa fallback
    └────────────────────┘
```

### Por qué Lex + Yacc y no solo XGrammar

- **XGrammar** controla la distribución de probabilidad del LLM *durante* la generación
  (requiere acceso a los logits — solo funciona con Ollama local, no con APIs).
- **Lex + Yacc** es validación *post-generación*: el LLM genera libremente, luego
  nuestro parser verifica si el código cumple las reglas del nivel pedido.
- Ventaja: funciona con cualquier LLM (Claude API, GPT-4o, Ollama) sin acceso a logits.
- Ventaja académica: permite comparar el mismo LLM con y sin gramática de forma limpia.

**Librería:** `PLY` (Python Lex-Yacc) — es el estándar académico, bien documentado,
y genera parsers LALR(1) que son suficientes para nuestra gramática.

---

## Tokens Lex (tokenizador Triton)

El lexer convierte el código generado en tokens que el parser puede analizar.

```
Categoría          Tokens
─────────────────────────────────────────────────────
Imports            IMPORT_TRITON, IMPORT_TL
Decorador          TRITON_JIT
Estructura         DEF, COLON, LPAREN, RPAREN, COMMA
Triton API         PROGRAM_ID, ARANGE, LOAD, STORE
                   ZEROS, DEBUG_BARRIER, CONSTEXPR
Optimización       BLOCK_SIZE (identificador especial)
Literales          NUMBER (entero), FLOAT, STRING
Operadores         PLUS, STAR, LT, EQ, ASSIGN
Identificadores    NAME (nombres de variables/funciones)
Ignorados          espacios, comentarios, newlines vacíos
```

---

## Reglas Yacc por nivel

### Nivel 1 — Sintaxis básica válida

Garantiza que el kernel es Triton sintácticamente correcto y tiene la
estructura mínima requerida: imports, decorador, función, pid, offsets, mask,
al menos un load, una operación, un store.

```
kernel      → imports decorator funcdef
funcdef     → DEF NAME LPAREN params RPAREN COLON body
body        → pid_stmt offsets_stmt mask_stmt load_stmts compute_stmt store_stmt
pid_stmt    → NAME ASSIGN PROGRAM_ID LPAREN axis=0 RPAREN
offsets_stmt→ NAME ASSIGN NAME STAR NAME PLUS ARANGE LPAREN 0 COMMA NAME RPAREN
mask_stmt   → NAME ASSIGN NAME LT NAME
load_stmt   → NAME ASSIGN LOAD LPAREN expr COMMA mask=NAME RPAREN
store_stmt  → STORE LPAREN expr COMMA NAME COMMA mask=NAME RPAREN
```

**Qué detecta como error:** función sin decorador, store sin mask, load sin mask,
ausencia de pid o de cálculo de offsets.

---

### Nivel 2 — Memory Coalescing (extiende L1)

Garantiza que los patrones de acceso a memoria son coalescidos: hilos adyacentes
leen/escriben posiciones contiguas. Añade estas reglas encima de L1:

```
params          → ... BLOCK_SIZE COLON CONSTEXPR ...   (obligatorio)
block_size_val  → 64 | 128 | 256 | 512 | 1024         (solo potencias de 2)
offsets_pattern → pid STAR BLOCK_SIZE PLUS ARANGE LPAREN 0 COMMA BLOCK_SIZE RPAREN
                  (patrón exacto — no se aceptan variaciones)
```

**Qué detecta como error:** BLOCK_SIZE no es constexpr, valor no es potencia de 2,
patrón de offsets que rompe el coalescing (e.g., acceso strided arbitrario).

---

### Nivel 3 — Tiling + Shared Memory (extiende L2)

Garantiza que el kernel usa shared memory y tile loops para reducir accesos a
memoria global. Añade estas reglas encima de L2:

```
shared_alloc → NAME ASSIGN ZEROS LPAREN LBRACKET BLOCK_SIZE RBRACKET COMMA dtype RPAREN
tile_loop    → FOR NAME IN RANGE LPAREN 0 COMMA NAME COMMA BLOCK_SIZE RPAREN COLON tile_body
tile_body    → shared_load tile_compute barrier
barrier      → DEBUG_BARRIER LPAREN RPAREN
```

**Qué detecta como error:** no hay allocación de shared memory, tile loop ausente,
falta de barrier entre fases de carga y cómputo.

---

## FallbackHandler — lógica de degradación

```
intento 1: validar contra L3
    → pasa: continuar con TritonBench eval
    → falla: registrar qué regla L3 violó, intentar L2

intento 2: validar contra L2
    → pasa: continuar (con nota de fallback L3→L2)
    → falla: registrar qué regla L2 violó, intentar L1

intento 3: validar contra L1
    → pasa: continuar (con nota de fallback L3→L1)
    → falla: kernel inválido — reportar error completo
```

El fallback no re-genera el kernel. Solo aplica un parser más permisivo al mismo
output del LLM. Si el LLM genera un kernel L2 cuando se pedía L3, el sistema lo
acepta en L2 en lugar de rechazarlo.

---

## Métricas propias (lo que TritonBench no tiene)

Además de las métricas de TritonBench (call_acc, exe_acc, speedup), registramos:

| Métrica | Descripción |
|---|---|
| `grammar_level_requested` | Nivel pedido por el usuario (1/2/3) |
| `grammar_level_achieved` | Nivel que realmente pasó la validación |
| `fallback_occurred` | bool — ¿hubo degradación? |
| `failed_rule` | Nombre de la regla Yacc que falló |
| `fallback_reason` | Texto legible: "BLOCK_SIZE no es potencia de 2" |
| `parse_time_ms` | Tiempo del lexer + parser |
| `fallback_rate_by_op` | % de fallbacks por operación (tabla para el paper) |

---

## Tabla del paper: TritonBench baseline vs KernelForge

| Condición | Call Acc | Exe Acc | Speedup | Nivel logrado |
|---|---|---|---|---|
| GPT-4o sin gramática (baseline TritonBench) | — | — | — | — |
| Claude sin gramática (baseline TritonBench) | — | — | — | — |
| + KernelForge L1 | — | — | — | básico |
| + KernelForge L2 | — | — | — | coalesced |
| + KernelForge L3 | — | — | — | tiled |

La columna "Nivel logrado" + "failed_rule" + "fallback_rate" es la novedad
que aporta este trabajo sobre TritonBench. Eso es lo publicable.

---

## Plan de implementación (fases)

### Fase 1 — Lexer Triton
Construir el tokenizador PLY que reconoce todos los tokens del lenguaje Triton
relevantes para nuestras gramáticas. Verificar contra los kernels de `TritonBench_G_v1/`.

### Fase 2 — Parser L1
Implementar las reglas Yacc para sintaxis básica. Probar contra kernels simples
de TritonBench (`add_example.py`, `add_value.py`, `adam_update_triton.py`).

### Fase 3 — Parser L2
Extender L1 con reglas de coalescing. El parser L2 debe rechazar kernels válidos
en L1 que usen BLOCK_SIZE sin constexpr o con valores no potencia de 2.

### Fase 4 — Parser L3
Extender L2 con reglas de tiling y shared memory. Probar contra kernels de atención
(`attention_kernel.py`, `attn_fwd_triton.py`) que ya usan estos patrones.

### Fase 5 — FallbackHandler
Conectar los tres parsers con la lógica de degradación. Registrar todas las métricas
propias en cada intento.

### Fase 6 — Integración con TritonBench EVAL
Conectar el output del FallbackHandler con `0_call_acc.py` → `1_exe_acc.py` →
`2_efficiency.py`. El kernel que pasó la gramática se pasa al evaluador de TritonBench.

### Fase 7 — Métricas y reporte
Generar la tabla del paper automáticamente desde los resultados de las fases 5 y 6.
Incluir fallback_rate por operación y failed_rule más frecuente.

### Fase 8 — FastAPI + Frontend (opcional para demo)
Exponer el pipeline via API REST y mostrar resultados en dashboard React con Monaco Editor.

---

## Estructura de directorios

```
KernelForge/
│
├── CLAUDE.md
├── README.md
│
├── vendor/
│   └── TritonBench/              # NO MODIFICAR — usar como está
│       ├── data/
│       ├── EVAL/
│       └── LLM_generated/
│
├── backend/
│   ├── requirements.txt          # PLY, FastAPI, pydantic, pytest
│   │
│   └── app/
│       ├── main.py               # FastAPI app
│       │
│       ├── grammars/
│       │   ├── lexer.py          # PLY lex — todos los tokens Triton
│       │   ├── parser_l1.py      # PLY yacc — reglas nivel 1
│       │   ├── parser_l2.py      # PLY yacc — reglas nivel 2 (extiende L1)
│       │   ├── parser_l3.py      # PLY yacc — reglas nivel 3 (extiende L2)
│       │   └── grammar_result.py # Dataclass: nivel logrado, regla fallida
│       │
│       ├── core/
│       │   ├── fallback_handler.py  # Orquesta L3→L2→L1, registra métricas
│       │   ├── tritonbench_eval.py  # Wrapper sobre los scripts de EVAL/
│       │   └── metrics_collector.py # Agrega métricas propias + TritonBench
│       │
│       ├── llm/
│       │   ├── claude_runner.py  # Claude API (baseline sin gramática)
│       │   └── ollama_runner.py  # Ollama (con XGrammar opcional)
│       │
│       └── api/
│           └── routes/
│               └── generate.py   # POST /generate — pipeline completo
│
├── tests/
│   ├── test_lexer.py             # Tokens correctos sobre kernels de referencia
│   ├── test_parser_l1.py         # L1 acepta válidos, rechaza inválidos
│   ├── test_parser_l2.py         # L2 rechaza no-coalesced
│   ├── test_parser_l3.py         # L3 rechaza sin shared memory
│   └── test_fallback.py          # Fallback funciona correctamente
│
├── notebooks/
│   ├── 01_lexer_exploration.ipynb    # Explorar tokens sobre kernels reales
│   ├── 02_grammar_levels.ipynb       # Probar parsers L1/L2/L3 manualmente
│   └── 03_paper_experiments.ipynb    # Pipeline completo, genera tabla del paper
│
└── paper/
    ├── main.tex
    └── figures/
```

---

## Stack tecnológico

### Backend
- **Python 3.11+**
- **PLY** — lexer y parser (Lex + Yacc en Python)
- **FastAPI** — API REST para el pipeline
- **Anthropic SDK** — Claude API como LLM principal
- **Ollama** — LLMs locales (DeepSeek-Coder, CodeLlama) con XGrammar opcional
- **Triton 3.1.0** — alineado con TritonBench
- **PyTorch >= 2.5.1** — alineado con TritonBench
- **Pydantic v2** — validación de schemas
- **pytest** — tests de gramáticas y pipeline

### Frontend (Fase 8)
- **React 18 + TypeScript**
- **Vite** + **Tailwind CSS**
- **Monaco Editor** — visualizar el kernel generado
- **Recharts** — speedup, fallback rates, nivel logrado

---

## Gramáticas — diseño EBNF de referencia

### Nivel 1
```ebnf
kernel      ::= imports decorator funcdef
imports     ::= "import triton" newline "import triton.language as tl" newline
decorator   ::= "@triton.jit" newline
funcdef     ::= "def " identifier "(" params "):" newline body
body        ::= pid_stmt offsets_stmt mask_stmt load_stmts compute_stmt store_stmt
pid_stmt    ::= identifier " = tl.program_id(axis=0)"
offsets_stmt::= identifier " = " identifier " * " identifier " + tl.arange(0, " identifier ")"
mask_stmt   ::= identifier " = " identifier " < " identifier
load_stmt   ::= identifier " = tl.load(" expr ", mask=" identifier ")"
store_stmt  ::= "tl.store(" expr ", " identifier ", mask=" identifier ")"
```

### Nivel 2 (agrega sobre L1)
```ebnf
block_size_decl ::= "BLOCK_SIZE: tl.constexpr"
block_size_val  ::= "64" | "128" | "256" | "512" | "1024"
offsets_pattern ::= "pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)"
```

### Nivel 3 (agrega sobre L2)
```ebnf
shared_alloc ::= identifier " = tl.zeros([BLOCK_SIZE], dtype=" dtype ")"
tile_loop    ::= "for " identifier " in range(0, " identifier ", BLOCK_SIZE):" newline tile_body
tile_body    ::= shared_load tile_compute barrier
barrier      ::= "tl.debug_barrier()"
```

---

## Variables de entorno (.env)

```bash
# LLM Provider: "ollama" | "claude"
LLM_PROVIDER=claude

# Claude API
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-6

# Ollama (alternativa local)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=deepseek-coder:33b

# GPU
CUDA_VISIBLE_DEVICES=0
TRITON_CACHE_DIR=/tmp/triton_cache

# TritonBench paths
TRITONBENCH_DATA=vendor/TritonBench/data
TRITONBENCH_EVAL=vendor/TritonBench/EVAL

# App
BACKEND_PORT=8000
MAX_FALLBACK_ATTEMPTS=3
```

---

## Operaciones GPU soportadas (v1)

| Operación | Archivo TritonBench | Nivel inicial sugerido |
|---|---|---|
| `vector_add` | `add_example.py` | L3 |
| `vector_mul` | `add_value.py` | L3 |
| `dot_product` | `batched_vecmat_mult.py` | L2 (reducción — L3 puede no ayudar) |
| `matrix_scale` | — | L2 |
| `attention` | `attention_kernel.py` | L3 |

---

## Métricas del experimento (paper)

Para cada operación × nivel de gramática × LLM, se registra:

**De TritonBench (reutilizados):**
- `call_accuracy` — ¿el kernel pasa su test funcional?
- `execution_accuracy` — ¿ejecuta sin crash?
- `speedup` — tiempo CPU / tiempo GPU

**De KernelForge (nuevos):**
- `grammar_level_requested` / `grammar_level_achieved`
- `fallback_occurred` + `failed_rule` + `fallback_reason`
- `fallback_rate_by_operation` — tabla para el paper
- `parse_time_ms`

Mínimo 50 generaciones por condición para significancia estadística.

---

## Feedback recibido — puntos a mantener

### 1. El fallback automático es válido — mantenerlo
Fortaleza del sistema. L3→L2→L1 garantiza siempre un kernel funcional.

### 2. Las reglas de cada nivel deben ser intercambiables
Los parsers L2 y L3 deben ser configurables, no hardcodeados. Diseñar
`grammar_result.py` y las clases de parser para que las reglas sean
swapeables sin reescribir el parser entero.

### 3. Clarificar cuándo usar L3 vs niveles inferiores
El `GrammarSelector` debe recibir el tipo de operación y elegir el nivel
inicial más adecuado (e.g., dot_product empieza en L2, no L3).

### 4. Incluir tasas de fallback reales en el paper
Tabla explícita: "L3 falló X/10 veces para vector_add, Y/10 para dot_product".
El `MetricsCollector` debe generar esta tabla automáticamente.

---

## Roadmap del semestre

| Semana | Entregable |
|---|---|
| 1–2 | Setup, TritonBench corriendo, Fase 1: Lexer completo con tests |
| 3–4 | Fase 2–3: Parsers L1 y L2. Video 1 |
| 5–6 | Fase 4–5: Parser L3 y FallbackHandler. Video 2 |
| 7–8 | Fase 6–7: Integración TritonBench EVAL + tabla métricas. Video 3 |
| 9–10 | 50 generaciones por condición, análisis estadístico, redacción paper |
| 11–12 | Fase 8: Frontend demo, pulido, defensa oral |

---

## Nota sobre XGrammar vs PLY

XGrammar opera sobre los logits del modelo *durante* la generación (solo con Ollama).
PLY opera *después* de la generación como validador. Son complementarios:

- **PLY (principal):** valida cualquier output de cualquier LLM — es la capa central del paper.
- **XGrammar (opcional):** si se usa Ollama, se puede intentar guiar la generación además
  de validarla. Tratarlo como experimento adicional, no como requisito del sistema.

## Nota sobre TritonBench

TritonBench (`vendor/TritonBench/`) no se modifica. Es la base de evaluación.
KernelForge añade la capa de gramática encima y reutiliza el pipeline de evaluación.
Si TritonBench se actualiza upstream, se puede actualizar el vendor sin tocar nuestro código.
