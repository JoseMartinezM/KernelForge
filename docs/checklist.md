# KernelForge — Checklist de pendientes

Cada ítem corresponde a un **issue en GitHub**. Antes de empezar algo, abre el issue,
asígnatelo y pon la referencia en el PR. Cuando termines, márcalo como hecho aquí con
`[x]` y cierra el issue.

---

## Gramática GBNF

- [ ] **Completar `triton.gbnf` con elementos Triton-específicos**
  - Decoradores (`@triton.jit`)
  - Operadores binarios en expresiones (`+`, `*`, `<`, `==`, `//`)
  - `for` loops (necesario para tile loops)
  - Tipo `tl.constexpr` en parámetros (`BLOCK_SIZE: tl.constexpr`)
  - Llamadas calificadas (`tl.load`, `tl.store`, `tl.arange`, `tl.program_id`, `tl.zeros`)

- [ ] **Validar gramática contra kernels reales de TritonBench**
  - La gramática debe aceptar todos los kernels de referencia en `vendor/TritonBench/data/TritonBench_G_v1/`
  - Documentar cualquier caso que no acepta y decidir si es un bug de la gramática o del kernel

---

## XGrammar — Integración con inferencia

- [ ] **Conectar `triton.gbnf` al pipeline de inferencia**
  - Agregar modo `--grammar` / `--no-grammar` a `notebooks/benchmark/llm_inference.py`
  - XGrammar solo aplica a backends que exponen logits (Ollama / llama.cpp); documentar limitación con Lightning y Modal
  - Registrar en el JSONL de resultados si la generación usó gramática o no (`grammar_constrained: bool`)

- [ ] **Prueba de humo: generación constringida con un modelo local**
  - Correr al menos 1 kernel con XGrammar activo y verificar que el output es Python válido

---

## Semantic Checker (chequeos post-generación)

- [ ] **Implementar `notebooks/benchmark/semantic_checker.py`**
  - `tl.load(...)` sin argumento `mask=`
  - `tl.store(...)` sin argumento `mask=`
  - Función kernel sin decorador `@triton.jit`
  - `BLOCK_SIZE` usado sin anotación `tl.constexpr`
  - Ausencia de `tl.program_id` (kernel sin pid)
  - Devolver lista de warnings, no rechazar el kernel

- [ ] **Integrar semantic checker en el pipeline de evaluación**
  - Llamar al checker sobre cada kernel generado antes de pasarlo a TritonBench EVAL
  - Guardar `semantic_warnings: list[str]` en el JSONL de resultados

---

## Métricas y tabla del paper

- [ ] **Columna "con gramática" vs "sin gramática" en la tabla de resultados**
  - Para cada modelo (GPT, DeepSeek, Gemma) correr batch con y sin XGrammar
  - Comparar `call@1`, `exe@1`, speedup

- [ ] **Tabla de semantic warnings más frecuentes por modelo**
  - ¿Qué anti-patrones genera cada LLM con más frecuencia?
  - Agregar columna en `notebooks/evaluate/visualize.py`

- [ ] **Script para generar tabla del paper automáticamente**
  - Input: `runs/tritonbench/*.jsonl`
  - Output: tabla markdown / CSV lista para copiar al paper

---

## Frontend

- [ ] **Setup del proyecto React**
  - Vite + TypeScript + Tailwind CSS
  - Estructura de carpetas: `frontend/src/`

- [ ] **Monaco Editor — visualización del kernel generado**
  - Syntax highlighting Python/Triton
  - Read-only, se actualiza con cada generación

- [ ] **Formulario de generación**
  - Selector de operación (vector_add, attention, etc.)
  - Selector de modelo
  - Toggle XGrammar on/off
  - Botón "Generar"

- [ ] **Panel de resultados**
  - Kernel generado (Monaco)
  - Semantic warnings (lista)
  - Métricas: call@1, exe@1, speedup
  - Badge: "generado con gramática" / "sin gramática"

- [ ] **Gráficas comparativas**
  - Speedup con/sin gramática por operación (Recharts bar chart)
  - Frecuencia de semantic warnings por modelo

---

## Backend API

- [ ] **FastAPI app (`backend/app/main.py`)**
  - `POST /generate` — recibe operación + modelo + opciones, devuelve kernel + métricas
  - `GET /models` — lista modelos disponibles
  - CORS configurado para el frontend

- [ ] **Conectar pipeline completo en el endpoint `/generate`**
  - Inferencia LLM → semantic checker → TritonBench eval → respuesta JSON

---

## Tests

- [ ] **`tests/test_gbnf.py`** — la gramática acepta kernels válidos y rechaza Python genérico sin estructura Triton
- [ ] **`tests/test_semantic_checker.py`** — detecta correctamente cada anti-patrón documentado

---

## Instrucciones para el equipo

1. Elige un ítem de esta lista que nadie tenga asignado.
2. Abre un issue en GitHub con el mismo nombre del ítem.
3. Marca el ítem como `[x]` en este archivo en el mismo PR.
