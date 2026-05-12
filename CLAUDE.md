# CLAUDE.md — KernelForge GPU Kernel Generator

## Visión del proyecto

Sistema que usa LLMs guiados por gramáticas formales (EBNF + XGrammar) para generar
kernels GPU en Triton con tres niveles de optimización garantizados. El usuario describe
la operación que necesita; el sistema genera, valida y entrega el kernel correcto y
eficiente automáticamente, con fallback inteligente entre niveles si la optimización
más alta falla.

**Problema que resuelve:** Los ingenieros de ML/IA escriben kernels GPU que funcionan
pero son 10–30x más lentos de lo que podrían ser. Optimizarlos manualmente requiere
expertos escasos y caros. Este sistema democratiza la generación de kernels óptimos.

---

## Stack tecnológico

### Backend
- **Python 3.11+**
- **FastAPI** — API REST y WebSockets para streaming de generación
- **XGrammar** — constrained decoding, aplica gramáticas EBNF sobre el LLM
- **Triton 2.x** — compilación y ejecución de kernels GPU generados
- **PyTorch** — manejo de tensores y lanzamiento de kernels
- **Ollama** — LLMs open source locales (CodeLlama, DeepSeek-Coder)
- **Anthropic SDK** — Claude API como alternativa al LLM local
- **Pydantic v2** — validación de schemas y modelos de datos
- **pytest** — testing del pipeline de generación y validación

### Frontend
- **React 18 + TypeScript** — UI principal
- **Vite** — bundler y dev server
- **Tailwind CSS** — estilos utilitarios
- **Monaco Editor** — editor de código con syntax highlighting para Triton/Python
- **Recharts** — visualización de métricas de rendimiento (speedup, compilación)
- **React Query** — manejo de estado async y caché de generaciones

### Infraestructura
- **Docker + Docker Compose** — contenedores para back y front
- **Google Colab** — ejecución GPU durante desarrollo (T4 gratuita)
- **NVIDIA CUDA 11.8+** — requerido para Triton en GPU local

---

## Arquitectura del sistema

```
Usuario
   │
   ▼
┌─────────────────────────────────────────┐
│  Frontend React                          │
│  - Selector de operación                 │
│  - Selector de nivel de optimización     │
│  - Editor Monaco (código generado)       │
│  - Dashboard de métricas                 │
└───────────────┬─────────────────────────┘
                │ HTTP / WebSocket
                ▼
┌─────────────────────────────────────────┐
│  FastAPI Backend                         │
│                                          │
│  POST /generate                          │
│  GET  /metrics/{job_id}                  │
│  WS   /stream/{job_id}                   │
│                                          │
│  ┌─────────────────────────────────┐    │
│  │  GenerationPipeline              │    │
│  │                                  │    │
│  │  1. PromptBuilder                │    │
│  │  2. GrammarSelector (L1/L2/L3)   │    │
│  │  3. LLMRunner (Ollama | Claude)  │    │
│  │     + XGrammar constrained       │    │
│  │  4. KernelValidator              │    │
│  │     - compilación Triton         │    │
│  │     - ejecución en GPU           │    │
│  │     - corrección vs CPU          │    │
│  │  5. FallbackHandler              │    │
│  │     L3 falla → intenta L2 → L1   │    │
│  │  6. MetricsCollector             │    │
│  └─────────────────────────────────┘    │
└─────────────────────────────────────────┘
                │
                ▼
        GPU (Colab T4 / local)
        Triton compila PTX → SASS
        Kernel corre en paralelo
```

### Flujo de generación con fallback

```
Prompt → GrammarL3 → LLM → Kernel
                              │
                    ┌─────────┴─────────┐
                    │   Validación       │
                    └─────────┬─────────┘
                         pasa │   falla
                              │      │
                              ▼      ▼
                           Éxito   GrammarL2 → LLM → Kernel
                                                        │
                                              ┌─────────┴──────┐
                                              │   Validación    │
                                              └─────────┬──────┘
                                                   pasa │  falla
                                                        │      │
                                                        ▼      ▼
                                                     Éxito   GrammarL1 → ...
```

---

## Estructura de directorios

```
grammarforge/
│
├── CLAUDE.md                          # Este archivo
├── README.md
├── docker-compose.yml
├── .env.example
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── pyproject.toml
│   │
│   ├── app/
│   │   ├── main.py                    # FastAPI app, routers, CORS
│   │   ├── config.py                  # Settings (LLM provider, GPU config)
│   │   │
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── generate.py        # POST /generate — pipeline completo
│   │   │   │   ├── metrics.py         # GET /metrics/{job_id}
│   │   │   │   ├── stream.py          # WS /stream — generación en tiempo real
│   │   │   │   └── health.py          # GET /health
│   │   │   └── schemas/
│   │   │       ├── generation.py      # GenerateRequest, GenerateResponse
│   │   │       └── metrics.py         # MetricsResponse, BenchmarkResult
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py            # GenerationPipeline — orquesta todo
│   │   │   ├── prompt_builder.py      # Construye prompts por operación
│   │   │   ├── fallback_handler.py    # Lógica de degradación L3→L2→L1
│   │   │   └── metrics_collector.py  # Recolecta y persiste métricas
│   │   │
│   │   ├── grammars/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Clase base Grammar
│   │   │   ├── level1_basic.py        # Gramática sintaxis correcta
│   │   │   ├── level2_coalesced.py    # + memory coalescing, BLOCK_SIZE 2^n
│   │   │   ├── level3_tiled.py        # + tiling, shared memory
│   │   │   └── grammar_selector.py    # Elige gramática por nivel y operación
│   │   │
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── base_runner.py         # Interfaz abstracta LLMRunner
│   │   │   ├── ollama_runner.py       # Implementación Ollama + XGrammar
│   │   │   ├── claude_runner.py       # Implementación Claude API
│   │   │   └── xgrammar_wrapper.py    # Integración XGrammar con runners
│   │   │
│   │   └── validation/
│   │       ├── __init__.py
│   │       ├── validator.py           # Orquesta las 3 capas de validación
│   │       ├── compile_check.py       # Capa 1: Triton compile()
│   │       ├── execution_check.py     # Capa 2: lanzar kernel, verificar no crash
│   │       └── correctness_check.py   # Capa 3: resultado GPU == CPU
│   │
│   └── tests/
│       ├── test_grammars.py
│       ├── test_pipeline.py
│       ├── test_validation.py
│       └── test_api.py
│
├── frontend/
│   ├── Dockerfile
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   │
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       │
│       ├── components/
│       │   ├── GeneratorPanel/
│       │   │   ├── OperationSelector.tsx   # Dropdown: suma, matmul, reducción
│       │   │   ├── LevelSelector.tsx       # L1 / L2 / L3 con descripción
│       │   │   ├── LLMSelector.tsx         # Ollama | Claude API
│       │   │   └── GenerateButton.tsx
│       │   │
│       │   ├── CodeViewer/
│       │   │   ├── KernelEditor.tsx        # Monaco con el kernel generado
│       │   │   ├── GenerationSteps.tsx     # Pasos del pipeline en tiempo real
│       │   │   └── FallbackBadge.tsx       # Muestra si hubo fallback L3→L2
│       │   │
│       │   ├── MetricsDashboard/
│       │   │   ├── SpeedupChart.tsx        # Barra: GPU vs CPU speedup
│       │   │   ├── CompilationRate.tsx     # % compilaciones exitosas por nivel
│       │   │   ├── LevelComparison.tsx     # L1 vs L2 vs L3 side by side
│       │   │   └── FallbackStats.tsx       # Cuántas veces cayó cada nivel
│       │   │
│       │   └── shared/
│       │       ├── Badge.tsx
│       │       ├── LoadingSpinner.tsx
│       │       └── ErrorCard.tsx
│       │
│       ├── hooks/
│       │   ├── useGeneration.ts            # React Query para POST /generate
│       │   ├── useMetrics.ts               # Polling de métricas
│       │   └── useStream.ts                # WebSocket hook para streaming
│       │
│       ├── services/
│       │   └── api.ts                      # Axios client, endpoints tipados
│       │
│       └── types/
│           ├── generation.ts
│           └── metrics.ts
│
├── notebooks/
│   ├── 01_grammar_exploration.ipynb       # Diseño y prueba de gramáticas
│   ├── 02_kernel_benchmarks.ipynb         # Comparativa L1/L2/L3 en Colab
│   └── 03_experiments_paper.ipynb         # 50 generaciones para el paper
│
└── paper/
    ├── main.tex
    └── figures/
```

---

## Gramáticas — diseño por niveles

### Nivel 1 — Básico (sintaxis válida)
```ebnf
kernel      ::= imports decorator funcdef
imports     ::= "import triton" newline "import triton.language as tl" newline
decorator   ::= "@triton.jit" newline
funcdef     ::= "def " identifier "(" params "):" newline body
params      ::= param ("," param)*
param       ::= identifier (":" type_hint)?
body        ::= pid_stmt offsets_stmt mask_stmt load_stmts compute_stmt store_stmt
pid_stmt    ::= identifier " = tl.program_id(axis=0)" newline
offsets_stmt::= identifier " = " identifier " * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)" newline
mask_stmt   ::= identifier " = " identifier " < " identifier newline
load_stmt   ::= identifier " = tl.load(" expr ", mask=" identifier ")" newline
store_stmt  ::= "tl.store(" expr ", " identifier ", mask=" identifier ")" newline
```

### Nivel 2 — Coalesced (memory coalescing forzado)
Extiende Nivel 1 y agrega:
```ebnf
block_size_decl ::= "BLOCK_SIZE: tl.constexpr"
block_size_val  ::= "64" | "128" | "256" | "512" | "1024"
offsets_pattern ::= "pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)"
```
La gramática solo permite este patrón de offsets exacto, garantizando
que hilos adyacentes accedan a posiciones contiguas de memoria.

### Nivel 3 — Tiled (shared memory + tiling)
Extiende Nivel 2 y agrega:
```ebnf
shared_alloc ::= identifier " = tl.zeros([BLOCK_SIZE], dtype=" dtype ")" newline
tile_loop    ::= "for " identifier " in range(0, " identifier ", BLOCK_SIZE):" newline tile_body
tile_body    ::= shared_load tile_compute barrier
barrier      ::= "tl.debug_barrier()" newline
```

---

## Variables de entorno (.env)

```bash
# LLM Provider: "ollama" | "claude"
LLM_PROVIDER=ollama

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=codellama:13b

# Claude API (solo si LLM_PROVIDER=claude)
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-sonnet-4-20250514

# GPU
CUDA_VISIBLE_DEVICES=0
TRITON_CACHE_DIR=/tmp/triton_cache

# App
BACKEND_PORT=8000
FRONTEND_PORT=5173
MAX_GENERATION_ATTEMPTS=3
```

---

## Operaciones GPU soportadas (v1)

| Operación | Descripción | Paralelismo |
|---|---|---|
| `vector_add` | Suma elemento a elemento | 1 hilo por elemento |
| `vector_mul` | Multiplicación elemento a elemento | 1 hilo por elemento |
| `dot_product` | Producto punto de dos vectores | Reducción paralela |
| `matrix_scale` | Escalar una matriz por constante | 1 hilo por elemento |

Estas cuatro operaciones son suficientes para demostrar los tres niveles
de optimización y generar resultados publicables en el paper.

---

## Métricas del experimento (paper)

Para cada operación × nivel de gramática × LLM, se registra:

- `compilation_success` (bool) — ¿compiló sin error?
- `execution_success` (bool) — ¿corrió sin crash?
- `correctness` (bool) — ¿resultado GPU == CPU con tolerancia 1e-5?
- `speedup` (float) — tiempo CPU / tiempo GPU
- `generation_attempts` (int) — intentos hasta éxito (fallback)
- `tokens_generated` (int) — longitud del kernel
- `grammar_level_achieved` (int) — nivel final logrado (puede bajar por fallback)

Mínimo 50 generaciones por condición para significancia estadística.

---

## User Stories

### US-01 — Ingeniero de ML en empresa de IA
**Como** ingeniero de ML que necesita un kernel GPU para una operación
personalizada de atención en mi modelo,
**quiero** describir la operación en lenguaje natural y recibir un kernel
Triton optimizado listo para producción,
**para** no tener que esperar semanas a que un experto en GPU lo escriba
manualmente ni pagar $200k/año por ese experto.

**Criterios de aceptación:**
- El sistema genera el kernel en menos de 60 segundos
- El kernel compila sin modificaciones
- El speedup sobre CPU es al menos 10x para N > 1M elementos
- El nivel de optimización alcanzado se muestra claramente en la UI

---

### US-02 — Equipo de MLOps con pipeline automatizado
**Como** ingeniero de MLOps que mantiene un pipeline de CI/CD para
despliegue de modelos,
**quiero** que el sistema intente generar el kernel más optimizado posible
y, si falla, degrade automáticamente al siguiente nivel sin intervención,
**para** que el pipeline nunca se rompa por un kernel que no compiló y
siempre tengamos al menos un kernel funcional desplegado.

**Criterios de aceptación:**
- Si el nivel 3 falla, el sistema intenta nivel 2 automáticamente
- Si nivel 2 falla, intenta nivel 1
- El nivel final alcanzado queda registrado en el log con razón del fallback
- El tiempo total incluyendo reintentos no supera 3 minutos
- La API retorna el mejor kernel logrado junto con metadata del fallback

---

### US-03 — Investigador científico sin experiencia en GPU
**Como** investigador de bioinformática que necesita procesar millones de
secuencias genómicas,
**quiero** generar un kernel GPU para mi operación de comparación sin
necesitar aprender Triton ni CUDA,
**para** reducir el tiempo de mis experimentos de 45 minutos a segundos
y poder iterar más rápido en mi investigación.

**Criterios de aceptación:**
- El usuario puede describir la operación en términos de su dominio
- El sistema genera y valida el kernel sin que el usuario vea código
- Se muestra el speedup real medido en sus datos
- El kernel generado es exportable como archivo .py listo para usar

---

### US-04 — Startup de IA optimizando costos en la nube
**Como** CTO de una startup que gasta $30k/mes en GPUs en AWS,
**quiero** reemplazar nuestros kernels escritos manualmente con kernels
generados al nivel 3 de optimización,
**para** reducir el tiempo de cómputo y bajar nuestra factura de GPU
sin contratar más ingenieros especializados.

**Criterios de aceptación:**
- El sistema puede procesar un batch de operaciones distintas
- Las métricas de speedup se reportan por operación
- Los kernels generados son compatibles con PyTorch y el stack existente
- El ROI estimado (horas GPU ahorradas) se muestra en el dashboard

---

### US-05 — Profesor o investigador evaluando el sistema
**Como** evaluador académico del proyecto,
**quiero** ver una comparativa clara entre kernels generados sin gramática
versus con gramática en niveles 1, 2 y 3,
**para** verificar que la gramática tiene un impacto real y medible en
la calidad y rendimiento del código generado.

**Criterios de aceptación:**
- El dashboard muestra las 4 condiciones (sin gramática, L1, L2, L3)
- Las métricas incluyen tasa de compilación, corrección y speedup
- Los resultados son reproducibles con el mismo seed
- El código de los kernels generados es inspeccionable en la UI

---

## Comandos de desarrollo

```bash
# Levantar todo con Docker
docker-compose up --build

# Solo backend en desarrollo
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Solo frontend en desarrollo
cd frontend
npm install
npm run dev

# Correr experimentos del paper en Colab
# Abrir notebooks/02_kernel_benchmarks.ipynb en Google Colab
# Activar GPU: Runtime → Change runtime type → T4 GPU

# Tests
cd backend
pytest tests/ -v
```

---

## Nota sobre XGrammar y Claude API

XGrammar opera sobre el modelo de forma local, controlando la distribución
de probabilidad de los tokens en tiempo real. Por esto:

- **Con Ollama (recomendado para desarrollo):** XGrammar funciona nativamente
  porque tienes acceso al modelo local.
- **Con Claude API:** La API no expone control de logits, por lo que XGrammar
  no puede aplicarse directamente. En este caso, Claude se usa como baseline
  de comparación (generación sin gramática) o se implementa post-processing
  de validación estructural sobre la salida.

Para el experimento principal del paper, usar Ollama + CodeLlama con XGrammar.
Claude API sirve como baseline de "LLM sin restricciones".

---

## Feedback recibido — puntos a tener en cuenta

Estos puntos surgieron de una revisión externa del proyecto y deben guiar decisiones de diseño e implementación:

### 1. El fallback automático es válido — mantenerlo
El mecanismo de auto-fallback L3→L2→L1 fue valorado positivamente. Es una fortaleza del sistema y debe seguir siendo la estrategia central para garantizar siempre un kernel funcional.

### 2. Las reglas de cada nivel deben ser intercambiables
Las reglas/patrones que se aplican en cada nivel actualmente están fijas (e.g., el patrón de coalescing exacto en L2, tiling en L3). Esto es un problema: puede haber otras buenas prácticas igualmente válidas que la gramática bloquea.

**Acción pendiente:** Diseñar los módulos de gramática (`level2_coalesced.py`, `level3_tiled.py`) de forma que las reglas sean configurables o intercambiables, no hardcodeadas. La clase base `Grammar` en `grammars/base.py` debe facilitar esto. El objetivo es poder swapear o extender las reglas de un nivel sin reescribir la gramática entera.

### 3. Clarificar cuándo usar L3 vs niveles inferiores
No siempre tiene sentido intentar L3 primero. Los patrones de L2 y L3 (coalescing, tiling, shared memory) son beneficiosos en operaciones memory-bound con acceso regular, pero pueden ser innecesarios o contraproducentes en kernels simples o con patrones de acceso irregulares.

**Acción pendiente:** Documentar (y opcionalmente implementar en el sistema) criterios explícitos para cuándo recomendar L3 vs L2 vs L1 como punto de partida. Ejemplo: operaciones de reducción o con dependencias entre hilos pueden no beneficiarse del tiling de L3. El `GrammarSelector` debería poder recibir hints sobre el tipo de operación para elegir el nivel inicial más adecuado.

### 4. Incluir resultados de testing reales en la documentación y el paper
El evaluador señaló que falta evidencia empírica del comportamiento del fallback. Se espera incluir datos concretos como:

> "En nuestras pruebas, L3 falló y cayó a L2 aproximadamente 2 de cada 10 veces para `vector_add`. Para `dot_product`, la tasa de fallback fue mayor (~4/10)."

**Acción pendiente:** Al correr los experimentos del paper (`notebooks/02_kernel_benchmarks.ipynb`), registrar y reportar explícitamente las tasas de fallback por operación y nivel. Incluir esta tabla en el paper y en el dashboard de métricas (`FallbackStats.tsx`).

---

## Roadmap del semestre

| Semana | Entregable |
|---|---|
| 1–2 | Setup del proyecto, Triton funcionando en Colab, primera gramática L1 |
| 3–4 | Pipeline básico: LLM → gramática → kernel → compilación. Video 1 |
| 5–6 | Gramáticas L2 y L3, fallback handler, métricas básicas. Video 2 |
| 7–8 | Frontend completo, dashboard de métricas, experimentos del paper. Video 3 |
| 9–10 | 50 generaciones por condición, análisis estadístico, redacción paper |
| 11–12 | Pulido, demo final, preparación defensa oral |