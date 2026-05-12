# KernelForge

KernelForge generates optimized GPU kernels automatically. You describe the operation you need, and the system produces a working [Triton](https://github.com/openai/triton) kernel — validated and ready to use.

**The problem it solves:** Writing fast GPU kernels requires specialized expertise. Engineers end up with kernels that work but run 10–30× slower than they should. KernelForge automates the optimization.

---

## How it works

You select an operation (e.g. vector addition, matrix scaling) and an optimization level. The system prompts an LLM — constrained by a formal grammar — to generate the kernel, then validates it by compiling and running it on the GPU. If the requested optimization level fails, it automatically falls back to a simpler one.

**Three optimization levels:**
- **L1 — Basic:** valid Triton syntax that compiles and runs
- **L2 — Coalesced:** adds memory coalescing for better memory bandwidth
- **L3 — Tiled:** adds tiling and shared memory for maximum performance

---

## Stack

- **Backend:** FastAPI, Triton, PyTorch, XGrammar, Ollama / Claude API
- **Frontend:** React, TypeScript, Monaco Editor, Recharts

---

## Quick start

```bash
cp .env.example .env   # set your LLM provider (Ollama or Claude API)
docker-compose up --build
```

Frontend → `http://localhost:5173`  
Backend → `http://localhost:8000`

---

## Supported operations (v1)

`vector_add` · `vector_mul` · `dot_product` · `matrix_scale`
