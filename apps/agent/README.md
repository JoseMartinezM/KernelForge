# KernelForge Agent

Agente de Pi que genera kernels de Triton, los valida en una GPU de Modal, y guarda los resultados.

## Qué hace

1. Le pides un kernel → llama al LLM para generarlo
2. Lo valida en Modal T4 → revisa si corre y si los valores son correctos
3. Si falla → lee el error y lo intenta de nuevo solo
4. Cuando pasa → guarda el resultado en `runs/agent/`

## Lo que necesitas

**Una sola vez:**
```powershell
# Autenticarte en Modal (para la validación con GPU)
uv run modal token new

# Instalar dependencias de Node
npm install
```

**Cada sesión:**
```powershell
$env:ANTHROPIC_API_KEY = "..."   # cerebro del agente (console.anthropic.com)

# Una de estas según el modelo que quieras usar:
$env:LIGHTNING_API_KEY = "..."   # para DeepSeek, GPT, Gemma 31B
# o
$env:MODAL_API_KEY = "..."       # para Gemma E4B (+ el servidor vLLM corriendo)
$env:MODAL_API_SECRET = "..."
```

## Cómo correrlo

```powershell
npx pi
```

Ejemplos de lo que le puedes pedir:

```
Generate and validate a tanh kernel using lightning-ai/deepseek-v4-pro
Generate and validate softmax.py using google/gemma-4-E4B-it
```

## Modelos disponibles

| Modelo | Provider | Key necesaria |
|---|---|---|
| `google/gemma-4-E4B-it` | Modal | `MODAL_API_KEY` + servidor vLLM corriendo |
| `lightning-ai/deepseek-v4-pro` | Lightning | `LIGHTNING_API_KEY` |
| `openai/gpt-5.4-2026-03-05` | Lightning | `LIGHTNING_API_KEY` |
| `lightning-ai/gemma-4-31B-it` | Lightning | `LIGHTNING_API_KEY` |

## Resultados

Cada kernel procesado se guarda en `runs/agent/{modelo}.jsonl` con:
- El código generado
- `call@1` / `exe@1` (si corrió y si los valores son correctos)
- `semantic_warnings` (anti-patrones de Triton detectados)
- `attempts` (cuántas veces reintentó el agente)
