import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from os import path
    import marimo as mo

    TRITONBENCH_ROOT = "vendor/TritonBench"
    with open(path.join(TRITONBENCH_ROOT, "data/TritonBench_T_simp_alpac_v1.json")) as f:
        T_simple_alpaca = json.loads(f.read())
    with open(path.join(TRITONBENCH_ROOT, "data/TritonBench_T_v1.jsonl")) as f:
        T_json = json.loads(f.read())
    return T_json, T_simple_alpaca, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multi-Model Triton Kernel Comparison

    Compares **Gemma 4B** (Google AI Studio), **GPT-5.5** and **DeepSeek V4 Pro** (Lightning AI)
    on Triton kernel generation from TritonBench-T.

    Results are cached by marimo — API calls only happen once per entry/model pair.
    Final output is saved to `notebooks/results.json`.
    """)
    return


@app.cell
def _(T_json, T_simple_alpaca):
    import re

    T_simple = []
    errors = []

    for entry in T_simple_alpaca:
        match = re.search(r"Wrapper Entry Information: (.+?)\(", entry["instruction"])
        if match is None:
            errors.append({"entry": entry, "error": "no wrapper entry info"})
            continue
        funcname = match.group(1)
        full_entry = list(
            filter(
                lambda el: re.search(fr"^{funcname}\(", el["func_inputs"]) is not None,
                T_json,
            )
        )
        if len(full_entry) != 1:
            errors.append({
                "entry": entry,
                "error": f"multiple or no matches for {funcname}",
                "matches": full_entry,
            })
            continue
        T_simple.append(full_entry[0])

    print(f"{len(errors)} error(s); T_simple = {len(T_simple)} entries")
    return (T_simple,)


@app.function
def make_prompt(entry):
    def clean(value):
        text = str(value or "").strip()
        return "" if text in {"N/A", "None"} else text

    math = clean(entry.get("math"))
    other = clean(entry.get("other"))
    math_block = f"\n## Mathematical Formulation\n{math}" if math else ""
    other_block = f"\n## Additional Information\n{other}" if other else ""

    return f"""\
## Functional Description
{clean(entry.get('description'))}
## Function Signature & Arguments
{clean(entry.get('func_inputs'))}{math_block}
## Reference PyTorch Implementation
```python
{clean(entry.get('torch_code'))}
```{other_block}"""


@app.cell
def _():
    from openai import OpenAI
    from os import environ

    LIGHTNING_API_KEY = environ.get("LIGHTNING_API_KEY")
    GOOGLE_API_KEY = environ.get("GOOGLE_API_KEY")
    assert LIGHTNING_API_KEY is not None, "LIGHTNING_API_KEY not set — see .env.example"
    assert GOOGLE_API_KEY is not None, "GOOGLE_API_KEY not set — see .env.example"

    lightning_client = OpenAI(
        base_url="https://lightning.ai/api/v1",
        api_key=LIGHTNING_API_KEY,
    )
    google_client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=GOOGLE_API_KEY,
    )
    return lightning_client, google_client


@app.cell
def _(mo, lightning_client, google_client):
    SYSTEM_PROMPT = """\
You are an expert GPU kernel engineer specializing in Triton, the open-source DSL \
and compiler for writing high-performance, platform-agnostic GPU kernels.

Your task: given a PyTorch reference implementation and a functional specification, \
write an equivalent, self-contained Triton implementation.

## Triton Constraints
- All kernel functions must use the `@triton.jit` decorator.
- Access global memory exclusively through `tl.load` / `tl.store` with pointer \
arithmetic. Never use Python-level indexing inside a `@triton.jit` kernel.
- Tile computations using `tl.program_id` and `tl.arange`; expose block sizes as \
`tl.constexpr` parameters.
- Prefer native Triton ops (`tl.dot`, `tl.sum`, `tl.sqrt`, `tl.exp`, `tl.maximum`, \
`tl.sigmoid`, etc.) over manual emulation.
- Batch dimensions must be handled via strides or grid flattening, not Python loops.
- Dtypes, broadcast semantics, and numerical defaults must match the reference exactly.

## Output Requirements
- Output only valid Python source code. No Markdown fences, no prose, no tests.
- Begin with the import block: `import triton`, `import triton.language as tl`.
- Define the public wrapper with exactly the name, parameters, and return structure given.
- Include every `@triton.jit` kernel the wrapper depends on.
- No core computation may silently fall back to PyTorch.\
"""

    # To change a model: look up its ID at the provider's model page and update here.
    # Lightning AI models: https://lightning.ai/models?section=allmodels
    # Google AI Studio models: https://aistudio.google.com/models
    MODEL_CONFIGS = {
        "gemma_4b": {
            "client": google_client,
            "model": "gemma-4-4b-it",
            "label": "Gemma 4B (Google AI Studio)",
        },
        "gpt_55": {
            "client": lightning_client,
            "model": "openai/gpt-5.5-2026-04-23",
            "label": "GPT-5.5 (Lightning AI)",
        },
        "deepseek_v4": {
            "client": lightning_client,
            "model": "lightning-ai/deepseek-v4-pro",
            "label": "DeepSeek V4 Pro (Lightning AI)",
        },
    }

    @mo.persistent_cache
    def call_model(entry, model_key: str, max_tokens: int = 2500):
        config = MODEL_CONFIGS[model_key]
        completion = config["client"].chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": make_prompt(entry)},
            ],
            max_tokens=max_tokens,
        )
        return completion

    return MODEL_CONFIGS, call_model


@app.cell
def _(T_simple, call_model, MODEL_CONFIGS):
    import ast

    # How many entries from TritonBench-T to run. Start with 3 to check costs.
    ENTRIES_TO_COMPARE = [0, 1, 2]

    def cleanup(message: str) -> str:
        if "```python" in message:
            message = message.split("```python")[-1]
        message = message.replace("```", "")
        try:
            tree = ast.parse(message)
            imports, funcs = [], []
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports.append(ast.unparse(node))
                elif isinstance(node, ast.FunctionDef):
                    funcs.append(ast.unparse(node))
            return "\n".join(imports) + "\n\n" + "\n\n".join(funcs)
        except Exception as e:
            print(f"Code is not valid Python: {e}")
            return message

    results = []
    for idx in ENTRIES_TO_COMPARE:
        entry = T_simple[idx]
        row = {
            "entry_id": idx,
            "func_inputs": entry.get("func_inputs", ""),
            "description": entry.get("description", ""),
        }
        for model_key in MODEL_CONFIGS:
            response = call_model(entry, model_key)
            row[model_key] = cleanup(response.choices[0].message.content)
        results.append(row)

    print(f"Done: {len(results)} entries × {len(MODEL_CONFIGS)} models")
    return results, cleanup


@app.cell
def _(results):
    import json
    output_path = "notebooks/results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results → {output_path}")
    return


@app.cell
def _(mo, results):
    entry_slider = mo.ui.slider(0, len(results) - 1, label="Entry", value=0)
    entry_slider
    return (entry_slider,)


@app.cell
def _(mo, results, MODEL_CONFIGS, entry_slider):
    row = results[entry_slider.value]
    mo.vstack([
        mo.md(f"**Entry {row['entry_id']}:** `{row['func_inputs']}`"),
        mo.md(f"_{row['description']}_"),
        mo.tabs({
            config["label"]: mo.md(f"```python\n{row[key]}\n```")
            for key, config in MODEL_CONFIGS.items()
        }),
    ])
    return


if __name__ == "__main__":
    app.run()
