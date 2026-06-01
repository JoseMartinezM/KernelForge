# Ledger Schema

KernelForge maintains two separate ledgers with distinct responsibilities:

## Inference Ledger (llm_inference.py output)

**Purpose:** Record the immutable generation record from LLM inference.

**Location:** `notebooks/data/*.jsonl` (configured by model provider)

**Responsibility:** Capture what the LLM generated for each prompt.

**Key Fields:**
- `entry_file`: Path to the original TritonBench entry
- `entry_index`: Index in the TritonBench dataset
- `model`: Model identifier
- `model_label`: Human-readable model label
- `content`: The generated Triton kernel code (raw LLM output)
- `prompt`: The input prompt sent to the LLM
- Other metadata from the inference run (timestamp, temperature, etc.)

**What it does NOT store:**
- `call@1`, `exe@1`: Evaluation metrics (belongs in eval ledger)
- `semantic_warnings`: Semantic checks (belongs in eval ledger)
- `execution_time`: Performance measurement (belongs in eval ledger)
- `speedup`: Derived metric (visualization calculates from raw data)

**Schema Stability:** Changes to this ledger require careful consideration because old inference records cannot be regenerated.

---

## Evaluation Ledger (eval_results.json)

**Purpose:** Record the evaluation results and semantic analysis of generated kernels.

**Location:** `notebooks/results/eval_results.json`

**Responsibility:** Evaluate generated kernels against TritonBench reference implementations and extract semantic warnings.

**Key Fields:**
- `file`: Path to the TritonBench entry
- `model`: Model identifier
- `model_label`: Human-readable model label
- `call@1`: Boolean - whether generated kernel's control flow matches reference
- `exe@1`: Boolean - whether generated kernel execution matches reference
- `mismatches`: List of evaluation errors or mismatches
- `semantic_warnings`: List of semantic checker warnings (e.g., missing `@triton.jit`, missing mask on vectorized load)
- `execution_time`: Float in seconds or null - raw execution time for generated kernel (not speedup)

**Derived Metrics (calculated during visualization):**
- Speedup = reference_execution_time / generated_execution_time

**Schema Stability:** This ledger can be regenerated from the inference ledger by re-running evaluation. Changes to the schema are less critical.

---

## Workflow

1. **Inference Phase** (`llm_inference.py`):
   - Query LLM with prompt
   - Record generated code in inference ledger
   - **Stop:** Do not evaluate here

2. **Evaluation Phase** (`notebooks/evaluate/run_eval.py`):
   - Load inference records
   - For each record:
     - Run semantic checker (`check_kernel()`) → `semantic_warnings`
     - Execute evaluation (`evaluate_entry()`) → `call@1`, `exe@1`, `mismatches`, `execution_time`
   - Write results to eval ledger

3. **Visualization Phase** (`notebooks/evaluate/visualize.py`):
   - Load eval ledger
   - Calculate derived metrics (e.g., speedup from `execution_time`)
   - Render charts and tables

---

## Separation of Concerns

This separation ensures:
- **Immutability of generation record**: Inference ledger is never modified
- **Reproducibility**: Evaluation can be re-run on existing inference records without regenerating them
- **Clear responsibility**: Each ledger has a single, well-defined purpose
- **Efficient storage**: Evaluation metadata is not duplicated in the immutable inference record
