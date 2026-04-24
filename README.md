# LROBench

**Large Language Model–Enhanced Relational Operators: Taxonomy, Benchmark, and Analysis**

LROBench is a benchmark for **LLM-Enhanced Relational Operators (LROs)**: components that (i) operate over relational data, (ii) take a natural-language requirement (instruction or condition), (iii) invoke an LLM for semantic decisions during execution, and (iv) produce relational outputs.

---

## 📑 Contents

- [🎯 Benchmark overview](#benchmark-overview)
- [🗂️ Data and workloads](#data-and-workloads)
- [⚙️ Setup](#setup)
- [🧪 Running evaluations](#running-evaluations)
- [🔧 Customizing operators and models](#customizing-operators-and-models)
- [🔌 Forward evaluation interface (`forward_eval_interface.py`)](#forward-evaluation-interface-forward_eval_interfacepy)

---

## 🎯 Benchmark overview

As LLMs mature, many systems integrate them through operator-like building blocks—semantic filters, prompt-based table imputation, reasoning-driven entity matching, and harder semantic query workloads. From an operator perspective, LROs today are fragmented in definition, vary in implementation, and lack a unified evaluation harness.

LROBench is organized around three questions:

| | |
|---|---|
| **Q1** | How many distinct operating logics underlie current LROs? |
| **Q2** | How do different LRO implementations affect performance? |
| **Q3** | How should multi-LRO systems incorporate operator-level best practices, and do existing systems do so effectively? |

The benchmark stresses diversity in **operator logic**, **operand granularity** (cell, row, column, table), and **implementation variants** (e.g., LLM-ALL, LLM-ONE). LROs are grouped into five mutually exclusive logics: **Select, Match, Impute, Cluster, and Order**.

---

## 🗂️ Data and workloads

- **Databases:** more than **27** real-world databases across **10+** domains.
- **Single-LRO workload (290 queries):** fine-grained, operator-level evaluation. Metadata files:

  | LRO | Metadata file |
  |-----|-----------------|
  | Select | `select_metadata.json` |
  | Match | `match_metadata.json` |
  | Impute | `impute_metadata.json` |
  | Cluster | `cluster_metadata.json` |
  | Order | `order_metadata.json` |

- **Multi-LRO workload (60 queries):** `multi_metadata.json`—composed queries stratified by complexity for end-to-end system evaluation.

---

## ⚙️ Setup

### 1. Python environment

Use Python 3.10 and install dependencies:

```bash
conda create -n LROBench python=3.10
conda activate LROBench
pip install -r requirements.txt
```

### 2. Databases

Download [databases.zip](https://drive.google.com/uc?export=download&id=1tB2gMT3h92OtzkWr_rHzfJml00fLjiRY) and extract it at the **project root**:

```bash
unzip databases.zip
```

### 3. LLM API

**Linux / macOS:**

```bash
export BASE_URL="https://xxx"
export API_KEY="sk-xxxx"
export MODEL="gpt-5"
```

**Windows PowerShell:**

```powershell
$env:BASE_URL="https://xxx"
$env:API_KEY="sk-xxxx"
$env:MODEL="gpt-5"
```

---

## 🧪 Running evaluations

Scripts are provided for single-LRO and multi-LRO runs.

**Single-LRO:**

```bash
bash ./scripts/select_eval.sh
bash ./scripts/match_eval.sh
bash ./scripts/impute_eval.sh
bash ./scripts/cluster_eval.sh
bash ./scripts/order_eval.sh
```

**Multi-LRO (best-practice baseline):**

```bash
bash ./scripts/multi_eval.sh
```

---

## 🔧 Customizing operators and models

- **Model settings** (e.g., temperature, max tokens): edit `src/conf/conf.json`.
- **Operator behavior** (operands, implementation, CoT, few-shot): construct operators in code, for example:

```python
from src.operators.logical import LogicalSelect
from src.core.enums import OperandType, ImplType

op = LogicalSelect(operand_type=OperandType.COLUMN)
result = op.execute(
    impl_type=ImplType.LLM_ALL,
    condition="The column is related to the SAT test.",
    df=scores,
    example_num=3,
    thinking=True,
)
```

---

## 🔌 Forward evaluation interface (`forward_eval_interface.py`)

At the repository root, **`forward_eval_interface.py`** is a **forward-facing evaluation scaffold**: it loads queries from each task’s metadata JSON, you plug in **your** system via `your_system_run`, then it scores predictions with the same **`score`** conventions and metrics as the existing `eval/*.py` pipelines, producing per-query records and task summaries.

**You must implement `your_system_run` with real logic before evaluation is meaningful.** The file ships with an empty stub: it always returns `None`, so every query is recorded as *no prediction* and skipped for scoring (`has_prediction: false`). The CLI will still exit successfully, but you will **not** get usable metrics until you replace the stub—e.g., load tables from `./databases/<db_used>/`, run your LROs or LLM pipeline, and return a prediction in the shape expected by **`score`** for that `query.task`.

### Three pieces

| Piece | Role |
|-------|------|
| **`build_query(task, metadata_path=None)`** | Loads all queries for a task into a list of `BenchQuery` objects (default paths match `DEFAULT_METADATA`). |
| **`your_system_run(query: BenchQuery) -> Any`** | **Required implementation:** add your logic here (read `./databases/<db_used>/`, call LLMs or LROs, etc.) and return a non-`None` prediction when possible; the default stub returns `None` only so the module imports and the CLI runs—**not** as a runnable evaluator. |
| **`score(task, ground_truth, prediction, attributes)`** | Task-aware scoring aligned with the `eval` scripts (e.g., precision/recall/F1 for select/match, hr@k and tau for order, ARI/NMI for cluster, etc.). |

Each **`BenchQuery`** holds: `task`, `query_id`, `question`, `db_used`, `ground_truth`, and `attributes` (extra fields from metadata).

### Default metadata

Tasks `select`, `match`, `impute`, `cluster`, `order`, and `multi_lro` map to the corresponding `*_metadata.json` files at the project root, as in `DEFAULT_METADATA`.

### Prediction shapes (for `score`)

Match the module docstring and `eval/*.py` conventions. In short:

- **select / match:** `list[list[Any]]` or a `DataFrame` row-aligned with ground truth.
- **order:** `list[Any]` (ordered Top-k).
- **cluster:** `list[int]` with the same length as `ground_truth`.
- **impute:** when metadata `ground_truth` is a list, a row-aligned `list[Any]`.
- **multi_lro:** exact table match; scalars as a 1×1 table, nested lists as rows×columns, a flat list as n×1.

### Command line

```bash
# All tasks
python forward_eval_interface.py --tasks all

# Comma-separated subset
python forward_eval_interface.py --tasks select,match

# Write full JSON report
python forward_eval_interface.py --tasks all --out report.json
```

After **`your_system_run`** returns real predictions (not `None`), use the commands above to batch over queries under a single contract, print summaries to the terminal, and optionally emit per-query `scores` plus task-level `summary` (including mean metrics) with `--out`.
