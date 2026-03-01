# Large Language Model-Enhanced Relational Operators: Taxonomy, Benchmark, and Analysis

## LROBench

LROBench is a benchmark targeting a novel class of operator-like components: **LLM-Enhanced Relational Operators (LROs)**. These operators share a common pattern: (i) operate over relational data, (ii) take a natural-language requirement (i.e., instruction or condition), (iii) invoke an LLM to make semantic decisions during execution, and (iv) produce relational outputs.


With the development of LLMs, numerous studies integrate LLMs through such operator-like components to enhance relational data processing tasks, e.g., filters with semantic predicates, prompt-based table imputation**, reasoning-driven entity matching, and more challenging*semantic query processing. Unfortunately, from an operator perspective, existing LROs suffer from fragmented definition, various implementation strategies, and inadequate evaluation benchmarks.

To bridge these gaps, LROBench is designed around three core questions:

- **Q1:** How many distinct operating logics underlie current LROs?
- **Q2:** How do different LRO implementations impact their performance?
- **Q3:** How should multi-LRO systems be designed to incorporate operator-level best practices, and do existing systems do so effectively?

LROBench introduces diversity across **operator logics**, **operand granularities**, and **implementation variants**. It establishes a unified taxonomy that categorizes LROs into five mutually exclusive logics—**Select, Match, Impute, Cluster, and Order**—and covers operand granularities including cell, row, column, and table, along with multiple implementation variants (e.g., LLM-ALL, LLM-ONE).

Currently, LROBench features **290 single-LRO queries** in *single-metadata.json* and **60 multi-LRO queries** in *multi_metadata.json*, spanning 27 real-world databases across more than 10 domains. The single-LRO workload provides full coverage of operating logics and operand granularities for fine-grained operator-level evaluation, while the multi-LRO workload provides challenging composed queries stratified by query complexity for end-to-end system evaluation.


## Setup Guide
### 1. Setup Environment
Create a Python 3.10 environment and install the required packages in `requirements.txt`:
```bash
conda create -n LROBench python=3.10
conda activate LROBench
pip install -r requirements.txt
```
### 2. Prepare Databases
Download the [databases.zip](https://drive.google.com/uc?export=download&id=1tB2gMT3h92OtzkWr_rHzfJml00fLjiRY) and directly extract it under the project root:
```bash
unzip databases.zip
```

### 3. Configure API Access
Set your Base URL, API key and model to invoke LLMs:

Linux/MacOS:
```bash
export BASE_URL="https://xxx"
export API_KEY="sk-xxxx"
export MODEL="gpt-5"
```
Windows PowerShell:
```shell
$env:BASE_URL="https://xxx"
$env:API_KEY="sk-xxxx"
$env:MODEL="gpt-5"
```
### 4. Evaluation Quickstarts
We provide scripts for quickly running benchmarks. These include both single-LRO evaluations and multi-LRO evaluations:

Run single-LRO evaluations:
```bash
bash ./scripts/select_eval.sh
bash ./scripts/match_eval.sh
bash ./scripts/impute_eval.sh
bash ./scripts/cluster_eval.sh
bash ./scripts/order_eval.sh
```

Run multi-LRO evaluation with our best-practice baseline:
```bash
bash ./scripts/multi_eval.sh
```

### 5. Modifying Models and Parameters
#### Change Model Settings
Modify LLM configurations (e.g., temperature, max_token) in `src/conf/conf.json`.

#### Customize LROs
Adjust LRO options (operands, implementation, CoT, few-shot examples) directly in the code:
```python
from src.operators.logical import LogicalSelect

op = LogicalSelect(operand_type=OperandType.COLUMN)         # Operand type
result = op.execute(impl_type = ImplType.LLM_ALL,          # Implementation type
                    condition = "The column is related to the SAT test.",
                    df = scores, 
                    example_num = 3,                        # Few-shot examples
                    thinking = True)                        # CoT
```