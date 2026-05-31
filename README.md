# Reproduction of MultiLifeQA: A Multidimensional Lifestyle QA Benchmark

A reproduction study of *MultiLifeQA: A Multidimensional Lifestyle Question Answering Benchmark for Comprehensive Health Reasoning with LLMs*.

**Author:** Shengqi Gui;
**Date:** May 2026;
**Model Used:** `gpt-4o-mini` (OpenAI API);

---

## 1. Paper Overview

MultiLifeQA is a large-scale QA benchmark designed to evaluate LLMs' ability to reason over multidimensional lifestyle health data. The dataset contains 22,573 questions spanning four lifestyle domains — diet, physical activity, sleep, and emotion — sourced from AI4FoodDB, a one-month multimodal lifestyle database of 100 participants.

### 1.1 Core Design

The benchmark organises questions into five categories of increasing reasoning difficulty:

| Category | Description | Example |
|---|---|---|
| Fact Query (FQ) | Retrieve specific data points | "What was user X's step count on date Y?" |
| Aggregated Statistics (AS) | Compute over temporal windows | "What is the average deep sleep duration over one week?" |
| Numeric Comparison (NC) | Compare values across records | "Which day had the highest activity level?" |
| Conditional Query (CQ) | Filter by thresholds | "How many days had stress > 50 and sleep < 6h?" |
| Trend Analysis (TA) | Detect temporal patterns | "Did heart rate decrease for 3 consecutive days?" |

Questions further vary along two axes: **dimension complexity** (single-domain → cross-2-domain → cross-4-domain) and **user scope** (single-user → multi-user).

### 1.2 Two Evaluation Settings

The paper proposes two complementary evaluation paradigms:

**Context Prompting (CP):** The raw data table is embedded directly into the prompt. The LLM must parse the table, locate relevant rows/columns, and compute the answer — acting as both a database engine and a reasoning engine. This setting only supports single-user queries due to context window limits.

**Database-augmented Prompting (DP):** The LLM receives only the database schema, generates a SQL query, which is then executed on a MySQL database. The query results are fed back to the LLM for final reasoning. This decouples retrieval (handled by the database) from reasoning (handled by the LLM), and supports both single-user and multi-user queries. Conceptually, this is analogous to Retrieval-Augmented Generation (RAG), but with structured SQL queries instead of semantic vector retrieval.

### 1.3 Evaluation Metrics (DP)

- **Accuracy**: Final answer correctness with type-specific tolerances
- **SQL Validity (VA)**: Proportion of generated SQL that executes without errors
- **Execution Accuracy (EX)**: Proportion where SQL returns the correct information
- **Acc/EX**: Answer accuracy conditioned on correct SQL execution

---

## 2. Reproduction Setup

### 2.1 Environment

| Component | Specification |
|---|---|
| Hardware | MacBook Air (Apple Silicon) |
| OS | macOS |
| Python | 3.10+ (conda environment) |
| MySQL | 8.x (installed via Homebrew) |
| LLM | gpt-4o-mini via OpenAI API |

### 2.2 Installation Steps

```bash
# 1. Install Homebrew (if not present)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install MySQL
brew install mysql
brew services start mysql
mysql_secure_installation

# 3. Install Git LFS and clone the repository
brew install git-lfs
git lfs install 

# 4. Create database
mysql -u root -p -e "CREATE DATABASE MultilifeQA;"

# 5. Install Python dependencies
pip install pymysql openai tqdm pandas numpy python-dotenv

# 6. Set environment variables
export MYSQL_USER="root"
export MYSQL_PASSWORD="<your_password>"
export MYSQL_PWD="<your_password>"
```

### 2.3 Data Loading

```bash
# Download AI4FoodDB data into ./data/
# Download FoodNExtDB into ./data/FoodNExtDB/

# Load data into MySQL
python load_mysql_db.py
python load_food_db.py

# Verify
mysql -u root -p MultilifeQA -e "SHOW TABLES;"
```

### 2.4 Subset Sampling Strategy

Due to API budget constraints, I evaluated on representative subsets rather than the full dataset. Two sampling scripts were created (`sample_subset.py` for CP, `sample_subset_sql.py` for DP) that perform stratified per-file sampling across all question types and dimension settings.

| Setting | Full Dataset | Subset Size | Sampling Method |
|---|---|---|---|
| CP | 13,452 (single-user only) | 775 | 5/file + 10 extra for easy types in single/ |
| DP | 22,573 (single + multi-user) | 730 | 3/file + 7 extra for easy types in single/ |

For single-dimension FQ/CQ/TA questions, additional samples were drawn to ensure coverage of cases where the model is expected to succeed, providing a balanced view of both strengths and weaknesses.

Note: Following the original paper's methodology, CP and DP subsets were independently sampled. The paper itself does not enforce identical question sets across the two settings, as they serve different evaluation scopes (CP: single-user only; DP: single-user + multi-user).

---

## 3. Results

### 3.1 Overall Accuracy

| Setting | gpt-4o-mini (this work) | GPT-4o (paper) |
|---|---|---|
| CP Accuracy | 58.97% (n=775) | 57.02% (n=13,452) |
| DP Accuracy | 51.78% (n=730) | 34.71% (n=22,573) |
| DP SQL Success Rate | 87.67% | 63.85% (VA) |
| DP Acc/EX | 59.06% | 95.65% |

### 3.2 Results by Question Type

| Type | CP | DP | Paper GPT-4o (CP) | Paper GPT-4o (DP) |
|---|---|---|---|---|
| FQ | **77.44%** | 62.17% | 69.78% | 42.50% |
| CQ | 58.97% | **60.87%** | 71.45% | 58.47% |
| TA | **69.23%** | 42.22% | 79.08% | 14.89% |
| NC | **48.42%** | 36.67% | 49.58% | 31.31% |
| AS | 10.53% | **26.67%** | 15.63% | 26.45% |

### 3.3 Results by Dimension Complexity (CP)

| Dimension | Accuracy |
|---|---|
| Single (non-joint) | 65.64% |
| Joint (cross-dimensional) | 42.67% |

### 3.4 DP: SQL Execution by Question Type

| Type | SQL Success Rate |
|---|---|
| CQ | 98.26% |
| AS | 87.78% |
| FQ | 84.78% |
| NC | 80.00% |
| TA | 75.56% |

---

## 4. Analysis and Discussion

### 4.1 Key Finding 1: AS Benefits from Database Offloading

Aggregated Statistics questions show a reversal in the CP vs DP comparison: AS accuracy is higher under DP (26.67%) than CP (10.53%). This is because AS questions require arithmetic over long data sequences (sums, averages over weeks). Under CP, the model must perform these calculations by parsing a text-formatted table — a task where LLMs are known to struggle. Under DP, the computation is offloaded to MySQL via SQL aggregation functions (`SUM`, `AVG`), and the model only needs to generate the correct query. This pattern is consistent with the paper's GPT-4o results (AS: CP 15.63% vs DP 26.45%) and directly supports the paper's argument for database-augmented approaches.

### 4.2 Key Finding 2: SQL Generation vs Reasoning Bottleneck

gpt-4o-mini achieves a high SQL success rate (87.67%) but a relatively low Acc/EX (59.06%), compared to GPT-4o's 95.65%. This indicates that:

- gpt-4o-mini is competent at translating natural language questions into executable SQL queries.
- However, once the correct data is retrieved, gpt-4o-mini struggles to reason over the results to produce the correct final answer — a task where GPT-4o excels.

This suggests that for the DP pipeline, the reasoning step (not SQL generation) is the primary bottleneck for smaller models, whereas the paper identified SQL generation as the main bottleneck for most open-source models.

### 4.3 Key Finding 3: Cross-Dimensional Degradation is Robust

The accuracy drop from single-dimension to cross-dimension tasks is consistently observed:

- CP: 65.64% (single) → 42.67% (joint), a 23-point drop
- DP: ~63% (single) → ~35% (joint), a 28-point drop

This aligns with the paper's central finding (Figure 5, right) and confirms that cross-dimensional health reasoning remains a fundamental challenge for current LLMs, independent of model version or evaluation scale.

### 4.4 Limitations

- **Subset-based evaluation**: Results are based on ~750 examples per setting rather than the full dataset. While the sampling covers all question types and dimensions, absolute accuracy numbers may differ from full-dataset evaluation.
- **Single model**: Only gpt-4o-mini was evaluated, whereas the paper benchmarks 11 models. Comparative analysis across model families is not possible.
- **Independent sampling**: CP and DP subsets contain different specific questions, though the sampling distributions are aligned by design.
- **Distribution bias**: The per-file uniform sampling does not preserve the original question distribution across folders. As a result, the reported overall accuracy is not a weighted estimate of full-dataset performance, but rather reflects equal-weight performance across all category–domain combinations. This is consistent with the paper's per-category analysis approach (Tables 8 and 10) but may differ from a proportionally sampled estimate.

---

## 5. Repository Structure

```
.
├── README.md                          # This file
├── sample_subset.py                   # CP subset sampling script
├── sample_subset_sql.py               # DP subset sampling script
├── eval/                              # CP evaluation outputs
│   └── gpt-4o-mini/
│       ├── all_outputs.jsonl
│       └── summary.json
├── eval_sql/                          # DP evaluation outputs
│   └── gpt-4o-mini/
│       ├── all_outputs.jsonl
│       └── summary.json
└── gen_data_processed/                # Dataset (not included, see setup)
    ├── simple/                        # CP format
    ├── sql/                           # DP format
    ├── simple_subset/                 # Sampled CP subset
    └── sql_subset/                    # Sampled DP subset
```

---

## 6. How to Reproduce

```bash
# CP evaluation
python sample_subset.py \
    --data-root ./gen_data_processed/simple \
    --out-root ./gen_data_processed/simple_subset \
    --per-file 5 --easy-boost 10 --seed 42

python eval_simple.py \
    --data-root ./gen_data_processed/simple_subset \
    --eval-root ./eval \
    --model gpt-4o-mini \
    --max-new-tokens 32 \
    --api-key "<YOUR_KEY>"

# DP evaluation
python sample_subset_sql.py \
    --data-root ./gen_data_processed/sql \
    --out-root ./gen_data_processed/sql_subset \
    --per-file 3 --easy-boost 7 --seed 42

python eval_sql.py \
    --data-root ./gen_data_processed/sql_subset \
    --eval-root ./eval_sql \
    --model gpt-4o-mini \
    --sql-max-new-tokens 480 \
    --ans-max-new-tokens 48 \
    --api-key "<YOUR_KEY>"
```

---

## References

- MultiLifeQA (under review at ICLR 2026): [Original Paper](https://openreview.net/forum?id=5882BZyFdS), [Anonymous Repository](https://anonymous.4open.science/r/MultilifeQA-05D2)
- AI4FoodDB: Romero-Tapiador et al., 2023. *Database: The Journal of Biological Databases and Curation.*
