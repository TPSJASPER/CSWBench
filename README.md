<div align="center">
  <img
    src="assets/cswbench-logo.png"
    alt="CSWBench"
    width="350"
  />

  <p>
    <strong>Evaluating Context-Sensitive Code-Switching<br />in Creative Writing</strong>
  </p>

  <p>
    <a href="DATASET_CARD.md"><img src="https://img.shields.io/badge/Prompts-807-557C99?style=flat-square" alt="807 prompts" /></a>
    <a href="DATASET_CARD.md"><img src="https://img.shields.io/badge/Languages-Chinese%E2%80%93English-66758C?style=flat-square" alt="Languages: Chinese–English" /></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/Code-Apache--2.0-496B85?style=flat-square" alt="Code license: Apache-2.0" /></a>
    <a href="DATA_LICENSE.md"><img src="https://img.shields.io/badge/Data-CC%20BY--NC%204.0-73718D?style=flat-square" alt="Data license: CC BY-NC 4.0" /></a>
  </p>

  <p>
    <a href="#overview">Overview</a> &nbsp;·&nbsp;
    <a href="DATASET_CARD.md">Dataset</a> &nbsp;·&nbsp;
    <a href="#quick-start">Quick Start</a> &nbsp;·&nbsp;
    <a href="evaluation/PROTOCOL.md">Evaluation Protocol</a> &nbsp;·&nbsp;
    <a href="#statistical-analysis">Statistical Analysis</a>
  </p>
</div>

## Overview

**CSWBench** is a benchmark for context-sensitive Chinese–English code-switching in creative writing. It provides **807 final narrative prompts** across two settings: **Focused Narrative** and **Extended Narrative**. **CSWEval** evaluates switch events and task-specific whole-text quality using the released rubric.

| Prompt Cards | Narrative Settings | Evaluated Models | Reported Narratives |
| :---: | :---: | :---: | :---: |
| **807** | **2** | **12** | **9,684** |

The reported evaluation contains **4,884 Focused** and **4,800 Extended** narratives across twelve generation models.

## Benchmark and Evaluation

### Narrative Settings

| Setting | Prompts | Target Length¹ | Switch Content and Locations |
| --- | ---: | ---: | --- |
| **Focused Narrative** | 407 | ~700 | Model-selected |
| **Extended Narrative** | 400 | ~4,000 | Model-selected |

¹ Target lengths are measured in non-whitespace Unicode code points.

Browse the [Focused prompts](data/focused.jsonl), [Extended prompts](data/extended.jsonl), and [dataset card](DATASET_CARD.md).

### CSWEval Framework

| Level | Dimension |
| --- | --- |
| Event | **D1 · Licensing** |
| Event | **D2 · Contextual Appropriateness** |
| Event | **D3 · Functional Gain** |
| Whole text | **Task-specific quality** |

CSWEval combines these dimensions into a total score out of **20**. The two quality judges are **GPT-5.6-Sol** and **DeepSeek-V4-Flash**. Event discovery and Extended evidence construction use GPT-5.6-Sol.

See the [evaluation protocol](evaluation/PROTOCOL.md), [scoring prompts](evaluation/prompts/), and [schemas](evaluation/schemas/) for scoring details.

## Quick Start

### 1. Install Dependencies

Run the following commands from the repository root:

```bash
python -m pip install -r requirements.txt
```

### 2. Verify the Prompt Data

```bash
python scripts/recompute_prompt_stats.py --check
```

### 3. Generate and Evaluate Narratives

Preview a generation run:

```bash
python generation/generate.py \
  --task focused \
  --config configs/models.example.json \
  --dry-run
```

For narrative generation, remove `--dry-run` to make API calls with your authorized endpoint and key environment variables. The generation CLI uses `--task` and `--config`; it has no `--mode live` flag.

Inspect the evaluation and aggregation options:

```bash
python evaluation/evaluate.py --help
python aggregation/aggregate.py --help
```

Evaluation defaults to offline `prepare`. To execute judge APIs, add `--mode live --config my_judges.json` alongside the task and article arguments. Follow the [end-to-end evaluation guide](evaluation/E2E_QUICKSTART.md) for task-specific commands.

> **API usage:** Live generation and evaluation calls may incur fees.

## Repository Guide

| Module | Purpose | Entry |
| --- | --- | --- |
| **Prompt data** | All 807 final Prompt Cards | [Focused](data/focused.jsonl) · [Extended](data/extended.jsonl) |
| **Seed map** | Map the 807 Prompt Cards to 546 Seeds for Seed-clustered bootstrap | [`prompt_seed_map.csv`](data/prompt_seed_map.csv) |
| **Generation** | Generate narratives using configured model APIs | [`generate.py`](generation/generate.py) |
| **Evaluation** | Discover events, freeze manifests, construct Extended evidence, and run two-judge evaluation | [`evaluate.py`](evaluation/evaluate.py) |
| **Aggregation** | Aggregate event, text, and model-level scores | [`aggregate.py`](aggregation/aggregate.py) |
| **Statistical analysis** | Compute uncertainty, agreement, robustness, and intervention summaries | [`robustness.py`](experiments/statistics/robustness.py) |
| **Configuration** | API configuration examples | [`configs/`](configs/) |
| **Data checks** | Recompute prompt statistics and check data integrity | [`recompute_prompt_stats.py`](scripts/recompute_prompt_stats.py) |

Prompt, schema, and scoring-parameter filenames use stable descriptive names. Protocol identifiers are stable; exact file and request identities are verified using content hashes (SHA-256).

## Statistical Analysis

The analysis scripts operate on structured evaluation records and produce statistical summaries without additional model API calls.

| Procedure | Analysis |
| --- | --- |
| `bootstrap` | Bootstrap confidence intervals and statistical comparisons |
| `judges` | Inter-judge agreement and ranking robustness |
| `pipeline` | Pipeline sensitivity and source-balanced reweighting |
| `interventions` | Counterfactual and position-swap intervention analyses |

The CLI accepts the `article_scores.csv` produced by [`aggregation/aggregate.py`](aggregation/aggregate.py) and task-specific analysis records.

<details>
<summary><strong>Run the analysis commands</strong></summary>

**Bootstrap confidence intervals and model comparisons**

```bash
python experiments/statistics/robustness.py bootstrap \
  --scores results/article_scores.csv \
  --seeds data/prompt_seed_map.csv \
  --out results/statistics
```

**Judge agreement and ranking robustness**

```bash
python experiments/statistics/robustness.py judges \
  --scores results/article_scores.csv \
  --out results/statistics
```

**Pipeline sensitivity and source-balanced reweighting**

```bash
python experiments/statistics/robustness.py pipeline \
  --scores results/article_scores.csv \
  --cards data/focused.jsonl data/extended.jsonl \
  --out results/statistics
```

**Counterfactual and position-swap interventions**

```bash
python experiments/statistics/robustness.py interventions \
  --counterfactual counterfactual_judgments.csv \
  --swaps position_swap_scores.csv \
  --out results/statistics
```

</details>

<details>
<summary><strong>Input formats and Seed mapping</strong></summary>

| Input | Required Columns |
| --- | --- |
| Seed map (`--seeds`) | `task,prompt_card_id,seed_id` |
| Human ratings (`judges --human`, optional) | `task,article_id,human_1,human_2,human_3,dual_total` |
| Pipeline panel (`--panels`) | `task,constructor,annotator,model,article_id,total` |
| Counterfactual judgments | `event_id,narrative_id,group,judge,order,outcome` |
| Position-swap records | `narrative_id,judge,condition,pairs,whole,multilingual,general` |

The Seed map ships with the data in [`data/prompt_seed_map.csv`](data/prompt_seed_map.csv) and matches the reported clustering: **546 Seeds**, of which **261** are shared across tasks, **146** are Focused-only, and **139** are Extended-only.

Seed identifiers use the CSWBench namespace (`SEED_0001`–`SEED_0546`) and carry no source text or upstream corpus record identifiers. Each Prompt Card's source corpus is given by its `source_dataset` field.

Counterfactual outcomes are `cs`, `tie`, or `zh`. Intervention inputs contain the corresponding observed evaluator decisions and scores.

</details>

<details>
<summary><strong>Statistical details and outputs</strong></summary>

Bootstrap uncertainty uses shared Seed-cluster draws across models and settings, percentile intervals, and within-setting Holm correction for model pairs. The position-swap analysis uses complete-case narrative-matched conditions.

Statistical outputs are written as JSON and, where applicable, CSV.

</details>

## License and Data Provenance

| Component | License |
| --- | --- |
| Original CSWBench code | [Apache-2.0](LICENSE) |
| Released CSWBench Prompt Card content | [CC BY-NC 4.0](DATA_LICENSE.md) |

The release does not redistribute original ASCEND or SwitchLingua records and does not grant rights to those source records beyond their applicable terms. See [`DATA_LICENSE.md`](DATA_LICENSE.md).
