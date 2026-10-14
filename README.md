<div align="center">
  <img
    src="assets/cswbench-logo.png"
    alt="CSWBench"
    width="400"
  />
</div>

**CSWBench** is a benchmark for context-sensitive Chinese–English code-switching in creative writing. It provides **807** final narrative prompts: **407 Focused** and **400 Extended**. **CSWEval** evaluates switch events and whole-text organization using the released CSWEval rubric.

## Benchmark

| Setting | Prompts | Target length | Switch content and locations |
| --- | ---: | --- | --- |
| Focused Narrative | 407 | ~700 non-whitespace Unicode code points | Model-selected |
| Extended Narrative | 400 | ~4,000 non-whitespace Unicode code points | Model-selected |

CSWEval combines event-level D1 Licensing, D2 Contextual Appropriateness, and D3 Functional Gain with task-specific whole-text quality, for a total score out of **20**. Two quality judges are GPT-5.6-Sol and DeepSeek-V4-Flash. Event discovery and Extended evidence construction use GPT-5.6-Sol. Scoring details are specified in [evaluation/PROTOCOL.md](evaluation/PROTOCOL.md) and the released scoring prompts and schemas.

## Repository contents

- `data/focused.jsonl` and `data/extended.jsonl`: all 807 final Prompt Cards
- `data/prompt_seed_map.csv`: the 807 Prompt Cards mapped to their 546 Seeds (`task,prompt_card_id,seed_id`), used by the Seed-clustered bootstrap
- `generation/generate.py`: narrative generation using configured model APIs
- `evaluation/evaluate.py`: event discovery, frozen event manifest, Extended evidence construction, two-judge evaluation
- `aggregation/aggregate.py`: event, text and model-level score aggregation
- `configs/`: API configuration examples
- `scripts/recompute_prompt_stats.py`: prompt statistics and integrity checks

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/recompute_prompt_stats.py --check

python generation/generate.py --task focused --config configs/models.example.json --dry-run
python evaluation/evaluate.py --help
python aggregation/aggregate.py --help
```

For narrative **generation**, run `generation/generate.py` with `--task` and `--config`; this CLI has no `--mode live` flag. Remove `--dry-run` to make generation API calls with your authorized endpoint and key environment variables. For **evaluation**, `evaluation/evaluate.py` defaults to offline `prepare`; executing judge APIs requires `--mode live --config my_judges.json` in addition to the task and article arguments. See [the evaluation guide](evaluation/E2E_QUICKSTART.md) for task-specific commands. Generation and evaluation APIs may incur fees.

Prompt, schema, and scoring-parameter filenames use stable descriptive names. Protocol identifiers are stable; exact file and request identities are verified using content hashes (SHA-256).

## Reported evaluation

The reported evaluation contains **4,884 Focused** and **4,800 Extended** narratives across twelve generation models.

## Statistical Analysis

The statistical analysis module implements the core procedures used in CSWBench, including:

- Bootstrap confidence intervals and statistical comparisons.
- Inter-judge agreement and ranking robustness.
- Pipeline sensitivity and source-balanced reweighting.
- Counterfactual and position-swap intervention analyses.

The analysis scripts operate on structured evaluation records and produce statistical summaries without additional model API calls.

The CLI accepts the `article_scores.csv` produced by `aggregation/aggregate.py` and task-specific analysis records:

```bash
python experiments/statistics/robustness.py bootstrap --scores results/article_scores.csv --seeds data/prompt_seed_map.csv --out results/statistics
python experiments/statistics/robustness.py judges --scores results/article_scores.csv --out results/statistics
python experiments/statistics/robustness.py pipeline --scores results/article_scores.csv --cards data/focused.jsonl data/extended.jsonl --out results/statistics
python experiments/statistics/robustness.py interventions --counterfactual counterfactual_judgments.csv --swaps position_swap_scores.csv --out results/statistics
```

For Seed-clustered inference, `--seeds` provides the columns `task,prompt_card_id,seed_id`. The map ships with the data in `data/prompt_seed_map.csv` and matches the reported clustering: 546 Seeds, of which 261 are shared across tasks, 146 Focused-only and 139 Extended-only. Seed identifiers use the CSWBench namespace (`SEED_0001`–`SEED_0546`) and carry no source text or upstream corpus record identifiers; each Prompt Card's source corpus is given by its `source_dataset` field. Optional human-rater input for `judges --human` uses `task,article_id,human_1,human_2,human_3,dual_total`. The pipeline panel input (`--panels`) uses `task,constructor,annotator,model,article_id,total`. Counterfactual judgments use `event_id,narrative_id,group,judge,order,outcome` (outcomes: `cs`, `tie`, `zh`); position-swap records use `narrative_id,judge,condition,pairs,whole,multilingual,general`. Intervention inputs contain the corresponding observed evaluator decisions and scores.

Bootstrap uncertainty uses shared Seed-cluster draws across models and settings, percentile intervals, and within-setting Holm correction for model pairs. The position-swap analysis uses complete-case narrative-matched conditions. Statistical outputs are written as JSON and, where applicable, CSV.

## Licenses and data provenance

Original CSWBench code is licensed under [Apache-2.0](LICENSE). The released CSWBench Prompt Card content is licensed under **CC BY-NC 4.0**. The release does not redistribute original ASCEND or SwitchLingua records and does not grant rights to those source records beyond their applicable terms. See [DATA_LICENSE.md](DATA_LICENSE.md).
