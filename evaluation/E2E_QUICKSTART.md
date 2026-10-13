# CSWEval quick start

From the repository root:

```bash
python -m pip install -r requirements.txt
```

Copy `configs/judges.example.json` into your own configuration and supply authorized API URLs plus API-key environment variables.

## Focused

```bash
python evaluation/evaluate.py \
  --task focused --prompt-id PCV4_238 \
  --article story.txt --article-id example-focused-001 \
  --generation-model gpt-5.6-sol \
  --output outputs/example-focused-001
```

This default **prepare** mode builds the discovery input without network calls. To execute event discovery, quality judging, and aggregation, add `--mode live --config my_judges.json`.

The runner uses the approximate 700-code-point target specified by the Focused Prompt.

## Extended

```bash
python evaluation/evaluate.py \
  --task extended --prompt-id LGPC_0004 \
  --article story.txt --article-id example-extended-001 \
  --generation-model gpt-5.6-sol --blind-id LJ-B0001 \
  --output outputs/example-extended-001
```

The Extended A/B character registry is extracted from the final Prompt Card by default. Optionally pass `--registry my_registry.json` to supply an explicit registry using source-grounded information. The `--blind-id` identifies the document in the scoring protocol.

With `--mode live --config my_judges.json`, the runner performs model discovery, freezes the event manifest, constructs Sol narrative evidence, uses the same frozen input for the two quality judges, and aggregates their scores. The runner validates stage schemas and refuses incompatible scene IDs or evidence citations.

## Generation and aggregation

```bash
python generation/generate.py --task focused --config configs/models.example.json --dry-run
python aggregation/aggregate.py --help
python scripts/recompute_prompt_stats.py --check
```

To actually generate narratives, run `python generation/generate.py --task focused --config my_models.json` (without `--dry-run`). The generation CLI does not accept `--mode live`; that flag belongs only to `evaluation/evaluate.py`, which defaults to `--mode prepare` and requires `--mode live --config my_judges.json` for API judging.

Generation and evaluation calls may incur provider charges. The reported Qwen runs used provider-default thinking. The published reproduction configuration explicitly sets `enable_thinking: true` for all four Qwen models.
