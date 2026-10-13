# Aggregation input

`aggregation/aggregate.py` consumes one JSONL record per article and judge side (`gpt` and `deepseek`). It computes per-article, per-model and overall task scores.

Every event includes two D1 gates, salience, and either Focused D2/D3 subdimension scores or Extended D2/D3 holistic scores. Whole-text scores use L1–L5 for Focused and M-F1, M-F2A, M-F2B, M-F3.1–M-F3.4 for Extended.

D1 is five times the proportion of events passing both gates. D2/D3 average applicable (non-NA/non-IE) event scores, weighted by salience S1=1, S2=2, S3=3. Whole is the mean of the task's whole-text dimensions. The four dimension scores sum to /20. Two judge sides are averaged at article level before model-level aggregation.

```bash
python aggregation/aggregate.py judgements.jsonl --out results/
```

The output includes `article_scores.csv`, `model_scores.csv`, and `overall_scores.csv`. No cross-setting composite is defined.
