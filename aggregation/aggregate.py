#!/usr/bin/env python3
"""Aggregate CSWEval judgements into dimension scores and /20 totals.

This implements the aggregation rule that produced the reported CSWBench results:

  D1    = 5 x (share of scorable events whose participant-level AND world-level
          licensing checks both PASS)
  D2    = salience-weighted mean over events of the event's contextual-appropriateness
          score. Focused narratives are rated on sub-dimensions per event and the
          sub-dimensions are averaged within the event first. Extended narratives are
          rated with a single overall score per event.
  D3    = same construction as D2, using the functional-gain sub-dimensions / overall.
  Whole = unweighted mean of the whole-text dimensions (L1-L5 for Focused,
          M-F1...M-F3.4 for Extended).
  total = D1 + D2 + D3 + Whole, out of 20.

Salience weights are S1 = 1, S2 = 2, S3 = 3 and apply to D2 and D3 only. D1 is
unweighted. An event whose sub-dimension is NA or IE contributes no value for that
sub-dimension and is never treated as zero; an event whose salience is NA or IE carries no
weight and is dropped from that dimension's weighted mean.

Two judges score every article under the published protocol. Their article-level dimension
scores are averaged before model-level aggregation.

Input
-----
One JSONL file, one record per (article, judge). See DATA_CONTRACT.md.

    python aggregate.py judgements.jsonl --out results/

Outputs
-------
  article_scores.csv      one row per (article, judge) plus one `dual` row per article
  model_scores.csv        per task, per model, per judge side (including `dual`)
  overall_scores.csv      per model, the dual-judge /20 total on each task
  summary.json            counts and the rule actually applied
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from statistics import mean

SALIENCE_WEIGHTS = {"S1": 1, "S2": 2, "S3": 3}
NOT_A_NUMBER = {None, "", "NA", "IE"}
DIMENSIONS = ("D1", "D2", "D3", "Whole", "total")
JUDGE_SIDES = ("gpt", "deepseek")
FOCUSED_SUBDIMENSIONS = {"D2": {f"D2.{i}" for i in range(1, 9)}, "D3": {f"D3.{i}" for i in range(1, 6)}}
WHOLE_FOCUSED = ("L1", "L2", "L3", "L4", "L5")
WHOLE_EXTENDED = ("M-F1", "M-F2A", "M-F2B", "M-F3.1", "M-F3.2", "M-F3.3", "M-F3.4")
WHOLE_FIELDS = {"focused": WHOLE_FOCUSED, "extended": WHOLE_EXTENDED}


def numeric(value):
    """Return a float score, or None for NA/IE/missing. Reject out-of-range values."""
    if value in NOT_A_NUMBER:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite score: {value!r}")
    if not 1 <= result <= 5:
        raise ValueError(f"score outside 1-5: {value!r}")
    return result


def available_mean(values):
    valid = [value for value in values if value is not None]
    return mean(valid) if valid else None


def weighted_mean(pairs):
    valid = [(value, weight) for value, weight in pairs if value is not None and weight is not None]
    if not valid:
        return None
    return sum(value * weight for value, weight in valid) / sum(weight for _, weight in valid)


def licensing_score(events):
    """D1: 5 x share of events passing both the participant and the world check."""
    pairs = [(event.get("d1_role"), event.get("d1_world")) for event in events]
    scorable = [(role, world) for role, world in pairs if role in ("PASS", "FAIL") and world in ("PASS", "FAIL")]
    if not scorable:
        return None, 0
    passed = sum(role == "PASS" and world == "PASS" for role, world in scorable)
    return 5 * passed / len(scorable), len(scorable)


def dimension_score(events, task, dimension):
    """Salience-weighted D2 or D3, event-first for Focused and direct for Extended."""
    pairs = []
    for event in events:
        if task == "focused":
            sub = event.get(f"{dimension.lower()}_subdimensions") or {}
            unknown = set(sub) - FOCUSED_SUBDIMENSIONS[dimension]
            if unknown:
                raise ValueError(f"unknown {dimension} labels: {sorted(unknown)}")
            value = available_mean([numeric(score) for score in sub.values()])
        else:
            value = numeric(event.get(f"{dimension.lower()}_overall"))
        weight = SALIENCE_WEIGHTS.get(event.get("salience"))
        pairs.append((value, weight))
    return weighted_mean(pairs)


def whole_score(record):
    fields = WHOLE_FIELDS[record["task"]]
    scores = [numeric(record.get("whole", {}).get(field)) for field in fields]
    missing = [field for field, score in zip(fields, scores) if score is None]
    if missing:
        raise ValueError(f"{record['article_id']}: whole-text dimensions missing: {missing}")
    return mean(scores)


def score_record(record):
    events = record["events"]
    if not events:
        raise ValueError(f"{record['article_id']}: no events")
    d1, valid = licensing_score(events)
    if d1 is None:
        raise ValueError(f"{record['article_id']}: no event has a scorable D1 pair")
    scored = {
        "task": record["task"],
        "article_id": record["article_id"],
        "prompt_card_id": record.get("prompt_card_id", ""),
        "model": record["model"],
        "judge": record["judge"],
        "events": len(events),
        "valid_D1": valid,
        "D1": d1,
        "D2": dimension_score(events, record["task"], "D2"),
        "D3": dimension_score(events, record["task"], "D3"),
    }
    scored["Whole"] = whole_score(record)
    for dim in ("D2", "D3"):
        if scored[dim] is None:
            raise ValueError(f"{record['article_id']}: no eligible {dim} scores with valid salience")
    scored["total"] = sum(scored[dimension] for dimension in DIMENSIONS[:4])
    return scored


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate(records: list[dict]) -> dict[str, list[dict]]:
    per_judge = [score_record(record) for record in records]
    by_article = defaultdict(list)
    for row in per_judge:
        by_article[(row["task"], row["article_id"])].append(row)
    dual_rows = []
    for (task, article_id), rows in by_article.items():
        sides = {row["judge"] for row in rows}
        if len(rows) != 2 or sides != set(JUDGE_SIDES):
            raise ValueError(f"{article_id}: expected exactly one gpt and one deepseek record, got {len(rows)} {sorted(sides)}")
        if rows[0]["model"] != rows[1]["model"] or rows[0]["prompt_card_id"] != rows[1]["prompt_card_id"]:
            raise ValueError(f"{article_id}: paired judges have inconsistent article metadata")
        averaged = dict(rows[0], judge="dual")
        for dimension in DIMENSIONS:
            averaged[dimension] = mean(row[dimension] for row in rows)
        dual_rows.append(averaged)
    article_rows = per_judge + dual_rows

    model_rows = []
    for task in sorted({row["task"] for row in article_rows}):
        models = sorted({row["model"] for row in article_rows if row["task"] == task})
        for side in (*JUDGE_SIDES, "dual"):
            for model in models:
                selected = [row for row in article_rows
                            if row["task"] == task and row["judge"] == side and row["model"] == model]
                if not selected:
                    continue
                row = {"task": task, "model": model, "judge": side, "articles": len(selected)}
                for dimension in DIMENSIONS:
                    row[dimension] = mean(item[dimension] for item in selected)
                model_rows.append(row)

    overall_rows = []
    dual_lookup = {(row["task"], row["model"]): row for row in model_rows if row["judge"] == "dual"}
    for model in sorted({row["model"] for row in model_rows}):
        if ("focused", model) in dual_lookup and ("extended", model) in dual_lookup:
            overall_rows.append({
                "model": model,
                "focused_total_20": dual_lookup["focused", model]["total"],
                "extended_total_20": dual_lookup["extended", model]["total"],
            })
    return {"article": article_rows, "model": model_rows, "overall": overall_rows}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = ["task", "model", "judge", "articles", *DIMENSIONS]
    if "article_id" in rows[0]:
        fields = ["task", "article_id", "prompt_card_id", "model", "judge", "events", "valid_D1", *DIMENSIONS]
    elif "focused_total_20" in rows[0]:
        fields = ["model", "focused_total_20", "extended_total_20"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("judgements", type=Path, help="JSONL of judge records (see DATA_CONTRACT.md)")
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args(argv)

    records = load(args.judgements)
    tables = aggregate(records)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "article_scores.csv", tables["article"])
    write_csv(args.out / "model_scores.csv", tables["model"])
    write_csv(args.out / "overall_scores.csv", tables["overall"])
    summary = {
        "judge_records": len(records),
        "articles": len({(row["task"], row["article_id"]) for row in tables["article"]}),
        "models": sorted({row["model"] for row in tables["model"]}),
        "rule": "D1 = 5 x pass share; D2/D3 salience-weighted (event-first for focused); "
                "Whole = mean of whole-text dimensions; total = /20; dual = article-level mean of judges",
        "salience_weights": SALIENCE_WEIGHTS,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
