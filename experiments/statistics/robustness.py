#!/usr/bin/env python3
"""Statistical uncertainty, judge robustness, pipeline sensitivity, and interventions.

Inputs are structured CSWBench evaluation records. All analyses are offline.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import statistics


def records(path, fields):
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8-sig") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not set(fields) <= set(reader.fieldnames):
                raise ValueError(f"{path}: required columns: {', '.join(fields)}")
            rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no records")
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or any(row.get(field) in (None, "") for field in fields):
            raise ValueError(f"{path}: missing required field in record {index}")
    return rows


def number(value, label="score"):
    try:
        x = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: invalid numeric value") from exc
    if not math.isfinite(x):
        raise ValueError(f"{label}: non-finite numeric value")
    return x


def mean(values):
    if not values:
        raise ValueError("cannot average an empty set")
    return statistics.fmean(values)


def percentile(values, fraction):
    if not values:
        raise ValueError("empty bootstrap distribution")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (pos - lo) * (ordered[hi] - ordered[lo])


def interval(values):
    return [percentile(values, .025), percentile(values, .975)]


def average_ranks(values, descending=False):
    indexed = sorted(enumerate(values), key=lambda item: item[1], reverse=descending)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        for k in range(i, j):
            ranks[indexed[k][0]] = (i + 1 + j) / 2
        i = j
    return ranks


def pearson(xs, ys):
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("correlation requires two matched series")
    xbar, ybar = mean(xs), mean(ys)
    xc = [x - xbar for x in xs]
    yc = [y - ybar for y in ys]
    denom = math.sqrt(sum(v * v for v in xc) * sum(v * v for v in yc))
    if denom == 0:
        raise ValueError("correlation is undefined for a constant series")
    return sum(x * y for x, y in zip(xc, yc)) / denom


def spearman(xs, ys):
    return pearson(average_ranks(xs), average_ranks(ys))


def rank_comparison(reference, alternative):
    if set(reference) != set(alternative) or len(reference) < 2:
        raise ValueError("rank comparisons require the same models")
    models = sorted(reference)
    first = average_ranks([reference[m] for m in models], descending=True)
    second = average_ranks([alternative[m] for m in models], descending=True)
    differences = [abs(a - b) for a, b in zip(first, second)]
    return {"spearman": spearman(first, second),
            "mean_absolute_rank_shift": mean(differences),
            "maximum_rank_shift": max(differences)}


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    previous = 0.0
    for rank, i in enumerate(order):
        previous = max(previous, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adjusted[i] = previous
    return adjusted


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def emit(directory, name, result, tables=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (name + ".json")).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    for filename, rows in (tables or {}).items():
        write_csv(directory / filename, rows)


def article_rows(path):
    rows = records(path, ["task", "article_id", "prompt_card_id", "model", "judge", "total"])
    seen = set()
    for row in rows:
        key = (row["task"], row["article_id"], row["judge"])
        if key in seen:
            raise ValueError(f"duplicate article/judge identity: {key}")
        seen.add(key)
        row["total"] = number(row["total"], "article total")
    return rows


def grouped_means(rows, side="dual"):
    grouped = defaultdict(list)
    for row in rows:
        if row["judge"] == side:
            grouped[(row["task"], row["model"])].append(row["total"])
    if not grouped:
        raise ValueError(f"no article scores for judge side {side}")
    result = defaultdict(dict)
    for (task, model), scores in grouped.items():
        result[task][model] = mean(scores)
    return result


def bootstrap_analysis(args):
    rows = [r for r in article_rows(args.scores) if r["judge"] == "dual"]
    metadata = records(args.seeds, ["task", "prompt_card_id", "seed_id"])
    seed_lookup = {}
    for row in metadata:
        key = (row["task"], row["prompt_card_id"])
        if key in seed_lookup and seed_lookup[key] != row["seed_id"]:
            raise ValueError(f"conflicting Seed mapping for {key}")
        seed_lookup[key] = row["seed_id"]
    by_cell = defaultdict(lambda: defaultdict(lambda: [0., 0]))
    for row in rows:
        key = (row["task"], row["prompt_card_id"])
        if key not in seed_lookup:
            raise ValueError(f"missing Seed mapping for {key}")
        acc = by_cell[(row["task"], row["model"])][seed_lookup[key]]
        acc[0] += row["total"]
        acc[1] += 1
    seeds = sorted({seed for cells in by_cell.values() for seed in cells})
    models_by_task = defaultdict(list)
    observed = {}
    for key, cells in sorted(by_cell.items()):
        sums = sum(x[0] for x in cells.values())
        counts = sum(x[1] for x in cells.values())
        observed[key] = sums / counts
        models_by_task[key[0]].append(key[1])
    draws = {key: [] for key in by_cell}
    rng = random.Random(args.seed)
    for _ in range(args.replicates):
        counts = Counter(rng.choices(seeds, k=len(seeds)))
        for key, cells in by_cell.items():
            numerator = sum(values[0] * counts[seed] for seed, values in cells.items())
            denominator = sum(values[1] * counts[seed] for seed, values in cells.items())
            if not denominator:
                raise ValueError(f"bootstrap replicate has no observations for {key}")
            draws[key].append(numerator / denominator)
    ci_rows, pair_rows = [], []
    for (task, model), score in sorted(observed.items()):
        low, high = interval(draws[task, model])
        ci_rows.append(dict(task=task, model=model, mean=score,
                            ci_low=low, ci_high=high))
    for task, models in sorted(models_by_task.items()):
        models = sorted(models)
        current, pvals = [], []
        for i, first in enumerate(models):
            for second in models[i + 1:]:
                differences = [a - b for a, b in zip(draws[task, first], draws[task, second])]
                lo, hi = interval(differences)
                # Two-sided sign-tail p from the paired bootstrap distribution.
                p = min(1.0, 2 * min(
                    (1 + sum(x <= 0 for x in differences)) / (args.replicates + 1),
                    (1 + sum(x >= 0 for x in differences)) / (args.replicates + 1)))
                current.append(dict(task=task, model_a=first, model_b=second,
                                    difference=observed[task, first] - observed[task, second],
                                    ci_low=lo, ci_high=hi, p_unadjusted=p))
                pvals.append(p)
        for row, adjusted in zip(current, holm(pvals)):
            row["p_holm"] = adjusted
        pair_rows.extend(current)
    emit(args.out, "bootstrap", {"seed_clusters": len(seeds), "replicates": args.replicates,
         "random_seed": args.seed, "interval": "95% percentile",
         "multiple_comparisons": "Holm, within setting",
         "model_ci": ci_rows, "model_pairs": pair_rows},
         {"model_ci.csv": ci_rows, "model_pairs.csv": pair_rows})


def kendall_w(rows):
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError("Kendall's W requires articles and raters")
    n, m = len(rows), len(rows[0])
    if any(len(r) != m for r in rows):
        raise ValueError("inconsistent number of human raters")
    rank_matrix = [[0.] * m for _ in rows]
    tie_adjustment = 0
    for j in range(m):
        column = [row[j] for row in rows]
        ranks = average_ranks(column)
        multiplicities = Counter(column).values()
        tie_adjustment += sum(t ** 3 - t for t in multiplicities)
        for i, rank in enumerate(ranks):
            rank_matrix[i][j] = rank
    rank_sums = [sum(row) for row in rank_matrix]
    deviations = sum((v - m * (n + 1) / 2) ** 2 for v in rank_sums)
    denominator = m * m * (n ** 3 - n) - m * tie_adjustment
    if denominator <= 0:
        raise ValueError("Kendall's W undefined for these ratings")
    return 12 * deviations / denominator


def icc_absolute_single(rows):
    # ICC(A,1): two-way random-effects, absolute agreement, single measure.
    n = len(rows)
    k = len(rows[0])
    if n < 2 or k < 2 or any(len(r) != k for r in rows):
        raise ValueError("ICC requires a balanced matrix with >=2 articles and raters")
    grand = mean([v for row in rows for v in row])
    by_article = [mean(row) for row in rows]
    by_rater = [mean([row[j] for row in rows]) for j in range(k)]
    msr = k * sum((x - grand) ** 2 for x in by_article) / (n - 1)
    msc = n * sum((x - grand) ** 2 for x in by_rater) / (k - 1)
    mse = sum((rows[i][j] - by_article[i] - by_rater[j] + grand) ** 2
              for i in range(n) for j in range(k)) / ((n - 1) * (k - 1))
    denominator = msr + (k - 1) * mse + (k / n) * (msc - mse)
    if denominator == 0:
        raise ValueError("ICC undefined for constant ratings")
    return (msr - mse) / denominator


def judge_analysis(args):
    rows = article_rows(args.scores)
    sides = {side: grouped_means(rows, side) for side in ("gpt", "deepseek", "dual")}
    summary = {}
    for task in sorted(sides["dual"]):
        baseline = sides["dual"][task]
        if any(task not in sides[side] for side in ("gpt", "deepseek")):
            raise ValueError(f"missing judge side for {task}")
        gpt, deepseek = sides["gpt"][task], sides["deepseek"][task]
        excluded, family_excluded = {}, {}
        for model in baseline:
            if model not in gpt or model not in deepseek:
                raise ValueError(f"incomplete judge records for {task}/{model}")
            excluded[model] = (deepseek[model] if model == "gpt-5.6-sol" else
                               gpt[model] if model == "deepseek-v4-flash" else baseline[model])
            family_excluded[model] = (deepseek[model] if model in
                    {"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.5"} else baseline[model])
        summary[task] = {
            "gpt_vs_dual": rank_comparison(baseline, gpt),
            "deepseek_vs_dual": rank_comparison(baseline, deepseek),
            "gpt_vs_deepseek": rank_comparison(gpt, deepseek),
            "leave_self_judge_out": rank_comparison(baseline, excluded),
            "gpt_family_excluded": rank_comparison(baseline, family_excluded)}
    if args.human:
        human = records(args.human, ["task", "article_id", "human_1", "human_2",
                                      "human_3", "dual_total"])
        grouped = defaultdict(list)
        for row in human:
            grouped[row["task"]].append(row)
        results = {}
        for task, group in sorted(grouped.items()):
            if len({r["article_id"] for r in group}) != len(group):
                raise ValueError(f"duplicate human ratings in {task}")
            matrix = [[number(r[f"human_{i}"]) for i in range(1, 4)] for r in group]
            dual = [number(r["dual_total"]) for r in group]
            consensus = [mean(values) for values in matrix]
            results[task] = {"articles": len(group), "kendall_w": kendall_w(matrix),
                            "icc_a1": icc_absolute_single(list(zip(consensus, dual))),
                            "spearman": spearman(consensus, dual)}
        summary["human_validation"] = results
    emit(args.out, "judge_robustness", summary)


def pipeline_analysis(args):
    baseline_rows = [r for r in article_rows(args.scores) if r["judge"] == "dual"]
    baseline = grouped_means(baseline_rows)
    source_by_prompt = {}
    for filename in args.cards:
        for card in records(filename, ["prompt_id", "task_type", "source_dataset"]):
            task = {"focused_narrative": "focused", "extended_narrative": "extended"}.get(card["task_type"])
            if not task:
                raise ValueError("unrecognized Prompt Card task type")
            key = task, card["prompt_id"]
            if key in source_by_prompt:
                raise ValueError(f"duplicate Prompt Card: {key}")
            source_by_prompt[key] = card["source_dataset"]
    scores_by_source = defaultdict(lambda: defaultdict(list))
    for row in baseline_rows:
        key = row["task"], row["prompt_card_id"]
        if key not in source_by_prompt:
            raise ValueError(f"missing Prompt Card source for {key}")
        scores_by_source[row["task"], row["model"]][source_by_prompt[key]].append(row["total"])
    balance = {}
    for task, original in sorted(baseline.items()):
        adjusted = {}
        for model in original:
            cells = scores_by_source[task, model]
            if not {"ASCEND", "SwitchLingua"} <= set(cells):
                raise ValueError(f"missing source stratum for {task}/{model}")
            adjusted[model] = (mean(cells["ASCEND"]) + mean(cells["SwitchLingua"])) / 2
        balance[task] = {"rank_agreement": rank_comparison(original, adjusted),
                         "model_totals": {m: {"original": original[m], "balanced": adjusted[m],
                                                "delta": adjusted[m] - original[m]}
                                          for m in sorted(original)}}
    output = {"source_balanced": balance}
    if args.panels:
        panels = records(args.panels, ["task", "constructor", "annotator", "model", "article_id", "total"])
        cells = defaultdict(lambda: defaultdict(list))
        matched = defaultdict(dict)
        for row in panels:
            task, constructor, annotator, model, article = (row[k] for k in
                    ("task", "constructor", "annotator", "model", "article_id"))
            score = number(row["total"])
            pair_key = (task, constructor, model, article)
            if annotator in matched[pair_key]:
                raise ValueError(f"duplicate pipeline annotation: {pair_key}/{annotator}")
            matched[pair_key][annotator] = score
            cells[(task, constructor, annotator)][model].append(score)
        comparisons = []
        for (task, constructor, annotator), models in sorted(cells.items()):
            if task not in baseline:
                raise ValueError(f"unknown task {task}")
            scores = {model: mean(vals) for model, vals in models.items()}
            comparisons.append({"task": task, "constructor": constructor, "annotator": annotator,
                                **rank_comparison(baseline[task], scores),
                                "maximum_absolute_score_shift": max(
                                    abs(scores[m] - baseline[task][m]) for m in scores)})
        # Comparing the annotators only when they score identical articles.
        paired = defaultdict(list)
        for (task, constructor, _, _), scores in matched.items():
            if {"gpt", "qwen"} <= set(scores):
                paired[task, constructor].append(scores["qwen"] - scores["gpt"])
        output["configurations"] = comparisons
        output["within_constructor_annotator_contrast"] = [
            {"task": task, "constructor": constructor,
             "article_pairs": len(diffs), "qwen_minus_gpt": mean(diffs)}
            for (task, constructor), diffs in sorted(paired.items())]
    emit(args.out, "pipeline_sensitivity", output)


def counterfactual_analysis(path, reps, seed):
    rows = records(path, ["event_id", "narrative_id", "group", "judge", "order", "outcome"])
    encoded = {"cs": 1, "tie": 0, "zh": -1}
    by_event = {}
    seen = set()
    entries = []
    for row in rows:
        group = row["group"].upper()
        if group not in ("HIGH", "LOW"):
            raise ValueError(f"unknown counterfactual group {group}")
        outcome = row["outcome"].lower()
        if outcome not in encoded:
            raise ValueError(f"unknown counterfactual outcome {outcome}")
        key = row["event_id"], row["judge"], row["order"]
        if key in seen:
            raise ValueError(f"duplicate counterfactual judgment: {key}")
        seen.add(key)
        tag = row["event_id"], row["narrative_id"], group
        if row["event_id"] in by_event and by_event[row["event_id"]] != tag:
            raise ValueError(f"inconsistent event grouping: {row['event_id']}")
        by_event[row["event_id"]] = tag
        entries.append((tag, encoded[outcome]))
    grouped = defaultdict(list)
    for (_, _, group), value in entries:
        grouped[group].append(value)
    if set(grouped) != {"HIGH", "LOW"}:
        raise ValueError("counterfactual requires HIGH and LOW observations")
    result = {"judgments": len(entries), "events": len(by_event)}
    for group in ("HIGH", "LOW"):
        values = grouped[group]
        result[group.lower()] = {
            "n": len(values), "original_stronger": values.count(1) / len(values),
            "tie": values.count(0) / len(values),
            "monolingual_stronger": values.count(-1) / len(values),
            "directional": mean(values)}
    result["high_minus_low"] = mean(grouped["HIGH"]) - mean(grouped["LOW"])
    rng = random.Random(seed)
    intervals = {}
    for unit in ("event", "narrative"):
        clusters = defaultdict(list)
        for tag, value in entries:
            clusters[tag[0] if unit == "event" else tag[1]].append((tag[2], value))
        names = sorted(clusters)
        draws = []
        for _ in range(reps):
            chosen = rng.choices(names, k=len(names))
            sampled = defaultdict(list)
            for name in chosen:
                for group, value in clusters[name]:
                    sampled[group].append(value)
            if {"HIGH", "LOW"} <= set(sampled):
                draws.append(mean(sampled["HIGH"]) - mean(sampled["LOW"]))
        if len(draws) < max(20, reps // 2):
            raise ValueError("insufficient valid counterfactual cluster resamples")
        intervals[unit] = {"ci95": interval(draws), "valid_replicates": len(draws),
                           "clusters": len(names)}
    result["cluster_bootstrap"] = intervals
    return result


def position_swap_analysis(path, reps, seed):
    rows = records(path, ["narrative_id", "judge", "condition", "pairs",
                          "whole", "multilingual", "general"])
    expected = {("BASELINE", 0)} | {(condition, dose) for condition in ("CS", "ZH")
                                    for dose in (2, 4, 6, 8)}
    grouped = defaultdict(dict)
    for row in rows:
        name, judge = row["narrative_id"], row["judge"]
        condition = row["condition"].upper()
        pair = int(row["pairs"])
        key = condition, pair
        if key not in expected:
            raise ValueError(f"invalid swap condition: {key}")
        if key in grouped[name, judge]:
            raise ValueError(f"duplicate swap condition: {name}/{judge}/{key}")
        grouped[name, judge][key] = {metric: number(row[metric]) for metric in
                                     ("whole", "multilingual", "general")}
    narrative_judges = defaultdict(dict)
    for (narrative, judge), conditions in grouped.items():
        if set(conditions) != expected:
            raise ValueError(f"incomplete swap conditions: {narrative}/{judge}")
        narrative_judges[narrative][judge] = conditions
    values = []
    for narrative, judges in sorted(narrative_judges.items()):
        values.append({key: {metric: mean([conditions[key][metric] for conditions in judges.values()])
                              for metric in ("whole", "multilingual", "general")}
                       for key in expected})
    rng = random.Random(seed)
    outputs = []
    for dose in (2, 4, 6, 8):
        for metric in ("whole", "multilingual", "general"):
            b = [r["BASELINE", 0][metric] for r in values]
            cs = [r["CS", dose][metric] for r in values]
            zh = [r["ZH", dose][metric] for r in values]
            diffs = [right - left for left, right in zip(cs, zh)]
            boot = [mean(rng.choices(diffs, k=len(diffs))) for _ in range(reps)]
            lo, hi = interval(boot)
            outputs.append({"pairs": dose, "dimension": metric, "narratives": len(values),
                            "baseline": mean(b), "cs": mean(cs), "zh": mean(zh),
                            "zh_minus_cs": mean(diffs), "ci_low": lo, "ci_high": hi})
    return outputs


def intervention_analysis(args):
    if not args.counterfactual and not args.swaps:
        raise ValueError("at least one intervention input is required")
    output, tables = {}, {}
    if args.counterfactual:
        output["counterfactual"] = counterfactual_analysis(
            args.counterfactual, args.replicates, args.seed)
    if args.swaps:
        output["position_swaps"] = position_swap_analysis(
            args.swaps, args.replicates, args.seed)
        tables["position_swaps.csv"] = output["position_swaps"]
    emit(args.out, "interventions", output, tables)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    bootstrap = sub.add_parser("bootstrap", help="Seed-clustered intervals and paired model comparisons")
    bootstrap.add_argument("--scores", type=Path, required=True, help="aggregation/article_scores.csv")
    bootstrap.add_argument("--seeds", type=Path, required=True, help="CSV task,prompt_card_id,seed_id")
    judge = sub.add_parser("judges", help="Judge concordance, ablation, and self-exclusion")
    judge.add_argument("--scores", type=Path, required=True)
    judge.add_argument("--human", type=Path, help="CSV task,article_id,human_1,human_2,human_3,dual_total")
    pipeline = sub.add_parser("pipeline", help="Constructor/annotator and source-balance sensitivity")
    pipeline.add_argument("--scores", type=Path, required=True)
    pipeline.add_argument("--cards", type=Path, nargs="+", required=True, help="Published Prompt Card JSONL files")
    pipeline.add_argument("--panels", type=Path, help="CSV task,constructor,annotator,model,article_id,total")
    interventions = sub.add_parser("interventions", help="Counterfactual and position-swap analyses")
    interventions.add_argument("--counterfactual", type=Path,
        help="CSV event_id,narrative_id,group,judge,order,outcome (cs/tie/zh)")
    interventions.add_argument("--swaps", type=Path,
        help="CSV narrative_id,judge,condition,pairs,whole,multilingual,general")
    for command in (bootstrap, judge, pipeline, interventions):
        command.add_argument("--out", type=Path, required=True)
    for command in (bootstrap, interventions):
        command.add_argument("--replicates", type=int, default=10000)
        command.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args(argv)
    if hasattr(args, "replicates") and args.replicates < 20:
        parser.error("--replicates must be >=20")
    {"bootstrap": bootstrap_analysis, "judges": judge_analysis,
     "pipeline": pipeline_analysis, "interventions": intervention_analysis}[args.command](args)


if __name__ == "__main__":
    main()
