#!/usr/bin/env python3
"""Recompute current Prompt Card hashes, lengths and pairwise 3-gram Jaccard,
together with the released Seed map's record count and SHA-256.

Run after any modification to data/:
    python scripts/recompute_prompt_stats.py
To verify committed values without writing:
    python scripts/recompute_prompt_stats.py --check

These are checks over the final released Prompt Cards and Seed map.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.85


def grams(text: str) -> set[str]:
    return {text[i:i + 3] for i in range(len(text) - 2)}


def describe(name: str, rows: list[dict], raw_bytes: bytes) -> dict:
    sets = [grams(row["prompt_text"]) for row in rows]
    top = (-1.0, None)
    exceeded = 0
    pairs = 0
    for index, first in enumerate(sets):
        for other_index in range(index + 1, len(sets)):
            second = sets[other_index]
            intersection = len(first & second)
            union = len(first) + len(second) - intersection
            score = intersection / union if union else 1.0
            pairs += 1
            if score > top[0]:
                top = (score, [rows[index]["prompt_id"], rows[other_index]["prompt_id"]])
            if score >= THRESHOLD:
                exceeded += 1
    lengths = [len(row["prompt_text"]) for row in rows]
    return {
        "records": len(rows),
        "pairs_compared": pairs,
        "maximum_3gram_jaccard": top[0],
        "highest_pair": top[1],
        "pairs_at_or_above_0_85": exceeded,
        "min_prompt_chars": min(lengths),
        "max_prompt_chars": max(lengths),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "size_bytes": len(raw_bytes),
    }


def describe_map(cards: dict) -> dict:
    raw = (ROOT / "data" / "prompt_seed_map.csv").read_bytes()
    lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    if lines[0] != "task,prompt_card_id,seed_id":
        raise SystemExit("unexpected Seed map header: " + lines[0])
    mapped = {}
    seed_tasks: dict[str, set] = {}
    for line in lines[1:]:
        task, prompt_id, seed_id = line.split(",")
        mapped[task, prompt_id] = seed_id
        seed_tasks.setdefault(seed_id, set()).add(task)
    published = {(task, row["prompt_id"]) for task, rows in cards.items() for row in rows}
    return {
        "records": len(mapped),
        "distinct_seeds": len(seed_tasks),
        "shared_across_tasks": sum(1 for tasks in seed_tasks.values() if len(tasks) > 1),
        "focused_only": sum(1 for tasks in seed_tasks.values() if tasks == {"focused"}),
        "extended_only": sum(1 for tasks in seed_tasks.values() if tasks == {"extended"}),
        "covers_published_prompts": set(mapped) == published,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
    }


def calculate() -> dict:
    output: dict = {}
    cards: dict = {}
    for task in ("focused", "extended"):
        raw = (ROOT / "data" / (task + ".jsonl")).read_bytes()
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        cards[task] = rows
        output[task] = describe(task, rows, raw)
        output[task]["sources"] = dict(Counter(row["source_dataset"] for row in rows))
    output["seed_map"] = describe_map(cards)
    return output


def update_files(result: dict) -> None:
    stats_path = ROOT / "DATASET_STATS.json"
    manifest_path = ROOT / "data" / "MANIFEST.json"
    redundancy_path = ROOT / "PROMPT_REDUNDANCY.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    redundancy = json.loads(redundancy_path.read_text(encoding="utf-8"))
    for task in ("focused", "extended"):
        current = result[task]
        stats[task]["file_sha256"] = current["sha256"]
        stats[task]["prompt_char_length_min"] = current["min_prompt_chars"]
        stats[task]["prompt_char_length_max"] = current["max_prompt_chars"]
        stats[task]["source_dataset"] = current["sources"]
        manifest["files"][task + ".jsonl"].update({
            "records": current["records"],
            "size_bytes": current["size_bytes"],
            "sha256": current["sha256"],
        })
        redundancy[task] = {k: v for k, v in current.items() if k != "sources"}
    seed_map = result["seed_map"]
    stats["seed_map"] = {k: v for k, v in seed_map.items() if k != "covers_published_prompts"}
    manifest["files"]["prompt_seed_map.csv"] = {
        "records": seed_map["records"],
        "size_bytes": seed_map["size_bytes"],
        "sha256": seed_map["sha256"],
    }
    for path, payload in ((stats_path, stats), (manifest_path, manifest), (redundancy_path, redundancy)):
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def check_files(result: dict) -> list[str]:
    stats = json.loads((ROOT / "DATASET_STATS.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "data" / "MANIFEST.json").read_text(encoding="utf-8"))
    redundancy = json.loads((ROOT / "PROMPT_REDUNDANCY.json").read_text(encoding="utf-8"))
    problems = []
    for task in ("focused", "extended"):
        current = result[task]
        expected = redundancy[task]
        for key, value in current.items():
            if key == "sources":
                if stats[task]["source_dataset"] != value:
                    problems.append(f"{task}: stale source distribution")
            elif key == "maximum_3gram_jaccard":
                if abs(expected[key] - value) > 1e-12:
                    problems.append(f"{task}: stale maximum 3-gram Jaccard")
            elif expected.get(key) != value:
                problems.append(f"{task}: stale redundancy field {key}")
        if stats[task]["file_sha256"] != current["sha256"]:
            problems.append(f"{task}: stale dataset SHA")
        if stats[task]["prompt_char_length_min"] != current["min_prompt_chars"] or stats[task]["prompt_char_length_max"] != current["max_prompt_chars"]:
            problems.append(f"{task}: stale character-length distribution")
        file_record = manifest["files"][task + ".jsonl"]
        for k, v in (("records", current["records"]), ("size_bytes", current["size_bytes"]), ("sha256", current["sha256"])):
            if file_record.get(k) != v:
                problems.append(f"{task}: stale manifest {k}")
        if current["pairs_at_or_above_0_85"]:
            problems.append(f"{task}: {current['pairs_at_or_above_0_85']} prompt pairs exceed threshold")
    seed_map = result["seed_map"]
    if not seed_map["covers_published_prompts"]:
        problems.append("seed map: does not cover exactly the published Prompt Cards")
    for key in ("records", "distinct_seeds", "shared_across_tasks", "focused_only", "extended_only",
                "sha256", "size_bytes"):
        if stats["seed_map"].get(key) != seed_map[key]:
            problems.append(f"seed map: stale {key}")
    file_record = manifest["files"].get("prompt_seed_map.csv", {})
    for k, v in (("records", seed_map["records"]), ("size_bytes", seed_map["size_bytes"]),
                 ("sha256", seed_map["sha256"])):
        if file_record.get(k) != v:
            problems.append(f"seed map: stale manifest {k}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify existing files, without updating")
    args = parser.parse_args()
    result = calculate()
    problems = check_files(result) if args.check else []
    if not args.check:
        update_files(result)
    for task in ("focused", "extended"):
        value = result[task]
        print(f"{task}: {value['records']} prompts, maximum Jaccard={value['maximum_3gram_jaccard']:.9f}, pair={value['highest_pair']}, >=0.85={value['pairs_at_or_above_0_85']}")
    seed_map = result["seed_map"]
    print(f"seed map: {seed_map['records']} cards, {seed_map['distinct_seeds']} Seeds "
          f"({seed_map['shared_across_tasks']} shared, {seed_map['focused_only']} focused-only, "
          f"{seed_map['extended_only']} extended-only), covers published prompts="
          f"{seed_map['covers_published_prompts']}")
    for item in problems:
        print("FAIL:", item)
    return int(bool(problems))


if __name__ == "__main__":
    raise SystemExit(main())
