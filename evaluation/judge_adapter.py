"""Convert structured CSWEval judge outputs into CSWEval aggregation records.

Rejects inconsistent event identity, extra dimensions and a GPT judge other than
GPT-5.6-Sol. Preserves authenticated model labels and score IDs.
"""
from __future__ import annotations

from typing import Any

D2 = {f"D2.{i}" for i in range(1, 9)}
D3 = {f"D3.{i}" for i in range(1, 6)}
WHOLE_EXT = {"M-F1", "M-F2A", "M-F2B", "M-F3.1", "M-F3.2", "M-F3.3", "M-F3.4"}


def _validate_scores(scores: dict, required: set):
    if set(scores) != required:
        raise ValueError(f"dimensions mismatch: missing={sorted(required-set(scores))}, extra={sorted(set(scores)-required)}")
    for key, score in scores.items():
        if score in ("NA", "IE", None):
            continue
        if type(score) is not int or not 1 <= score <= 5:
            raise ValueError(f"{key} is not an ordinal score 1-5/NA/IE")


def _focused(output: dict):
    whole = output["literary_quality"]["dimensions"]
    if isinstance(whole, list):
        whole = {e.get("dimension", e.get("dimension_id")): e["score"] for e in whole}
    if any(isinstance(v, dict) for v in whole.values()):
        whole = {k: v["score"] if isinstance(v, dict) else v for k, v in whole.items()}
    _validate_scores(whole, {f"L{i}" for i in range(1, 6)})
    events = []
    for e in output["event_evaluations"]:
        _validate_scores(e["d2"], D2)
        _validate_scores(e["d3"], D3)
        events.append({"event_id": e["event_id"], "d1_role": e["d1"]["role"],
                       "d1_world": e["d1"]["world"], "salience": e["narrative_salience"],
                       "d2_subdimensions": e["d2"], "d3_subdimensions": e["d3"]})
    return whole, events


def _extended(output: dict):
    whole = {e["dimension_id"]: e["score"] for e in output["whole_text_dimensions"]}
    _validate_scores(whole, WHOLE_EXT)
    events = []
    for e in output["event_judgments"]:
        events.append({"event_id": e["event_id"],
                       "d1_role": e["d1"]["role"]["verdict"],
                       "d1_world": e["d1"]["world"]["verdict"],
                       "salience": e["scene_salience"]["label"],
                       "d2_overall": e["d2_overall"]["score"],
                       "d3_overall": e["d3_overall"]["score"]})
    return whole, events


def to_aggregation(task: str, output: dict, *, article_id: str, prompt_card_id: str,
                   generation_model: str, judge_model: str, frozen_events: list[dict]) -> dict[str, Any]:
    if task not in ("focused", "extended"):
        raise ValueError("task must be focused or extended")
    mapping = {"gpt-5.6-sol": "gpt", "deepseek-v4-flash": "deepseek"}
    if judge_model not in mapping:
        raise ValueError(f"unsupported judge model: {judge_model}")
    if task == "extended" and output.get("judge_model") not in (None, judge_model):
        raise ValueError("judge model in response differs from authenticated route")
    whole, events = _focused(output) if task == "focused" else _extended(output)
    ids = [e["event_id"] for e in frozen_events]
    if not ids or ids != [e["event_id"] for e in events] or len(set(ids)) != len(ids):
        raise ValueError("judge output does not exactly follow frozen event list")
    for e in events:
        if e["d1_role"] not in ("PASS", "FAIL") or e["d1_world"] not in ("PASS", "FAIL"):
            raise ValueError("invalid D1 gate")
        if e["salience"] not in ("S1", "S2", "S3", "NA", "IE"):
            raise ValueError("invalid salience")
    return {"task": task, "article_id": article_id, "prompt_card_id": prompt_card_id,
            "model": generation_model, "judge": mapping[judge_model], "whole": whole,
            "events": events}