"""Small, invented protocol fixtures; never model judgments or paper results."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STORIES = json.loads((Path(__file__).with_name("narratives.json")).read_text(encoding="utf-8"))
WHOLE_EXT = ("M-F1", "M-F2A", "M-F2B", "M-F3.1", "M-F3.2", "M-F3.3", "M-F3.4")


def focused_discovery(payload):
    candidates = payload["candidates"]
    first = candidates[0]
    return {
        "metadata": {k: payload[k] for k in ("run_id", "prompt_id", "text_id", "protocol_id", "params_id", "params_sha256")},
        "candidate_reviews": [
            {"candidate_id": c["candidate_id"], "label": "TRUE_CSW" if c is first else "NON_CSW", "reason": "Constructed label"}
            for c in candidates
        ],
        "event_proposal": {
            "proposal_id": "EP-example", "prompt_id": payload["prompt_id"],
            "text_id": payload["text_id"],
            "normalized_story_text_sha256": payload["normalized_story_text_sha256"],
            "params_id": payload["params_id"], "params_sha256": payload["params_sha256"],
            "proposal_sha256": "a" * 64, "status": "MODEL_DISCOVERED",
            "events": [{
                "event_id": "E001", "start_codepoint": first["start_char"],
                "end_codepoint": first["end_char"], "speaker_or_source_id": "speaker-A",
                "carrier_type": "SPOKEN", "candidate_ids": [first["candidate_id"]],
                "boundary_reason": "One constructed switch"
            }]
        }
    }


def focused_score(payload, judge="gpt"):
    score = 3 if judge == "gpt" else 4
    meta = {k: payload[k] for k in ("run_id", "prompt_id", "text_id", "params_id",
                                      "params_sha256", "vocabulary_id", "vocabulary_sha256")}
    meta["protocol_id"] = "csweval_focused_stage_b"
    return {
        "metadata": meta,
        "event_evaluations": [{
            "event_id": e["event_id"],
            "d1": {"role": "PASS", "world": "PASS"},
            "d2": {f"D2.{i}": score for i in range(1, 9)},
            "d3": {f"D3.{i}": score for i in range(1, 6)},
            "narrative_salience": "S2", "evidence_ids": [payload["evidence_units"][0]["evidence_id"]],
            "summary": "Synthetic assessment"
        } for e in payload["events"]],
        "literary_quality": {
            "dimensions": {f"L{i}": 3 for i in range(1, 6)},
            "high_score_evidence": []
        },
        "exceptions": []
    }


def extended_discovery(payload):
    first = payload["mechanical_candidates"][0]
    return {
        "protocol_id": "csweval_extended_event_discovery_output",
        "blind_document_id": payload["blind_document_id"],
        "article_sha256": payload["article_sha256"], "judge_model": "gpt-5.6-sol",
        "manifest_challenge": {"raised": False, "reason": None},
        "candidate_assessments": [{
            "candidate_id": c["candidate_id"],
            "classification": "TRUE_CSW" if c is first else "NON_CSW",
            "speaker_slot": "A", "function_labels": ["constructed"],
            "rationale": "Synthetic classification"
        } for c in payload["mechanical_candidates"]],
        "events": [{
            "proposal_id": "P001", "candidate_ids": [first["candidate_id"]],
            "speaker_slot": "A", "scene_id": "S001", "rationale": "One constructed switch"
        }]
    }


def extended_semantic(payload):
    prompt_id = next(c["evidence_id"] for c in payload["evidence_catalog"]["candidates"]
                     if c["source"] == "PROMPT")
    article_id = next(c["evidence_id"] for c in payload["evidence_catalog"]["candidates"]
                      if c["source"] == "ARTICLE")
    def fact(eid):
        return {"fact_id": "synthetic", "statement": "Observed only in fixture",
                "evidence": [eid]}
    def profile():
        return {
            "baseline": {
                "goals": ["Resolve a small question"], "motivations": ["Curiosity"],
                "knowledge": ["A short note"], "abilities": ["Speak"],
                "disposition": ["Patient"], "relationship_start_state": "Peers",
                "evidence": [prompt_id]
            },
            "language_resources": {
                "repertoire": ["Chinese", "English"], "function_mapping": ["Conversation"],
                "relationship_mapping": ["Peers"], "affect_mapping": ["Curiosity"],
                "language_arc": "One switch"
            },
            "meaningful_non_switching": [], "profile_evidence": [prompt_id]
        }
    def state():
        return {
            "presence": "PRESENT", "goal": "Discuss", "knowledge": ["One fact"],
            "emotion": "Calm", "stance_toward_other": "Listening",
            "language_behavior": "Speaks", "change_from_previous": "Initial",
            "evidence": [article_id]
        }
    return {
        "event_manifest_challenge": {"raised": False, "reason": None},
        "prompt_expected_facts": [fact(prompt_id)],
        "article_observed_facts": [fact(article_id)],
        "narrative_system": {
            "scenes": [{
                "state_before": "Before", "actions": ["Dialogue"],
                "state_after": "After", "causal_from_previous": None,
                "relationship_state": "Peers",
                "character_states": {"A": state(), "B": state()},
                "evidence": [article_id]
            }],
            "world_facts": [fact(article_id)],
            "global_causal_summary": [fact(article_id)],
            "turning_points": ["A conversation"],
            "closure_state": "Resolved", "voice_and_focalization": "Third person"
        },
        "character_profiles": {"A": profile(), "B": profile()}, "ambiguities": []
    }


def extended_score(payload, model):
    score = 3 if model == "gpt-5.6-sol" else 4
    units = {unit["evidence_id"]: unit["text"] for unit in payload["evidence_units"]}
    prompt_id = next(k for k in units if k.startswith("P"))
    article_id = next(k for k in units if k.startswith("T"))
    def value(ids, **fields):
        return {"evidence_ids": ids, "evidence_strings": [units[i] for i in ids],
                "rationale": "Synthetic assessment", **fields}
    return {
        "protocol_id": "csweval_extended_scoring_output",
        "params_id": payload["params_id"], "params_sha256": payload["params_sha256"],
        "blind_document_id": payload["blind_document_id"],
        "article_sha256": payload["article_sha256"], "judge_model": model,
        "manifest_challenges": {
            "event_manifest": {"raised": False, "reason": None},
            "evidence_manifest": {"raised": False, "reason": None}
        },
        "prompt_constraint_compliance": "PASS",
        "prompt_constraint_evidence_ids": [prompt_id],
        "event_judgments": [{
            "event_id": event["event_id"],
            "d1": {
                "role": value([prompt_id], verdict="PASS"),
                "world": value([prompt_id], verdict="PASS")
            },
            "d2_overall": value([article_id], score=score),
            "d3_overall": value([article_id], score=score),
            "scene_salience": value([article_id], label="S2")
        } for event in payload["event_manifest"]["events"]],
        "whole_text_dimensions": [
            {"dimension_id": dim, **value([article_id], score=score)}
            for dim in WHOLE_EXT
        ],
        "composite_score": None, "ranking": None
    }


class SyntheticTransport:
    """Returns structurally valid model-like responses from the current frozen input."""
    def __init__(self, fail_stage=None):
        self.calls = []
        self.fail_stage = fail_stage

    def run(self, name, model, payload, prompt, schema, tool, task):
        self.calls.append((name, model, copy.deepcopy(payload)))
        if name == self.fail_stage:
            raise ValueError("Synthetic stage rejected")
        if task == "focused":
            return focused_discovery(payload) if name == "discovery" else focused_score(
                payload, "gpt" if model == "gpt-5.6-sol" else "deepseek")
        if name == "discovery":
            return extended_discovery(payload)
        if name == "evidence":
            return extended_semantic(payload)
        return extended_score(payload, model)
