"""Deterministic annotation/quality-stage inputs for CSWEval.

This module derives stage inputs from supplied text and model outputs. Model calls are
handled by the transport layer; the protocol files in this repository define scoring and naming.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from evidence_catalog import validate_evidence_references

def _latin_letter_pattern():
    """Unicode Latin letters with attached marks, using the source text unchanged."""
    letters, marks = [], []
    for codepoint in range(0x110000):
        char = chr(codepoint)
        category = unicodedata.category(char)
        if category.startswith("L") and "LATIN" in unicodedata.name(char, ""):
            letters.append(char)
        elif category.startswith("M"):
            marks.append(char)
    return rf"(?:[{re.escape(''.join(letters))}][{re.escape(''.join(marks))}]*)"


LATIN_LETTER = _latin_letter_pattern()
LATIN_ALNUM = rf"(?:{LATIN_LETTER}|[0-9])"
LATIN_TOKEN = (rf"(?:{LATIN_LETTER}(?:\.{LATIN_LETTER})+\.)|"
               rf"(?:{LATIN_LETTER}{LATIN_ALNUM}*(?:[-']{LATIN_ALNUM}+)*(?:\+\+|#)?)")
LATIN_CONNECTOR = r"(?:[ \t]+|[ \t]*(?:&|/)[ \t]*)"
FOCUSED_LATIN = re.compile(rf"(?:{LATIN_TOKEN})(?:{LATIN_CONNECTOR}(?:{LATIN_TOKEN}))*")
SENTENCE = re.compile(r"[^。！？!?\n]+[。！？!?]?", re.MULTILINE)
D2 = [f"D2.{i}" for i in range(1, 9)]
D3 = [f"D3.{i}" for i in range(1, 6)]
WHOLE_EXTENDED = ["M-F1", "M-F2A", "M-F2B", "M-F3.1", "M-F3.2", "M-F3.3", "M-F3.4"]


def digest(value):
    if isinstance(value, str):
        data = value.encode("utf-8")
    else:
        data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def validate_candidates(article, candidates):
    if [v["candidate_id"] for v in candidates] != [f"C{i:03d}" for i in range(1, len(candidates) + 1)]:
        raise ValueError("candidate IDs not sequential")
    last = 0
    for c in candidates:
        start = c.get("start_codepoint", c.get("start_char"))
        end = c.get("end_codepoint", c.get("end_char"))
        if not (last <= start < end <= len(article)) or article[start:end] != c["text"]:
            raise ValueError(f"candidate offset mismatch: {c['candidate_id']}")
        last = end


def focused_candidates(article):
    result = [{"candidate_id": f"C{i:03d}", "text": m.group(), "start_char": m.start(), "end_char": m.end()}
              for i, m in enumerate(FOCUSED_LATIN.finditer(article), 1)]
    validate_candidates(article, result)
    return result


def _latin_char(c):
    return bool(unicodedata.combining(c) or ("LATIN" in unicodedata.name(c, "") and unicodedata.category(c).startswith("L")))


def extended_candidates(article):
    result, start = [], None
    for i, c in enumerate(article + "\0"):
        if i < len(article) and _latin_char(c):
            if start is None: start = i
            continue
        if start is None: continue
        result.append({"candidate_id": f"C{len(result)+1:03d}", "start_codepoint": start,
                       "end_codepoint": i, "text": article[start:i],
                       "segment_id": f"S{1 + article[:start].count(chr(10)):03d}"})
        start = None
    validate_candidates(article, result)
    return result


def evidence_units(text, prefix, task, protected=()):
    """Sentence-aligned evidence units; never cut a Latin candidate to fit 200 chars."""
    source = "VISIBLE_PROMPT" if prefix == "P" else ("GENERATED_TEXT" if task == "focused" else "ARTICLE")
    start_key, end_key = (("start_char", "end_char") if task == "focused" else ("start_codepoint", "end_codepoint"))
    if prefix == "P" and task == "extended":
        # Extended prompt-unit partition is run-specific. The public protocol uses a
        # deterministic partition, keeping character spans null as specified.
        keys = lambda a, b: {start_key: None, end_key: None}
    else:
        keys = lambda a, b: {start_key: a, end_key: b}
    pieces = []
    for match in SENTENCE.finditer(text):
        a, b = match.span()
        while a < b and text[a].isspace(): a += 1
        while b > a and text[b-1].isspace(): b -= 1
        while a < b:
            end = min(a + 180, b)
            for ps, pe in protected:
                if ps < end < pe: end = ps if ps > a else pe
            if end <= a or end - a > 200: raise ValueError("candidate exceeds evidence chunk limit")
            pieces.append((a, end))
            a = end
    merged = []
    for a, b in pieces:
        if task == "focused" and merged and b - merged[-1][0] <= 180:
            merged[-1] = (merged[-1][0], b)
        else: merged.append((a, b))
    return [{"evidence_id": f"{prefix}{i:03d}" if task == "focused" else f"{prefix}{i:04d}",
             "source": source, "text": text[a:b], **keys(a, b)} for i, (a, b) in enumerate(merged, 1)]


def d1_allowlist(event, units):
    start = event["start_codepoint"]
    allowed = []
    for u in units:
        uid = u["evidence_id"]
        if uid.startswith("P"):
            allowed.append(uid)
        elif uid.startswith("T"):
            end = u.get("end_codepoint", u.get("end_char"))
            if end is None: raise ValueError("article evidence missing ending offset")
            if end <= start: allowed.append(uid)
    return allowed


def freeze_focused(article, candidate_items, discovery_output, *, expected_identity=None):
    """Validate and freeze GPT-5.6-Sol's semantic event discovery."""
    proposal = discovery_output.get("event_proposal")
    if not proposal or proposal.get("status") != "MODEL_DISCOVERED":
        raise ValueError("expected model-discovered event proposal")
    metadata = discovery_output.get("metadata", {})
    for field in ("prompt_id", "text_id", "params_id", "params_sha256"):
        value = metadata.get(field)
        if not isinstance(value, str) or not value or proposal.get(field) != value:
            raise ValueError(f"Focused proposal identity mismatch: {field}")
    if proposal.get("normalized_story_text_sha256") != digest(article):
        raise ValueError("Focused proposal identity mismatch: normalized_story_text_sha256")
    if expected_identity is not None:
        for field in ("run_id", "prompt_id", "text_id", "params_id", "params_sha256"):
            if field in expected_identity and metadata.get(field) != expected_identity[field]:
                raise ValueError(f"Focused discovery identity does not match input: {field}")
    by_id = {c["candidate_id"]: c for c in candidate_items}
    review_ids = [v["candidate_id"] for v in discovery_output["candidate_reviews"]]
    if len(review_ids) != len(candidate_items) or set(review_ids) != set(by_id):
        raise ValueError("candidate reviews must cover every candidate exactly once")
    reviews = {v["candidate_id"]: v["label"] for v in discovery_output["candidate_reviews"]}
    true_ids = {i for i, label in reviews.items() if label == "TRUE_CSW"}
    events, covered, previous_end = [], set(), 0
    for e in proposal["events"]:
        ids = e["candidate_ids"]
        if not ids or len(set(ids)) != len(ids) or any(i not in by_id for i in ids) or covered.intersection(ids):
            raise ValueError("invalid/overlapping frozen candidates")
        if not true_ids.intersection(ids):
            raise ValueError("each event must contain at least one TRUE_CSW candidate")
        span = [by_id[i] for i in ids]
        start, end = min(x["start_char"] for x in span), max(x["end_char"] for x in span)
        if (start, end) != (e["start_codepoint"], e["end_codepoint"]):
            raise ValueError("proposal event span differs from candidates")
        contained = [c["candidate_id"] for c in candidate_items if start <= c["start_char"] and c["end_char"] <= end]
        if contained != ids: raise ValueError("event candidate membership differs from spanned candidates")
        if not 0 <= start < end <= len(article): raise ValueError("event outside narrative")
        if start < previous_end:
            raise ValueError("events must follow textual order without overlapping spans")
        previous_end = end
        covered.update(ids)
        events.append(e)
    if not true_ids.issubset(covered): raise ValueError("TRUE_CSW candidate missing from event manifest")
    if not events: raise ValueError("no CS events: not scorable")
    if [e["event_id"] for e in events] != [f"E{i:03d}" for i in range(1, len(events)+1)]:
        raise ValueError("event IDs not sequential")
    return {"events": events, "event_count": len(events), "review_status": "MODEL_DISCOVERED"}


def freeze_extended(article, candidates, discovery_output):
    """Freeze model-discovered Extended events without changing their boundaries."""
    a = discovery_output.get("annotation", discovery_output)
    by_id = {x["candidate_id"]: x for x in candidates}
    assessment_ids = [x["candidate_id"] for x in a["candidate_assessments"]]
    if len(assessment_ids) != len(candidates) or set(assessment_ids) != set(by_id):
        raise ValueError("candidate assessments must cover every candidate exactly once")
    assessments = {x["candidate_id"]: x["classification"] for x in a["candidate_assessments"]}
    true_ids = {k for k, v in assessments.items() if v == "TRUE_CSW"}
    seen, events, previous_end = set(), [], 0
    for i, item in enumerate(a["events"], 1):
        ids = item["candidate_ids"]
        if not ids or len(set(ids)) != len(ids) or any(cid not in by_id for cid in ids) or seen.intersection(ids):
            raise ValueError("overlapping/unknown event candidates")
        if not true_ids.intersection(ids):
            raise ValueError("each event must contain at least one TRUE_CSW candidate")
        group = [by_id[cid] for cid in ids]
        start, end = min(x["start_codepoint"] for x in group), max(x["end_codepoint"] for x in group)
        if not 0 <= start < end <= len(article): raise ValueError("event span outside article")
        if start < previous_end:
            raise ValueError("events must follow textual order without overlapping spans")
        previous_end = end
        events.append({"event_id": f"E{i:03d}", "candidate_ids": ids, "start_codepoint": start,
                       "end_codepoint": end, "text": article[start:end],
                       "speaker_slot": item["speaker_slot"], "scene_id": item["scene_id"]})
        seen.update(ids)
    if not true_ids.issubset(seen) or not events:
        raise ValueError("TRUE_CSW coverage or event count invalid")
    return {"protocol_id": "csweval_extended_effective_event_scoring_view",
            "status": "FROZEN_FOR_SCORING", "event_count": len(events), "events": events}


def build_focused_scoring_input(visible_prompt, article, discovery, *, prompt_id, text_id,
                                params_id, params_sha256, vocabulary_id, vocabulary_sha256):
    candidates = focused_candidates(article)
    frozen = freeze_focused(article, candidates, discovery, expected_identity={
        "prompt_id": prompt_id, "text_id": text_id,
        "params_id": params_id, "params_sha256": params_sha256})
    units = evidence_units(visible_prompt, "P", "focused") + evidence_units(
        article, "T", "focused", [(c["start_char"],c["end_char"]) for c in candidates])
    events = [{"event_id": e["event_id"], "start_codepoint": e["start_codepoint"],
               "end_codepoint": e["end_codepoint"], "event_text": article[e["start_codepoint"]:e["end_codepoint"]],
               "speaker_or_source_id": e["speaker_or_source_id"], "carrier_type": e["carrier_type"],
               "d1_allowed_evidence_ids": d1_allowlist(e, units)} for e in frozen["events"]]
    return {"run_id": f"focused-{text_id}", "prompt_id": prompt_id, "text_id": text_id,
            "protocol_id": "csweval_focused_stage_b_input",
            "source_judge_protocol_id": "csweval_focused_quality",
            "source_judge_input_sha256": digest(discovery),
            "event_manifest_sha256": digest(frozen), "visible_prompt": visible_prompt,
            "normalized_story_text": article, "normalized_story_text_sha256": digest(article),
            "evidence_units": units, "events": events, "params_id": params_id,
            "params_sha256": params_sha256, "vocabulary_id": vocabulary_id,
            "vocabulary_sha256": vocabulary_sha256}


def validate_frozen_evidence_units(visible_prompt, article, units):
    """Bind Judge citations to the actual immutable evidence-stage catalog."""
    if not isinstance(units, list) or not units:
        raise ValueError("Extended needs frozen evidence units from the Sol evidence stage")
    seen = set()
    for u in units:
        eid, text = u["evidence_id"], u["text"]
        if eid in seen or not text: raise ValueError("duplicate or empty evidence unit")
        seen.add(eid)
        start, end = u.get("start_codepoint"), u.get("end_codepoint")
        if eid.startswith("P"):
            if u["source"] != "VISIBLE_PROMPT" or start is not None or end is not None or text not in visible_prompt:
                raise ValueError(f"prompt evidence {eid} does not match the public prompt")
        elif eid.startswith("T"):
            if u["source"] != "ARTICLE" or type(start) is not int or type(end) is not int or not 0 <= start < end <= len(article) or article[start:end] != text:
                raise ValueError(f"article evidence {eid} does not match frozen story offsets")
        else:
            raise ValueError(f"unknown evidence ID prefix {eid}")
    return True


def build_extended_scoring_input(visible_prompt, article, registry, frozen_events, semantic,
                                 params, blind_id, *, params_file_sha256, frozen_evidence_units):
    """`semantic` must be an actual GPT-5.6-Sol evidence-stage response."""
    if len(registry) != 2 or [x["character_slot"] for x in registry] != ["A", "B"]:
        raise ValueError("Extended scoring needs real ordered A/B character registry")
    if not all(k in semantic for k in ("prompt_expected_facts", "article_observed_facts",
                                     "narrative_system", "character_profiles", "ambiguities")):
        raise ValueError("missing real model-constructed evidence")
    validate_frozen_evidence_units(visible_prompt, article, frozen_evidence_units)
    units = copy.deepcopy(frozen_evidence_units)
    validate_evidence_references(semantic, {unit["evidence_id"] for unit in units})
    events = [{**e, "d1_allowed_evidence_ids": d1_allowlist(e, units)} for e in frozen_events["events"]]
    manifest = {"protocol_id": "csweval_extended_evidence_manifest",
                "manifest_status": "READY_FOR_SCORING", "blind_document_id": blind_id,
                "article_sha256": digest(article), "frozen_evidence": {
                    "prompt_expected_facts": copy.deepcopy(semantic["prompt_expected_facts"]),
                    "article_observed_facts": copy.deepcopy(semantic["article_observed_facts"]),
                    "narrative_system": copy.deepcopy(semantic["narrative_system"]),
                    "character_profiles": [copy.deepcopy(semantic["character_profiles"][slot]) for slot in ("A", "B")],
                    "ambiguities": copy.deepcopy(semantic["ambiguities"])},
                "quality_scores_present": False, "scoring_allowed": True,
                "selected_evidence_model": "gpt-5.6-sol"}
    contract = {"d1_dimensions": ["D1.role", "D1.world"], "d2_internal_checklist": D2,
                "d2_output_dimension": "D2.overall", "d3_internal_checklist": D3,
                "d3_output_dimension": "D3.overall", "subdimension_scores_emitted": False,
                "scene_salience_labels": ["S1", "S2", "S3"],
                "scene_salience_numeric_weight": {"S1": 1, "S2": 2, "S3": 3}}
    return {"protocol_id": "csweval_extended_scoring_input",
            "params_id": params["params_id"], "params_sha256": params_file_sha256,
            "blind_document_id": blind_id, "article_sha256": digest(article), "article_text": article,
            "prompt_constraints": {"visible_prompt": visible_prompt}, "character_registry": registry,
            "evidence_units": units, "event_manifest": {**frozen_events, "events": events},
            "evidence_manifest": manifest, "event_judgment_contract": contract,
            "whole_text_dimension_ids": WHOLE_EXTENDED,
            "score_scale": {"minimum": 1, "maximum": 5},
            "separate_narrator_profile": False, "composite_score_requested": False}


def specialize_schema(schema, candidate_ids=(), event_ids=(), evidence_ids=(), generated_ids=()):
    """Pin request-specific ID enums for the current scoring input."""
    schema = copy.deepcopy(schema)
    mappings = {"^C[0-9]{3,}$": list(candidate_ids), "^E[0-9]{3,}$": list(event_ids),
                "^[PT][0-9]{3,}$": list(evidence_ids), "^T[0-9]{3,}$": list(generated_ids)}
    def walk(value):
        if isinstance(value, list):
            for v in value: walk(v)
        if isinstance(value, dict):
            pat = value.get("pattern")
            if pat in mappings and mappings[pat]:
                value.pop("pattern"); value["enum"] = mappings[pat]
            for v in value.values(): walk(v)
    walk(schema)
    return schema
def registry_from_prompt(visible_prompt):
    """Read Extended A/B names and language constraints directly from the final Prompt Card."""
    match = re.search(r"核心人物(?:是|为)([\u4e00-\u9fff]{2,4})与([\u4e00-\u9fff]{2,4})", visible_prompt)
    if not match:
        raise ValueError("could not identify the two named characters in the Extended Prompt Card; pass --registry")
    entries = []
    for slot, name in zip(("A", "B"), match.groups()):
        snippets = []
        for found in re.finditer(re.escape(name), visible_prompt):
            segment = visible_prompt[found.start():]
            segment = re.split(r"[；。\n]", segment, maxsplit=1)[0]
            if any(x in segment for x in ("英语", "英文", "语言条件", "中文")):
                snippets.append(segment)
        if not snippets:
            raise ValueError(f"language conditions not explicit for {name}; pass --registry")
        best = max(snippets, key=lambda x: len(x))
        entries.append({"character_slot": slot, "display_name": name,
                        "language_condition": {"prompt_excerpt": best}})
    return entries


def validate_scene_order(frozen_events, semantic):
    """Scene identifiers are chronological; refuse incompatible event-to-scene bindings."""
    scenes = semantic["narrative_system"]["scenes"]
    expected = [f"S{i:03d}" for i in range(1, len(scenes) + 1)]
    actual = [scene["scene_id"] for scene in scenes]
    if actual != expected:
        raise ValueError("evidence scenes are not sequential")
    indexed = {sid: i for i, sid in enumerate(expected)}
    prior = -1
    for event in sorted(frozen_events["events"], key=lambda x: x["start_codepoint"]):
        scene = event["scene_id"]
        if scene not in indexed:
            raise ValueError(f"discovery scene ID {scene} not in evidence timeline")
        if indexed[scene] < prior:
            raise ValueError("event scene IDs contradict chronological evidence scenes")
        prior = indexed[scene]
    return True
