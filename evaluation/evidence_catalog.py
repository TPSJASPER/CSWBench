"""Deterministic Extended evidence catalog, built from the released evidence extraction contract.

A catalog is frozen *before* GPT-5.6-Sol evidence construction and is reused
verbatim when composing the quality-judge input. Evidence texts are exact
substrings of the public Prompt Card or generated article.
"""
from __future__ import annotations

import hashlib
import json
import re

SEGMENT = re.compile(
    r"[^。！？!?；;，,：:\r\n\v\f\x85\u2028\u2029]+"
    r"(?:[。！？!?；;，,：:]+|(?=[\r\n\v\f\x85\u2028\u2029])|$)"
    r"|[。！？!?；;，,：:]+"
)


def canonical_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def catalog(visible_prompt: str, article: str):
    candidates = []
    units = []
    for prefix, source, public_source, source_tag in (
        ("P", "PROMPT", visible_prompt, "VISIBLE_PROMPT"),
        ("T", "ARTICLE", article, "ARTICLE"),
    ):
        index = 0
        covered_end = 0
        for m in SEGMENT.finditer(public_source):
            raw = m.group(0)
            substring = raw.strip()
            if not substring:
                continue
            left = len(raw) - len(raw.lstrip())
            start = m.start() + left
            end = start + len(substring)
            if public_source[start:end] != substring:
                raise ValueError("evidence source substring/offset mismatch")
            if any(not char.isspace() for char in public_source[covered_end:start]):
                raise ValueError(f"{prefix} evidence catalog omitted non-whitespace text")
            covered_end = end
            index += 1
            name = f"{prefix}{index:04d}"
            candidates.append({"evidence_id": name, "source": source, "text": substring})
            units.append({"evidence_id": name, "source": source_tag, "text": substring,
                          "start_codepoint": start if prefix == "T" else None,
                          "end_codepoint": end if prefix == "T" else None})
        if any(not char.isspace() for char in public_source[covered_end:]):
            raise ValueError(f"{prefix} evidence catalog omitted trailing text")
        if index == 0:
            raise ValueError(f"no {prefix} evidence candidates")
    item = {"protocol_id": "csweval_extended_evidence_catalog", "candidates": candidates}
    item["catalog_sha256"] = canonical_sha(item)
    return item, units


def evidence_request(visible_prompt, article, character_registry, frozen_events, evidence_catalog):
    if [x.get("character_slot") for x in character_registry] != ["A", "B"]:
        raise ValueError("an explicit ordered A/B character registry is required")
    return {
        "article_text": article,
        "prompt_constraints": {"visible_prompt": visible_prompt},
        "character_registry": [{"character_slot": e["character_slot"],
                                "language_condition": e["language_condition"]}
                               for e in character_registry],
        "event_manifest": frozen_events,
        "evidence_catalog": evidence_catalog,
        "separate_narrator_profile": False,
        "quality_scores_requested": False,
    }


def validate_evidence_references(records, permitted):
    """Every structured evidence citation must resolve to the frozen catalog."""
    def visit(value, parent=""):
        if isinstance(value, dict):
            for k, v in value.items():
                visit(v, k)
        elif isinstance(value, list):
            if parent in {"evidence", "profile_evidence"}:
                for identifier in value:
                    if identifier not in permitted:
                        raise ValueError(f"unrecognized evidence citation {identifier}")
            else:
                for v in value:
                    visit(v, parent)

    visit(records)


def validate_semantic(semantic, evidence_catalog):
    """Check ID citations and locally bind sequential scene IDs, never infer semantics."""
    import copy
    if not all(k in semantic for k in ("narrative_system", "character_profiles")):
        raise ValueError("Sol evidence response omitted required semantic sections")
    permitted = {x["evidence_id"] for x in evidence_catalog["candidates"]}
    result = copy.deepcopy(semantic)
    validate_evidence_references(result, permitted)
    scenes = result["narrative_system"].get("scenes", [])
    for n, scene in enumerate(scenes, 1):
        if "scene_id" in scene:
            raise ValueError("model must not generate deterministic scene IDs")
        scene["scene_id"] = f"S{n:03d}"
        for slot, state in scene.get("character_states", {}).items():
            if state.get("presence") == "NOT_PRESENT":
                if state.get("change_from_previous") != "NOT_PRESENT" or any(
                    state.get(k) not in (None, [], "") for k in
                    ("goal", "knowledge", "emotion", "stance_toward_other", "language_behavior")
                ):
                    raise ValueError(f"absent character {slot} has fabricated attributes")
    return result
