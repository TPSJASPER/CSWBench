#!/usr/bin/env python3
"""Run one CSWEval stage against a JSON-schema-constrained model endpoint.

CSWEval is a multi-stage protocol (see PROTOCOL.md). Every stage is a single request whose
system message is one of the frozen prompts in `prompts/`, whose user message wraps the
stage input as JSON, and whose output is forced through a function/tool call whose
parameters are one of the frozen schemas in `schemas/`. This script reproduces that
envelope. It does not decide any score: the scoring logic lives in the frozen prompts and
schemas shipped next to it.

Usage
-----
    # offline: show the request that would be sent, sending nothing
    python run_stage.py --task extended --prompt prompts/extended_scoring_prompt.md \
        --schema schemas/extended_scoring_output_schema.json \
        --tool-name submit_long_judge_scoring \
        --input my_scoring_input.json --dry-run

    python run_stage.py --task extended ... --input my_scoring_input.json \
        --config ../configs/models.example.json --model gpt-5.6-sol \
        --out outputs/

The API key is read from the environment variable named in the config file. It is never
written to any output.
"""
from __future__ import annotations

import argparse
import copy
import re
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from evidence_catalog import validate_evidence_references
from pipeline_inputs import freeze_focused


class ManifestChallengeError(ValueError):
    """A quality judge identified an unresolved frozen-manifest problem."""
    def __init__(self, challenges):
        self.challenges = copy.deepcopy(challenges)
        reasons = [f"{name}: {item.get('reason') or 'No reason supplied'}"
                   for name, item in challenges.items() if item.get("raised")]
        super().__init__("unresolved Extended manifest challenge: " + "; ".join(reasons))

# The frozen prompts refer to the wrapped payload as <JUDGE_INPUT> / <LONG_JUDGE_INPUT>.
INPUT_ENVELOPE = {
    "focused": "<JUDGE_INPUT>\n{payload}\n</JUDGE_INPUT>",
    "extended": "<LONG_JUDGE_INPUT>\n{payload}\n</LONG_JUDGE_INPUT>",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def build_body(prompt_text: str, schema: dict, tool_name: str, payload, envelope: str) -> dict:
    rendered = envelope.format(payload=json.dumps(payload, ensure_ascii=False, indent=2))
    return {
        "messages": [
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": rendered},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": tool_name,
                "description": "Submit the stage output object. Call this exactly once.",
                "parameters": schema,
            },
        }],
        "tool_choice": {"type": "function", "function": {"name": tool_name}},
    }


def call(config: dict, body: dict, timeout: int) -> dict:
    url = config["base_url"].rstrip("/") + "/chat/completions"
    api_key = os.environ.get(config["api_key_env"], "")
    if not api_key:
        raise RuntimeError(f"environment variable {config['api_key_env']} is not set")
    request_body = {**body, "model": config["model_id"], config.get("token_param", "max_tokens"): int(config.get("max_tokens", 30000))}
    if "reasoning_effort" in config:
        request_body["reasoning_effort"] = config["reasoning_effort"]
    if "enable_thinking" in config:
        request_body["enable_thinking"] = bool(config["enable_thinking"])
    for key in ("temperature", "top_p"):
        if key in config:
            request_body[key] = float(config[key])
    request = urllib.request.Request(
        url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')[:500]}") from exc


def extract_tool_call(payload: dict, tool_name: str) -> dict:
    """Pull the forced tool call's arguments out of an OpenAI-style response."""
    choice = (payload.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") == tool_name:
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                return json.loads(arguments)
            if isinstance(arguments, dict):
                return arguments
    content = message.get("content")
    if content:
        # Some providers return the JSON object as plain content when the tool call is forced.
        stripped = content.strip()
        if stripped.startswith("{"):
            return json.loads(stripped)
    raise RuntimeError(f"no `{tool_name}` tool call in the response")



# The Extended tool submits semantic fields only. The fixed metadata fields remain
# in the full on-disk output contract and are deterministically bound after the call.
EXTENDED_FIXED = (
    "protocol_id", "params_id", "params_sha256", "composite_score", "ranking",
)
# The Extended scoring contract requires these values to be copied by the judge.
EXTENDED_COPIED = ("blind_document_id", "article_sha256", "judge_model")


def tool_output_schema(full_schema: dict, task: str, tool_name: str) -> dict:
    schema = copy.deepcopy(full_schema)
    if task == "extended" and tool_name == "submit_long_judge_scoring":
        for field in EXTENDED_FIXED:
            schema["properties"].pop(field, None)
        schema["required"] = [k for k in schema["required"] if k not in EXTENDED_FIXED]
    return schema


def _evidence_map(payload: dict) -> dict:
    evidence = {}
    for unit in payload.get("evidence_units", []):
        eid = unit["evidence_id"]
        if eid in evidence:
            raise ValueError(f"duplicate evidence ID {eid}")
        evidence[eid] = unit
    return evidence


def validate_input_evidence(task: str, payload: dict) -> None:
    """Reject future or invalid D1 allowlist evidence before sending a scoring call."""
    evidence = _evidence_map(payload)
    if not evidence:
        raise ValueError("missing evidence units")
    events = payload["event_manifest"]["events"] if task == "extended" else payload["events"]
    article_source = "ARTICLE" if task == "extended" else "GENERATED_TEXT"
    for eid, unit in evidence.items():
        source = unit["source"]
        if source not in ("VISIBLE_PROMPT", article_source):
            raise ValueError(f"{eid}: invalid evidence source {source}")
        if (eid.startswith("P")) != (source == "VISIBLE_PROMPT"):
            raise ValueError(f"{eid}: evidence ID/source mismatch")
    if task == "extended":
        validate_evidence_references(payload["evidence_manifest"]["frozen_evidence"], set(evidence))
    for event in events:
        start = event["start_codepoint"]
        for eid in event["d1_allowed_evidence_ids"]:
            if eid not in evidence:
                raise ValueError(f"{event['event_id']}: unknown D1 allowlist ID {eid}")
            unit = evidence[eid]
            if unit["source"] == article_source:
                lo = unit.get("start_codepoint", unit.get("start_char"))
                hi = unit.get("end_codepoint", unit.get("end_char"))
                if lo is None or hi is None or lo > hi or hi > start:
                    raise ValueError(f"{event['event_id']}: future or unbounded D1 evidence {eid}")


def validate_judge_output(task: str, payload: dict, output: dict) -> None:
    """Fail closed on frozen event identity, known citations and D1 time validity."""
    if task == "extended":
        challenges = output.get("manifest_challenges", {})
        if any(challenges.get(name, {}).get("raised")
               for name in ("event_manifest", "evidence_manifest")):
            raise ManifestChallengeError(challenges)
        for key in ("blind_document_id", "article_sha256"):
            if output.get(key) != payload.get(key):
                raise ValueError(f"Extended scoring output has incorrect {key}")
        if output.get("judge_model") not in ("gpt-5.6-sol", "deepseek-v4-flash"):
            raise ValueError("Extended scoring output has invalid judge_model")
        events = payload["event_manifest"]["events"]
        judgments = output["event_judgments"]
    else:
        events = payload["events"]
        judgments = output["event_evaluations"]
    event_ids = [event["event_id"] for event in events]
    if not event_ids or event_ids != [item["event_id"] for item in judgments]:
        raise ValueError("scoring output must match the frozen event list, in order")
    validate_input_evidence(task, payload)
    evidence = _evidence_map(payload)
    for event, judgment in zip(events, judgments):
        allowed = set(event["d1_allowed_evidence_ids"])
        if task == "extended":
            for gate in ("role", "world"):
                for eid in judgment["d1"][gate]["evidence_ids"]:
                    if eid not in allowed:
                        raise ValueError(f"{event['event_id']}: D1 {gate} citation {eid} outside event-time allowlist")
            for dimension in ("d2_overall", "d3_overall"):
                score = judgment[dimension]["score"]
                if type(score) is not int or not 1 <= score <= 5:
                    raise ValueError(f"{event['event_id']}: invalid {dimension} score")
        else:
            for eid in judgment["evidence_ids"]:
                if eid not in evidence:
                    raise ValueError(f"{event['event_id']}: unknown Focused event evidence ID {eid}")
            for dimension in ("d2", "d3"):
                for score in judgment[dimension].values():
                    if score in ("NA", "IE"):
                        continue
                    if type(score) is not int or not 1 <= score <= 5:
                        raise ValueError(f"{event['event_id']}: invalid {dimension} sub-score")
    if task == "extended":
        expected = {"M-F1", "M-F2A", "M-F2B", "M-F3.1", "M-F3.2", "M-F3.3", "M-F3.4"}
        whole = output["whole_text_dimensions"]
        if len(whole) != 7 or {item["dimension_id"] for item in whole} != expected:
            raise ValueError("Extended whole-text dimensions must occur once each")
        for item in whole:
            if type(item["score"]) is not int or not 1 <= item["score"] <= 5:
                raise ValueError("whole-text score is outside 1-5")
        # Quote strings must match the immutable evidence catalogue exactly.
        def walk(value):
            if isinstance(value, dict):
                if "evidence_ids" in value:
                    ids = value["evidence_ids"]
                    if any(eid not in evidence for eid in ids):
                        raise ValueError("unknown evidence ID in Extended scoring output")
                    if value.get("evidence_strings") != [evidence[eid]["text"] for eid in ids]:
                        raise ValueError("evidence_strings must match frozen evidence text")
                for key, inner in value.items():
                    if key == "prompt_constraint_evidence_ids":
                        if any(eid not in evidence for eid in inner):
                            raise ValueError("unknown evidence ID in Extended scoring output")
                    elif key not in ("evidence_ids", "evidence_strings"):
                        walk(inner)
            elif isinstance(value, list):
                for inner in value:
                    walk(inner)
        walk(output)
    else:
        validate_focused_output(payload, output, {e["event_id"]: e for e in events}, evidence)


FOCUSED_OUTPUT_PROTOCOL_ID = read_json(
    Path(__file__).resolve().parent / "schemas/focused_scoring_output_schema.json"
)["$defs"]["RunMetadata"]["properties"]["protocol_id"]["const"]


def validate_focused_output(payload, output, by_id, evidence):
    metadata = output.get("metadata", {})
    for name in ("run_id", "prompt_id", "text_id", "params_id", "params_sha256",
                 "vocabulary_id", "vocabulary_sha256"):
        if metadata.get(name) != payload.get(name):
            raise ValueError(f"Focused metadata mismatch: {name}")
    if metadata.get("protocol_id") != FOCUSED_OUTPUT_PROTOCOL_ID:
        raise ValueError("Focused output protocol mismatch")
    expected = {}
    for event in output["event_evaluations"]:
        eid = event["event_id"]
        for d in ("role", "world"):
            if event["d1"][d] == "FAIL":
                expected[("EVENT", eid, "D1." + d)] = "FAIL"
        for group in ("d2", "d3"):
            for dim, score in event[group].items():
                if score in (1, 5, "NA", "IE"):
                    expected[("EVENT", eid, dim)] = score
    literary = output["literary_quality"]["dimensions"]
    for dim, score in literary.items():
        if score in (1, 5):
            expected[("LITERARY", None, dim)] = score
    found = set()
    for entry in output["exceptions"]:
        key = (entry.get("scope"), entry.get("event_id"), entry.get("dimension"))
        if key not in expected or entry.get("value") != expected[key] or key in found:
            raise ValueError(f"unmatched or duplicate exception: {key}")
        found.add(key)
        citations = entry.get("evidence_ids", [])
        if not isinstance(citations, list) or any(eid not in evidence for eid in citations):
            raise ValueError(f"unknown exception evidence: {key}")
        if key[2] in ("D1.role", "D1.world"):
            if not citations or any(eid not in by_id[key[1]]["d1_allowed_evidence_ids"] for eid in citations):
                raise ValueError(f"{key[1]}: D1 FAIL requires event-time evidence")
        if key[0] == "LITERARY" and entry["value"] == 5:
            loss = entry.get("counterfactual_loss")
            if not isinstance(loss, str) or not loss.strip():
                raise ValueError(f"{key[2]}: L5-point score needs counterfactual_loss")
    if found != set(expected):
        raise ValueError(f"missing exceptions: {sorted(set(expected) - found)}")
    needed = {dim for dim, score in literary.items() if score in (4, 5)}
    found = set()
    article = payload["normalized_story_text"]
    for entry in output["literary_quality"]["high_score_evidence"]:
        dim = entry["dimension"]
        if dim not in needed or dim in found:
            raise ValueError(f"extra or duplicate high-score evidence: {dim}")
        found.add(dim)
        ids = entry["evidence_ids"]
        if len(ids) != 2 or ids[0] == ids[1]:
            raise ValueError(f"{dim}: two different evidence IDs required")
        spans = []
        for eid in ids:
            unit = evidence.get(eid)
            if unit is None or unit.get("source") != "GENERATED_TEXT" or not eid.startswith("T"):
                raise ValueError(f"{dim}: invalid narrative evidence ID {eid}")
            lo, hi = unit.get("start_char"), unit.get("end_char")
            if (type(lo) is not int or type(hi) is not int or
                    not 0 <= lo < hi <= len(article) or article[lo:hi] != unit.get("text")):
                raise ValueError(f"{dim}: invalid text evidence span {eid}")
            spans.append((lo, hi))
        if not (spans[0][1] <= spans[1][0] or spans[1][1] <= spans[0][0]):
            raise ValueError(f"{dim}: evidence positions overlap")
    if found != needed:
        raise ValueError(f"missing high-score evidence: {sorted(needed - found)}")


def bind_extended_metadata(full_schema: dict, payload: dict, output: dict, model: str) -> dict:
    if any(key in output for key in EXTENDED_FIXED):
        raise ValueError("model returned a runner-owned Extended metadata field")
    for key, expected in (
        ("blind_document_id", payload["blind_document_id"]),
        ("article_sha256", payload["article_sha256"]),
        ("judge_model", model),
    ):
        if output.get(key) != expected:
            raise ValueError(f"Extended Judge must copy {key} exactly from its scoring request")
    fixed = {
        "protocol_id": full_schema["properties"]["protocol_id"]["const"],
        "params_id": payload["params_id"],
        "params_sha256": payload["params_sha256"],
        "composite_score": None,
        "ranking": None,
    }
    for key in ("protocol_id", "params_id"):
        expected = full_schema["properties"][key].get("const")
        if expected is not None and fixed[key] != expected:
            raise ValueError(f"{key} does not match the output protocol")
    allowed = full_schema["properties"]["judge_model"].get("enum", [])
    if model not in allowed:
        raise ValueError(f"judge_model {model} is not authorized by scoring schema")
    return {**fixed, **output}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", type=Path, required=True, help="frozen prompt file (system message)")
    parser.add_argument("--schema", type=Path, required=True, help="frozen output schema (tool parameters)")
    parser.add_argument("--tool-name", required=True, help="forced tool/function name declared by the prompt")
    parser.add_argument("--input", type=Path, required=True, help="stage input JSON to wrap in the user message")
    parser.add_argument("--task", choices=["focused", "extended"], required=True, help="selects the input envelope")
    parser.add_argument("--config", type=Path, help="model config JSON (required unless --dry-run)")
    parser.add_argument("--model", help="model_id to use from the config")
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--dry-run", action="store_true", help="print the request without sending it")
    args = parser.parse_args(argv)

    prompt_text = args.prompt.read_text(encoding="utf-8-sig")
    schema = read_json(args.schema)
    payload = read_json(args.input)
    if args.tool_name in ("submit_judge_output", "submit_long_judge_scoring"):
        validate_input_evidence(args.task, payload)
    body = build_body(prompt_text, tool_output_schema(schema, args.task, args.tool_name), args.tool_name, payload, INPUT_ENVELOPE[args.task])

    if args.dry_run:
        summary = {
            "prompt": args.prompt.as_posix(),
            "prompt_sha256": hashlib.sha256(args.prompt.read_bytes()).hexdigest(),
            "schema": args.schema.as_posix(),
            "schema_sha256": hashlib.sha256(args.schema.read_bytes()).hexdigest(),
            "tool_name": args.tool_name,
            "messages": [{"role": m["role"], "characters": len(m["content"])} for m in body["messages"]],
            "system_message_is_prompt_file": body["messages"][0]["content"] == prompt_text,
            "user_message_is_wrapped_input": True,
            "tool_choice": body["tool_choice"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if not args.config or not args.model:
        parser.error("--config and --model are required unless --dry-run")
    config = next((m for m in read_json(args.config)["models"] if m["model_id"] == args.model), None)
    if config is None:
        parser.error(f"{args.model} is not in {args.config}")
    if "<" in config["base_url"]:
        parser.error("base_url still holds a placeholder; edit your config first")

    response = call(config, body, args.timeout)
    output = extract_tool_call(response, args.tool_name)
    output_schema = tool_output_schema(schema, args.task, args.tool_name)
    from jsonschema import Draft202012Validator
    Draft202012Validator(output_schema).validate(output)
    if args.tool_name == "submit_discovery_output":
        freeze_focused(payload["normalized_story_text"], payload["candidates"], output,
                       expected_identity=payload)
    if args.tool_name in ("submit_judge_output", "submit_long_judge_scoring"):
        try:
            validate_judge_output(args.task, payload, output)
        except ManifestChallengeError as exc:
            args.out.mkdir(parents=True, exist_ok=True)
            input_sha = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()
            challenge = args.out / f"{args.tool_name}__{args.model}__{input_sha[:16]}__challenge.json"
            challenge.write_text(json.dumps({"model": args.model, "tool": args.tool_name,
                "input_sha256": input_sha, "manifest_challenges": exc.challenges},
                ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            raise
    if args.task == "extended" and args.tool_name == "submit_long_judge_scoring":
        output = bind_extended_metadata(schema, payload, output, args.model)
        Draft202012Validator(schema).validate(output)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / f"{args.tool_name}__{args.model}__{hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()[:16]}.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
