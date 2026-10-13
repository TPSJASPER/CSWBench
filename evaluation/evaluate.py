#!/usr/bin/env python3
"""CSWEval model event discovery, evidence construction and dual judging.

Prepare mode makes no network calls. --mode live uses the configured endpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evidence_catalog import catalog, evidence_request, validate_semantic, canonical_sha
from pipeline_inputs import (digest, focused_candidates, extended_candidates,
                             registry_from_prompt, validate_scene_order,
                             freeze_focused, freeze_extended,
                             build_focused_scoring_input, build_extended_scoring_input,
                             specialize_schema)
from judge_adapter import to_aggregation
from run_stage import ManifestChallengeError

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
SCHEMA = EVAL / "schemas"
SOL = "gpt-5.6-sol"
DEEPSEEK = "deepseek-v4-flash"


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def immutable_json(path, data):
    """Never overwrite a frozen stage artifact or reassign its identity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != data:
            raise ValueError(f"immutable stage artifact changed: {path}")
    else:
        with path.open("x", encoding="utf-8") as fd:
            fd.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    return data


def check_schema(schema, data):
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("install jsonschema>=4.21 to validate full stage contracts") from exc
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(data))
    if errors:
        first = min(errors, key=lambda x: (len(list(x.path)), str(x)))
        raise ValueError("stage JSON Schema validation failed at " +
                         "/".join(map(str, first.path)) + ": " + first.message)


def read_card(task, prompt_id):
    with (ROOT / "data" / f"{task}.jsonl").open(encoding="utf-8") as fd:
        for line in fd:
            record = json.loads(line)
            if record["prompt_id"] == prompt_id and record["task_type"] == {"focused": "focused_narrative", "extended": "extended_narrative"}[task]:
                return record["prompt_text"]
    raise ValueError("prompt_id not found in the published Prompt Cards")


class LiveTransport:
    def __init__(self, path, attempts=2, timeout=1200):
        import run_stage
        self.stage = run_stage
        configs = read_json(path)["models"]
        self.config = {c["model_id"]: c for c in configs}
        if not {SOL, DEEPSEEK} <= set(self.config):
            raise ValueError("judges config must contain both supported judge models")
        if attempts < 1:
            raise ValueError("retry attempts must be at least one")
        self.attempts = attempts
        self.timeout = timeout

    def run(self, name, model, payload, prompt, schema, tool, task):
        if model not in (SOL, DEEPSEEK):
            raise ValueError("only supported judge models may be called")
        s = self.stage
        visible_schema = s.tool_output_schema(schema, task, tool)
        body = s.build_body(prompt, visible_schema, tool, payload, s.INPUT_ENVELOPE[task])
        err = None
        for n in range(self.attempts):
            try:
                response = s.call(self.config[model], body, self.timeout)
                result = s.extract_tool_call(response, tool)
                check_schema(visible_schema, result)
                if tool == "submit_discovery_output":
                    freeze_focused(payload["normalized_story_text"], payload["candidates"], result,
                                   expected_identity=payload)
                elif tool == "submit_long_event_discovery":
                    freeze_extended(payload["article_text"], payload["mechanical_candidates"], result)
                if tool == "submit_long_evidence_semantic_payload":
                    validate_semantic(result, payload["evidence_catalog"])
                if tool in ("submit_judge_output", "submit_long_judge_scoring"):
                    s.validate_input_evidence(task, payload)
                    s.validate_judge_output(task, payload, result)
                if task == "extended" and tool == "submit_long_judge_scoring":
                    result = s.bind_extended_metadata(schema, payload, result, model)
                    check_schema(schema, result)
                return result
            except ManifestChallengeError:
                raise
            except (ValueError, KeyError, RuntimeError, OSError, json.JSONDecodeError) as exc:
                err = exc
                if n + 1 == self.attempts:
                    break
                time.sleep(min(2**n, 4))
        raise RuntimeError(f"stage {name}/{model} failed after {self.attempts} attempt(s): {err}") from err


class FixtureTransport:
    """For offline E2E tests only; fixture outputs never imply real model calls."""
    def __init__(self, root):
        self.root = Path(root)

    def run(self, name, model, payload, prompt, schema, tool, task):
        path = self.root / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing offline fixture {path}")
        result = read_json(path)
        import run_stage as stage
        is_extended_score = task == "extended" and tool == "submit_long_judge_scoring"
        if is_extended_score and "protocol_id" not in result:
            check_schema(stage.tool_output_schema(schema, task, tool), result)
            stage.validate_judge_output(task, payload, result)
            result = stage.bind_extended_metadata(schema, payload, result, model)
        check_schema(schema, result)
        if tool in ("submit_judge_output", "submit_long_judge_scoring"):
            stage.validate_input_evidence(task, payload)
            stage.validate_judge_output(task, payload, result)
        return result


def _semantic_schema(schema):
    schema = json.loads(json.dumps(schema))
    if "properties" in schema:
        for k in ("protocol_id", "params_id", "params_sha256", "composite_score", "ranking"):
            schema["properties"].pop(k, None)
            if k in schema.get("required", []): schema["required"].remove(k)
    return schema


class Workflow:
    def __init__(self, *, task, prompt_id, article, workdir, transport, article_id,
                 generation_model, registry=None, blind_id=None, text_id=None):
        self.task, self.prompt_id, self.workdir = task, prompt_id, Path(workdir)
        self.article = Path(article).read_text(encoding="utf-8")
        self.prompt = read_card(task, prompt_id)
        self.model = generation_model
        self.article_id = article_id
        self.registry = (read_json(registry) if isinstance(registry, (str, Path))
                         else registry_from_prompt(self.prompt) if task == "extended" and registry is None
                         else registry)
        self.blind_id = blind_id
        self.text_id = text_id or article_id
        self.transport = transport
        if task not in ("focused", "extended"):
            raise ValueError("unsupported CSWEval task")
        if task == "extended":
            if not isinstance(blind_id, str) or not __import__("re").fullmatch(r"LJ-B\d{4}", blind_id):
                raise ValueError("Extended requires an explicit blind ID LJ-B####")
            if not isinstance(self.registry, list) or [i.get("character_slot") for i in self.registry] != ["A", "B"]:
                raise ValueError("Extended requires an actual A/B character registry JSON")
        self.workdir.mkdir(parents=True, exist_ok=True)
        immutable_json(self.workdir / "identity.json", {
            "task":task, "prompt_id":prompt_id, "article_sha256": digest(self.article),
            "prompt_sha256":digest(self.prompt), "article_id":article_id, "generation_model":generation_model,
            "blind_id":blind_id, "text_id":self.text_id, "registry_sha256":digest(self.registry) if self.registry else None,
            "judge_models":[SOL, DEEPSEEK],
        })

    def artifact(self, filename, data):
        return immutable_json(self.workdir / filename, data)

    def run_stage(self, name, model, payload, prompt_name, schema_name, tool, *, specialized=None):
        ipath = name + "_input.json"
        self.artifact(ipath, payload)
        prompt = (EVAL / "prompts" / prompt_name).read_text(encoding="utf-8")
        schema = specialized or read_json(SCHEMA / schema_name)
        filename = self.workdir / (name + "_output.json")
        receipt_file = self.workdir / (name + "_receipt.json")
        binding = {"stage": name, "task": self.task, "model": model, "tool": tool,
                   "input_sha256": digest(payload), "prompt_sha256": digest(prompt),
                   "schema_sha256": digest(schema)}
        if filename.exists():
            if not receipt_file.exists():
                raise ValueError(f"missing immutable execution receipt for cached stage: {name}")
            result = read_json(filename)
            receipt = read_json(receipt_file)
            if any(receipt.get(k) != v for k, v in binding.items()) or receipt.get("output_sha256") != digest(result):
                raise ValueError(f"cached {name} stage binding/identity was modified")
        else:
            if self.transport is None:
                raise RuntimeError("model execution needs --mode live or a test transport")
            try:
                result = self.transport.run(name, model, payload, prompt, schema, tool, self.task)
            except ManifestChallengeError as exc:
                self.artifact(name + "_challenge.json", {**binding, "manifest_challenges": exc.challenges})
                raise
            receipt = {**binding, "output_sha256": digest(result),
                       "execution_mode": type(self.transport).__name__}
        # Both cached and newly generated records must satisfy the current
        # stage's canonical schema; a stale or tampered replay never passes.
        expected = schema if tool != "submit_long_judge_scoring" else (
            schema if "protocol_id" in result else _semantic_schema(schema))
        check_schema(expected, result)
        if name == "discovery" and self.task == "focused":
            freeze_focused(payload["normalized_story_text"], payload["candidates"], result,
                           expected_identity=payload)
        elif name == "evidence":
            validate_semantic(result, payload["evidence_catalog"])
        elif name in ("judge_gpt", "judge_deepseek"):
            import run_stage as stage
            stage.validate_input_evidence(self.task, payload)
            try:
                stage.validate_judge_output(self.task, payload, result)
            except ManifestChallengeError as exc:
                self.artifact(name + "_challenge.json", {**binding, "manifest_challenges": exc.challenges})
                raise
            if self.task == "extended" and result.get("judge_model") != model:
                raise ValueError("quality Judge route/response identity mismatch")
        self.artifact(name + "_output.json", result)
        self.artifact(name + "_receipt.json", receipt)
        return result

    def run(self, *, offline_prepare=False, aggregate=True):
        if self.task == "focused":
            candidates = focused_candidates(self.article)
            if not candidates: raise ValueError("no Latin-script candidates")
            count = sum(not c.isspace() for c in self.article)
            discovery_in = {
                "run_id":f"focused-{self.text_id}", "prompt_id":self.prompt_id, "text_id":self.text_id,
                "protocol_id":"csweval_focused_discovery", "visible_prompt":self.prompt,
                "normalized_story_text":self.article, "normalized_story_text_sha256":digest(self.article),
                "candidates":candidates,
                "length_audit":{"formula_id":"non_whitespace_unicode_codepoints",
                    "target":700,"observed_count":count},
                "params_id":"csweval_focused_scoring_params", "params_sha256":digest({"task":"focused","rubric":"focused_scoring"}),
            }
            d_schema = "focused_discovery_output_schema.json"
            sc = specialize_schema(read_json(SCHEMA / d_schema), candidate_ids=[x["candidate_id"] for x in candidates])
            d_prompt="focused_event_discovery_prompt.md"
            d_tool="submit_discovery_output"
        else:
            candidates=extended_candidates(self.article)
            if not candidates: raise ValueError("no Latin-script candidates")
            discovery_in={"protocol_id":"csweval_extended_event_discovery_input",
                "blind_document_id":self.blind_id, "article_sha256":digest(self.article),
                "article_text":self.article, "prompt_constraints":{"visible_prompt":self.prompt},
                "character_registry":self.registry,
                "mechanical_candidates":candidates}
            d_schema="extended_event_discovery_output_schema.json"
            sc=None
            d_prompt="extended_event_discovery_prompt.md"
            d_tool="submit_long_event_discovery"
        self.artifact("discovery_input.json",discovery_in)
        if offline_prepare:
            return {"status":"READY_TO_DISCOVER","path":str(self.workdir/"discovery_input.json")}
        discovery=self.run_stage("discovery",SOL,discovery_in,d_prompt,d_schema,d_tool,specialized=sc)
        if self.task == "extended":
            if discovery.get("judge_model") != SOL or discovery.get("blind_document_id") != self.blind_id or discovery.get("article_sha256") != digest(self.article):
                raise ValueError("Extended discovery returned incompatible Sol/document identity")
            if discovery.get("manifest_challenge",{}).get("raised"):
                raise ValueError("Sol discovery flagged a manifest challenge; manual resolution required")
        else:
            m=discovery.get("metadata",{})
            if any(m.get(k)!=discovery_in[k] for k in ("run_id","prompt_id","text_id","params_id","params_sha256")):
                raise ValueError("Focused discovery metadata does not match its frozen input")
        check_schema(sc or read_json(SCHEMA / d_schema), discovery)
        if self.task == "focused":
            freeze = freeze_focused(self.article, candidates, discovery, expected_identity=discovery_in)
            ids = [item["candidate_id"] for item in discovery["candidate_reviews"]]
        else:
            freeze = freeze_extended(self.article, candidates, discovery)
            ids = [item["candidate_id"] for item in discovery["candidate_assessments"]]
        if set(ids) != {c["candidate_id"] for c in candidates} or len(ids) != len(candidates):
            raise ValueError("candidate decisions must cover every extracted candidate exactly once")
        self.artifact("frozen_events.json",freeze)
        if self.task=="focused":
            scored_in=build_focused_scoring_input(self.prompt,self.article,discovery,
                prompt_id=self.prompt_id,text_id=self.text_id,params_id="csweval_focused_scoring_params",
                params_sha256=digest({"task":"focused","rubric":"focused_scoring"}),
                vocabulary_id="csweval_focused_rubric",vocabulary_sha256=digest({"D2":list(range(1,9)),"D3":list(range(1,6))}),
                )
            s_schema="focused_scoring_output_schema.json"
            s_special=specialize_schema(read_json(SCHEMA/s_schema),
                event_ids=[e["event_id"] for e in scored_in["events"]],
                evidence_ids=[u["evidence_id"] for u in scored_in["evidence_units"]],
                generated_ids=[u["evidence_id"] for u in scored_in["evidence_units"] if u["evidence_id"].startswith("T")])
            score_prompt="focused_scoring_prompt.md"
            score_tool="submit_judge_output"
            frozen_judge_events=scored_in["events"]
        else:
            ecatalog, units = catalog(self.prompt,self.article)
            self.artifact("frozen_evidence_catalog.json",ecatalog)
            self.artifact("frozen_evidence_units.json",units)
            evidence_in=evidence_request(self.prompt,self.article,self.registry,freeze,ecatalog)
            self.artifact("evidence_input.json",evidence_in)
            evidence_out=self.run_stage("evidence",SOL,evidence_in,
                "extended_evidence_prompt.md","extended_evidence_semantic_payload_schema.json",
                "submit_long_evidence_semantic_payload")
            if evidence_out.get("event_manifest_challenge",{}).get("raised"):
                raise ValueError("Sol evidence builder challenged frozen event manifest: manual review required")
            evidence_out=validate_semantic(evidence_out,ecatalog)
            validate_scene_order(freeze, evidence_out)
            self.artifact("frozen_semantic_evidence.json",evidence_out)
            params_path=EVAL/"params"/"extended_scoring_params.json"
            params=read_json(params_path)
            scored_in=build_extended_scoring_input(self.prompt,self.article,self.registry,freeze,
                evidence_out,params,self.blind_id,
                params_file_sha256=hashlib.sha256(params_path.read_bytes()).hexdigest(),
                frozen_evidence_units=units)
            s_schema="extended_scoring_output_schema.json"
            s_special=None
            score_prompt="extended_scoring_prompt.md"
            score_tool="submit_long_judge_scoring"
            frozen_judge_events=scored_in["event_manifest"]["events"]
        if self.task == "extended":
            check_schema(read_json(SCHEMA / "extended_scoring_input_schema.json"), scored_in)
        self.artifact("scoring_input.json",scored_in)
        # Both judges operate on a single persisted, frozen input object.
        output_g=self.run_stage("judge_gpt",SOL,scored_in,score_prompt,s_schema,score_tool,specialized=s_special)
        output_d=self.run_stage("judge_deepseek",DEEPSEEK,scored_in,score_prompt,s_schema,score_tool,specialized=s_special)
        import run_stage as stage
        records=[]
        for model,output in ((SOL,output_g),(DEEPSEEK,output_d)):
            # Validator is applied even to replayed/cached outputs (fail closed).
            stage.validate_input_evidence(self.task,scored_in)
            stage.validate_judge_output(self.task,scored_in,output)
            if self.task=="extended" and output.get("judge_model") not in (None,model):
                raise ValueError("inconsistent judge route/identity")
            records.append(to_aggregation(self.task,output,article_id=self.article_id,
                prompt_card_id=self.prompt_id,generation_model=self.model,
                judge_model=model,frozen_events=frozen_judge_events))
        self.artifact("scoring_records.json",records)
        lines="".join(json.dumps(x,ensure_ascii=False,separators=(",",":"))+"\n" for x in records)
        scores_path=self.workdir/"judge_records.jsonl"
        if scores_path.exists() and scores_path.read_text(encoding="utf-8")!=lines:
            raise ValueError("immutable judge-record log changed")
        if not scores_path.exists():scores_path.write_text(lines,encoding="utf-8")
        if aggregate:
            aggregator=ROOT/"aggregation"/"aggregate.py"
            subprocess.run([sys.executable,str(aggregator),str(scores_path),"--out",str(self.workdir/"results")],check=True)
        return {"status":"COMPLETE","records":str(scores_path),"results":str(self.workdir/"results") if aggregate else None}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task",choices=("focused","extended"),required=True)
    p.add_argument("--prompt-id",required=True)
    p.add_argument("--article",type=Path,required=True)
    p.add_argument("--article-id",required=True)
    p.add_argument("--generation-model",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--config",type=Path,help="per-model endpoint/key-env configuration")
    p.add_argument("--mode",choices=("prepare","live"),default="prepare")
    p.add_argument("--registry",type=Path,help="Extended: source-grounded A/B characters")
    p.add_argument("--blind-id",help="Extended: blind document ID LJ-B####")
    p.add_argument("--text-id",help="Focused text identifier")
    p.add_argument("--retries",type=int,default=2)
    a=p.parse_args(argv)
    if a.mode=="live" and not a.config:
        p.error("--mode live requires --config with actual API endpoint variables")
    transport=LiveTransport(a.config,a.retries) if a.mode=="live" else None
    wf=Workflow(task=a.task,prompt_id=a.prompt_id,article=a.article,workdir=a.output,
        transport=transport,article_id=a.article_id,generation_model=a.generation_model,
        registry=a.registry,blind_id=a.blind_id,text_id=a.text_id)
    print(json.dumps(wf.run(offline_prepare=a.mode=="prepare"),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
