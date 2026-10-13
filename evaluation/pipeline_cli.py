"""Offline preparation for CSWEval discovery, evidence and judging stages.

Uses exactly the Prompt Cards in data/*.jsonl.
Use evaluation/run_stage.py to execute each prepared stage with authorized endpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pipeline_inputs import (digest, focused_candidates, extended_candidates,
                             registry_from_prompt,
                             freeze_extended, build_focused_scoring_input,
                             build_extended_scoring_input, specialize_schema, validate_scene_order)
from evidence_catalog import catalog, validate_semantic
from evaluate import check_schema
from judge_adapter import to_aggregation

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / 'evaluation' / 'schemas'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {path}')


def prompt_card(task, prompt_id):
    path = ROOT / 'data' / f'{task}.jsonl'
    with path.open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            if row['prompt_id'] == prompt_id:
                if row['task_type'] != {'focused': 'focused_narrative', 'extended': 'extended_narrative'}[task]:
                    raise ValueError('prompt task type mismatch')
                return row
    raise ValueError(f'prompt ID {prompt_id} not present in the public {task} Prompt Cards')


def prepare_focused(args):
    card = prompt_card('focused', args.prompt_id)
    story = Path(args.article).read_text(encoding='utf-8')
    candidates = focused_candidates(story)
    if not candidates: raise ValueError('no Latin candidates; cannot discover a CS event')
    length = sum(not c.isspace() for c in story)
    payload = {
        'run_id': args.run_id, 'prompt_id': args.prompt_id, 'text_id': args.text_id,
        'protocol_id': 'csweval_focused_discovery',
        'visible_prompt': card['prompt_text'],
        'normalized_story_text': story, 'normalized_story_text_sha256': digest(story),
        'candidates': candidates,
        'length_audit': {'formula_id': 'non_whitespace_unicode_codepoints',
                         'target': card['length_target_codepoints'],
                         'observed_count': length},
        'params_id': args.params_id, 'params_sha256': args.params_sha256}
    save(args.output, payload)
    if args.schema_output:
        schema = load(SCHEMAS / 'focused_discovery_output_schema.json')
        save(args.schema_output, specialize_schema(schema,candidate_ids=[x['candidate_id'] for x in candidates]))


def prepare_extended(args):
    card = prompt_card('extended', args.prompt_id)
    story = Path(args.article).read_text(encoding='utf-8')
    registry = load(args.registry) if args.registry else registry_from_prompt(card['prompt_text'])
    if not isinstance(registry, list) or [x.get('character_slot') for x in registry] != ['A', 'B']:
        raise ValueError('expected a source-grounded A/B registry')
    candidates = extended_candidates(story)
    if not candidates: raise ValueError('no Latin candidates; cannot discover a CS event')
    payload = {'protocol_id':'csweval_extended_event_discovery_input',
        'blind_document_id':args.blind_id,'article_sha256':digest(story),'article_text':story,
        'prompt_constraints':{'visible_prompt':card['prompt_text']}, 'character_registry':registry,
        'mechanical_candidates':candidates}
    save(args.output,payload)


def focused_judge(args):
    card=prompt_card('focused',args.prompt_id)
    story=Path(args.article).read_text(encoding='utf-8')
    result=build_focused_scoring_input(card['prompt_text'],story,load(args.discovery),
          prompt_id=args.prompt_id,text_id=args.text_id,params_id=args.params_id,
          params_sha256=args.params_sha256,vocabulary_id=args.vocabulary_id,
          vocabulary_sha256=args.vocabulary_sha256)
    save(args.output,result)
    if args.schema_output:
        schema=load(SCHEMAS/'focused_scoring_output_schema.json')
        save(args.schema_output,specialize_schema(schema,
              event_ids=[e['event_id'] for e in result['events']],
              evidence_ids=[e['evidence_id'] for e in result['evidence_units']],
              generated_ids=[e['evidence_id'] for e in result['evidence_units'] if e['evidence_id'].startswith('T')]))


def extended_judge(args):
    card = prompt_card('extended', args.prompt_id)
    story = Path(args.article).read_text(encoding='utf-8')
    registry = load(args.registry) if args.registry else registry_from_prompt(card['prompt_text'])
    if not isinstance(registry, list) or [r.get('character_slot') for r in registry] != ['A', 'B']:
        raise ValueError('Extended requires an ordered A/B character registry')
    candidates = extended_candidates(story)
    raw = load(args.discovery)
    check_schema(load(SCHEMAS / 'extended_event_discovery_output_schema.json'), raw)
    if (raw.get('judge_model') != 'gpt-5.6-sol'
            or raw.get('blind_document_id') != args.blind_id
            or raw.get('article_sha256') != digest(story)):
        raise ValueError('Extended discovery identity does not match the Sol/document request')
    if raw.get('manifest_challenge', {}).get('raised'):
        raise ValueError('event discovery flagged a manifest challenge')
    frozen = freeze_extended(story, candidates, raw)
    decisions = [item['candidate_id'] for item in raw['candidate_assessments']]
    if len(decisions) != len(candidates) or set(decisions) != {x['candidate_id'] for x in candidates}:
        raise ValueError('discovery must cover every mechanical candidate exactly once')
    evidence = load(args.evidence)
    if evidence.get('judge_model', 'gpt-5.6-sol') != 'gpt-5.6-sol':
        raise ValueError('Extended evidence must be constructed by GPT-5.6-Sol')
    semantic = evidence.get('semantic_payload', evidence)
    check_schema(load(SCHEMAS / 'extended_evidence_semantic_payload_schema.json'), semantic)
    ecatalog, expected_units = catalog(card['prompt_text'], story)
    frozen_units = load(args.evidence_units)
    if frozen_units != expected_units:
        raise ValueError('frozen evidence units must match the deterministic Prompt/article catalog')
    if semantic.get('event_manifest_challenge', {}).get('raised'):
        raise ValueError('evidence construction flagged an event-manifest challenge')
    validated = validate_semantic(semantic, ecatalog)
    validate_scene_order(frozen, validated)
    params_path = ROOT / 'evaluation' / 'params' / 'extended_scoring_params.json'
    params = load(params_path)
    result = build_extended_scoring_input(
        card['prompt_text'], story, registry, frozen, validated, params, args.blind_id,
        frozen_evidence_units=frozen_units,
        params_file_sha256=hashlib.sha256(params_path.read_bytes()).hexdigest(),
    )
    check_schema(load(SCHEMAS / 'extended_scoring_input_schema.json'), result)
    save(args.output, result)

def export_judges(args):
    p=load(args.scoring_input)
    events = p['events'] if args.task=='focused' else p['event_manifest']['events']
    if args.task=='focused':
        expected = p['visible_prompt']
    else: expected = p['prompt_constraints']['visible_prompt']
    card=prompt_card(args.task,args.prompt_id)
    if expected != card['prompt_text']:
        raise ValueError('scoring input prompt does not exactly match public Prompt Card')
    import run_stage as stage
    records=[]
    for model,path in (('gpt-5.6-sol', args.gpt),('deepseek-v4-flash',args.deepseek)):
        output=load(path)
        schema_name = ('extended_scoring_output_schema.json' if args.task == 'extended'
                       else 'focused_scoring_output_schema.json')
        check_schema(load(SCHEMAS / schema_name), output)
        if args.task == 'extended' and output['judge_model'] != model:
            raise ValueError('Extended quality Judge identity mismatch')
        stage.validate_judge_output(args.task,p,output)
        records.append(to_aggregation(args.task,output,article_id=args.article_id,
                       prompt_card_id=args.prompt_id,generation_model=args.generation_model,
                       judge_model=model,frozen_events=events))
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records),encoding='utf-8')
    print(f'wrote {dest}')


def check_counts(args):
    source=Path(args.scores)
    if not source.exists(): raise FileNotFoundError(source)
    ids={'focused':set(),'extended':set()}
    with source.open(encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip(): continue
            row=json.loads(line)
            ids[row['task']].add(row['article_id'])
    counts={k:len(v) for k,v in ids.items()}
    target={'focused':4884,'extended':4800}
    print(json.dumps({'observed':counts,'declared_targets':target,'matched':counts==target},indent=2))
    if counts != target: raise SystemExit(f'article count mismatch: expected {target}, observed {counts}')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('prepare-focused','prepare-extended','focused-judge','extended-judge'):
        a=sub.add_parser(name)
        a.add_argument('--prompt-id',required=True)
        a.add_argument('--article',required=True)
        a.add_argument('--output',required=True)
        if name in ('prepare-focused','focused-judge'):
            a.add_argument('--text-id',required=True)
            a.add_argument('--params-id',required=True)
            a.add_argument('--params-sha256',required=True)
        if name=='prepare-focused':
            a.add_argument('--run-id',required=True)
        if name=='focused-judge':
            a.add_argument('--vocabulary-id',required=True)
            a.add_argument('--vocabulary-sha256',required=True)
            a.add_argument('--discovery',required=True)
        if name in ('prepare-extended','extended-judge'):
            a.add_argument('--registry',help='optional override; otherwise read from the Extended Prompt Card')
            a.add_argument('--blind-id',required=True)
        if name=='extended-judge':
            a.add_argument('--discovery',required=True)
            a.add_argument('--evidence',required=True)
            a.add_argument('--evidence-units',required=True,help='frozen catalog from the same Sol evidence-stage run')
        if name.startswith('prepare') or name=='focused-judge':
            a.add_argument('--schema-output')
    a=sub.add_parser('export-judges')
    for name in ('task','prompt-id','scoring-input','gpt','deepseek','article-id','generation-model','output'):
        a.add_argument('--'+name,required=True)
    a=sub.add_parser('check-counts');a.add_argument('--scores',required=True)
    args=p.parse_args()
    return {'prepare-focused':prepare_focused,'prepare-extended':prepare_extended,
            'focused-judge':focused_judge,'extended-judge':extended_judge,
            'export-judges':export_judges,'check-counts':check_counts}[args.command](args)

if __name__=='__main__': main()