# CSWEval evaluation protocol

CSWEval evaluates two creative-writing tasks using a shared event-level and whole-text rubric. English expressions, switching events, and their narrative positions are model-selected.

## Evaluation stages

| Task | Stage | Prompt | Output schema |
| --- | --- | --- | --- |
| Focused | GPT-5.6-Sol event discovery | `prompts/focused_event_discovery_prompt.md` | `schemas/focused_discovery_output_schema.json` |
| Focused | GPT-5.6-Sol / DeepSeek-V4-Flash scoring | `prompts/focused_scoring_prompt.md` | `schemas/focused_scoring_output_schema.json` |
| Extended | GPT-5.6-Sol event discovery | `prompts/extended_event_discovery_prompt.md` | `schemas/extended_event_discovery_output_schema.json` |
| Extended | GPT-5.6-Sol narrative evidence construction | `prompts/extended_evidence_prompt.md` | `schemas/extended_evidence_semantic_payload_schema.json` |
| Extended | GPT-5.6-Sol / DeepSeek-V4-Flash scoring | `prompts/extended_scoring_prompt.md` | `schemas/extended_scoring_output_schema.json` |

The event discoverer classifies mechanical Latin-script candidates and groups active English usage into semantic events. **The program freezes and validates the model-produced event manifest** before quality judging.

Extended uses a **single chronological scene array**, with the number of scenes determined by the article itself. The schema requires a nonempty timeline but imposes no fixed upper bound or artificial caps on world facts and turning points. The frozen evidence is shared by both quality judges. The scoring code checks event and evidence identities, citations and D1 temporal cutoffs.

The evidence catalog covers all non-whitespace Prompt and article text, including lines without sentence-final punctuation, and preserves original Unicode code-point offsets. The shared frozen evidence contains Prompt-expected facts, article-observed facts, the narrative system, ordered A/B character profiles, and ambiguities. Every structured citation must resolve to the same frozen catalog; D1 citations remain restricted to each event's temporal allowlist.

## Scoring and aggregation

- **D1:** both participant/role and world licensing must pass; article D1 is five times the fraction of scorable events passing both.
- **Focused D2/D3:** average applicable subdimensions within each event, then take the salience-weighted event mean.
- **Extended D2/D3:** one holistic 1–5 judgment per event, salience-weighted.
- **Salience weights:** S1=1, S2=2, S3=3, applying to D2/D3 only.
- **Whole-text:** Focused uses L1–L5, whose dimension definitions and anchors are embedded in `prompts/focused_scoring_prompt.md`; Extended uses M-F1, M-F2A, M-F2B, and M-F3.1–M-F3.4.
- **Overall:** D1+D2+D3+Whole (/20). The two judges' scores are averaged at article level.
