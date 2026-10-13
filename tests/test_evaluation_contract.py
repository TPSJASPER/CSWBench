"""Event identity, citation and score admissibility contracts."""
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "tests"))
import pipeline_inputs as pipeline
import run_stage
from evidence_catalog import catalog
from evaluate import check_schema
from fixtures.synthetic import STORIES, focused_discovery, focused_score


def prepared():
    article = STORIES["focused"]
    candidates = pipeline.focused_candidates(article)
    discovery_input = {
        "run_id": "focused-fixture", "prompt_id": "synthetic-card", "text_id": "fixture",
        "protocol_id": "csweval_focused_discovery", "params_id": "params",
        "params_sha256": "a" * 64, "normalized_story_text_sha256": pipeline.digest(article),
        "candidates": candidates
    }
    discovery = focused_discovery(discovery_input)
    payload = pipeline.build_focused_scoring_input(
        "双方理解英语，并正在交谈。", article, discovery,
        prompt_id="synthetic-card", text_id="fixture",
        params_id="params", params_sha256="a" * 64,
        vocabulary_id="vocab", vocabulary_sha256="b" * 64)
    output = focused_score(payload)
    return payload, output, discovery, candidates


class EvaluationContract(unittest.TestCase):
    def test_extended_catalog_covers_unpunctuated_lines_and_preserves_offsets(self):
        prompt = " 角色甲🙂英语表达有限\n\n角色乙能够阅读英文资料 "
        articles = (
            "英语表达有限\n他说：“Hello.”\n周岚点头。",
            "首行\n\n第二行\n\n末行没有标点",
            "相同行\r\n相同行\r末尾",
            "Résumé🙂\u2028下一行\u2029\t尾段 ",
            "首行\v第二行\f第三行\x85尾段",
            "他说：“Really?!”然后离开。",
            "他问：“什么？！”\n他喊：“Wait!!”",
            "？！开头标点。；；\n!!\n末行！？",
            "Résumé🙂?!\r\n相同行！！\u2028尾段，；：",
        )
        for article in articles:
            with self.subTest(article=article):
                directory, units = catalog(prompt, article)
                self.assertEqual([entry["evidence_id"] for entry in directory["candidates"]],
                                 [unit["evidence_id"] for unit in units])
                for prefix, source in (("P", prompt), ("T", article)):
                    selected = [unit for unit in units if unit["evidence_id"].startswith(prefix)]
                    self.assertEqual(
                        "".join(char for char in source if not char.isspace()),
                        "".join(char for unit in selected for char in unit["text"]
                                if not char.isspace()))
                    self.assertEqual([unit["evidence_id"] for unit in selected],
                                     [f"{prefix}{i:04d}" for i in range(1, len(selected)+1)])
                    for unit in selected:
                        if prefix == "P":
                            self.assertIsNone(unit["start_codepoint"])
                            self.assertIsNone(unit["end_codepoint"])
                        else:
                            lo, hi = unit["start_codepoint"], unit["end_codepoint"]
                            self.assertEqual(source[lo:hi], unit["text"])

    def test_catalog_preserves_punctuation_only_prompt_and_article_units(self):
        directory, units = catalog("？！\n；；", "!!\r\n！？")
        self.assertEqual([u["text"] for u in units], ["？！", "；；", "!!", "！？"])
        self.assertEqual([c["text"] for c in directory["candidates"]],
                         [u["text"] for u in units])

    def test_candidate_spans_are_unicode_codepoints(self):
        for task, discover in (("focused", pipeline.focused_candidates),
                               ("extended", pipeline.extended_candidates)):
            story = "问候 naïve；读 résumé。又写 cafe\u0301 完毕。"
            found = discover(story)
            self.assertTrue(found)
            self.assertIn("naïve", "".join(x["text"] for x in found))
            self.assertIn("résumé", "".join(x["text"] for x in found))
            for candidate in found:
                lo = candidate.get("start_char", candidate.get("start_codepoint"))
                hi = candidate.get("end_char", candidate.get("end_codepoint"))
                self.assertEqual(story[lo:hi], candidate["text"])
                self.assertGreater(hi, lo)

    def test_discovered_events_cover_unique_ordered_nonoverlapping_candidates(self):
        payload, output, discovery, candidates = prepared()
        frozen = pipeline.freeze_focused(payload["normalized_story_text"], candidates, discovery)
        self.assertEqual([e["event_id"] for e in frozen["events"]], ["E001"])
        wrong = copy.deepcopy(discovery)
        wrong["candidate_reviews"][0]["label"] = "NON_CSW"
        with self.assertRaises(ValueError):
            pipeline.freeze_focused(payload["normalized_story_text"], candidates, wrong)
        wrong = copy.deepcopy(discovery)
        wrong["event_proposal"]["events"].append(copy.deepcopy(wrong["event_proposal"]["events"][0]))
        wrong["event_proposal"]["events"][-1]["event_id"] = "E002"
        with self.assertRaises(ValueError):
            pipeline.freeze_focused(payload["normalized_story_text"], candidates, wrong)
        wrong = copy.deepcopy(discovery)
        wrong["event_proposal"]["events"][0]["start_codepoint"] += 1
        with self.assertRaises(ValueError):
            pipeline.freeze_focused(payload["normalized_story_text"], candidates, wrong)

    def test_focused_discovery_identity_is_bound_before_freezing(self):
        payload, _, discovery, candidates = prepared()
        pipeline.freeze_focused(payload["normalized_story_text"], candidates, discovery)
        for field in ("normalized_story_text_sha256", "prompt_id", "text_id",
                      "params_id", "params_sha256"):
            wrong = copy.deepcopy(discovery)
            wrong["event_proposal"][field] = "0" * 64 if "sha256" in field else "other"
            with self.subTest(field=field), self.assertRaises(ValueError):
                pipeline.freeze_focused(payload["normalized_story_text"], candidates, wrong)
        for field in ("prompt_id", "text_id", "params_id", "params_sha256"):
            wrong = copy.deepcopy(discovery)
            wrong["metadata"][field] = wrong["event_proposal"][field] = (
                "0" * 64 if "sha256" in field else "other")
            with self.subTest(matching_wrong_metadata=field), self.assertRaises(ValueError):
                pipeline.build_focused_scoring_input(
                    payload["visible_prompt"], payload["normalized_story_text"], wrong,
                    **{k: payload[k] for k in ("prompt_id", "text_id", "params_id",
                        "params_sha256", "vocabulary_id", "vocabulary_sha256")})
    def test_output_follows_schema_frozen_events_and_metadata(self):
        payload, output, _, _ = prepared()
        schema = json.loads((ROOT / "evaluation/schemas/focused_scoring_output_schema.json").read_text())
        check_schema(schema, output)
        run_stage.validate_judge_output("focused", payload, output)
        bad = copy.deepcopy(output)
        bad["event_evaluations"][0]["event_id"] = "E999"
        with self.assertRaises(ValueError):
            run_stage.validate_judge_output("focused", payload, bad)
        for field in ("run_id", "text_id", "prompt_id", "params_sha256", "vocabulary_sha256"):
            bad = copy.deepcopy(output)
            bad["metadata"][field] = "wrong"
            with self.subTest(field=field), self.assertRaises(ValueError):
                run_stage.validate_judge_output("focused", payload, bad)

    def test_licensing_exceptions_and_literary_support(self):
        payload, output, _, _ = prepared()
        bad = copy.deepcopy(output)
        bad["event_evaluations"][0]["d1"]["role"] = "FAIL"
        bad["exceptions"] = [{
            "scope": "EVENT", "event_id": "E001", "dimension": "D1.role",
            "value": "FAIL", "reason": "Synthetic", "evidence_ids": ["T001"]
        }]
        with self.assertRaises(ValueError):
            run_stage.validate_judge_output("focused", payload, bad)
        bad["exceptions"][0]["evidence_ids"] = ["P001"]
        run_stage.validate_judge_output("focused", payload, bad)
        bad["exceptions"] = []
        with self.assertRaises(ValueError):
            run_stage.validate_judge_output("focused", payload, bad)

        supporting = copy.deepcopy(payload)
        text = supporting["normalized_story_text"]
        for number, (lo, hi) in enumerate(((0, 2), (len(text)-2, len(text))), 2):
            supporting["evidence_units"].append({
                "evidence_id": f"T{number:03d}", "source": "GENERATED_TEXT",
                "start_char": lo, "end_char": hi, "text": text[lo:hi]
            })
        anchored = copy.deepcopy(output)
        anchored["literary_quality"]["dimensions"]["L1"] = 4
        anchored["literary_quality"]["high_score_evidence"] = [{
            "dimension": "L1", "evidence_ids": ["T002", "T003"]
        }]
        run_stage.validate_judge_output("focused", supporting, anchored)
        anchored["literary_quality"]["high_score_evidence"][0]["evidence_ids"] = ["T002", "T002"]
        with self.assertRaises(ValueError):
            run_stage.validate_judge_output("focused", supporting, anchored)
        anchored = copy.deepcopy(output)
        anchored["literary_quality"]["dimensions"]["L5"] = 5
        anchored["exceptions"] = [{
            "scope": "LITERARY", "event_id": None, "dimension": "L5",
            "value": 5, "reason": "Synthetic", "counterfactual_loss": "   "
        }]
        anchored["literary_quality"]["high_score_evidence"] = [{
            "dimension": "L5", "evidence_ids": ["T002", "T003"]
        }]
        with self.assertRaises(ValueError):
            run_stage.validate_judge_output("focused", supporting, anchored)

    def test_score_applicability_is_enforced(self):
        payload, output, _, _ = prepared()
        schema = json.loads((ROOT / "evaluation/schemas/focused_scoring_output_schema.json").read_text())
        invalid = copy.deepcopy(output)
        invalid["event_evaluations"][0]["d2"]["D2.1"] = "NA"
        with self.assertRaises(Exception):
            check_schema(schema, invalid)
        invalid = copy.deepcopy(output)
        invalid["event_evaluations"][0]["d3"]["D3.4"] = "IE"
        with self.assertRaises(Exception):
            check_schema(schema, invalid)
        invalid = copy.deepcopy(output)
        invalid["event_evaluations"][0]["d2"]["D2.4"] = "IE"
        invalid["exceptions"].append({
            "scope": "EVENT", "event_id": "E001", "dimension": "D2.4",
            "value": "IE", "reason": "The permitted evidence is inconclusive"
        })
        check_schema(schema, invalid)
        run_stage.validate_judge_output("focused", payload, invalid)

    def test_event_level_citations_must_resolve_to_frozen_evidence(self):
        # A known contract probe: retain the assertion until all entrypoints enforce it.
        payload, output, _, _ = prepared()
        output["event_evaluations"][0]["evidence_ids"] = ["T999"]
        with self.assertRaises(ValueError):
            run_stage.validate_judge_output("focused", payload, output)


if __name__ == "__main__":
    unittest.main()
