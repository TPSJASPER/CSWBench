"""Two complete offline workflows with invented narratives and model-like responses."""
import copy
import csv
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "tests"))

import pipeline_inputs
import run_stage
from evaluate import LiveTransport, Workflow, check_schema
from fixtures.synthetic import STORIES, SyntheticTransport, extended_score, focused_discovery


class PipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def workflow(self, task, transport, *, workdir=None, article_text=None):
        story = self.root / (task + ".txt")
        story.write_text(STORIES[task] if article_text is None else article_text, encoding="utf-8")
        options = {
            "task": task,
            "prompt_id": "PCV4_238" if task == "focused" else "LGPC_0004",
            "article": story, "workdir": workdir or self.root / (task + "-work"),
            "transport": transport, "article_id": "synthetic-" + task,
            "generation_model": "constructed-model"
        }
        if task == "extended":
            options["blind_id"] = "LJ-B0001"
            options["registry"] = [
                {"character_slot": "A", "language_condition": {"prompt_excerpt": "Chinese and English"}},
                {"character_slot": "B", "language_condition": {"prompt_excerpt": "Chinese and English"}}
            ]
        return Workflow(**options)

    def assert_results(self, root, expected_total):
        with (root / "results/article_scores.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({r["judge"] for r in rows}, {"gpt", "deepseek", "dual"})
        dual = next(row for row in rows if row["judge"] == "dual")
        self.assertAlmostEqual(float(dual["total"]), expected_total)
        self.assertEqual(int(dual["events"]), 1)
        self.assertTrue((root / "judge_records.jsonl").is_file())

    def test_focused_and_extended_complete_and_replay_without_model_calls(self):
        for task, expected in (("focused", 15.0), ("extended", 15.5)):
            with self.subTest(task=task):
                transport = SyntheticTransport()
                workdir = self.root / (task + "-work")
                wf = self.workflow(task, transport, workdir=workdir)
                result = wf.run()
                self.assertEqual(result["status"], "COMPLETE")
                self.assert_results(workdir, expected)
                names = [call[0] for call in transport.calls]
                self.assertEqual(
                    names, ["discovery", "judge_gpt", "judge_deepseek"] if task == "focused"
                    else ["discovery", "evidence", "judge_gpt", "judge_deepseek"])
                judges = [entry for entry in transport.calls if entry[0].startswith("judge_")]
                self.assertEqual(len(judges), 2)
                self.assertEqual(judges[0][2], judges[1][2],
                                 "judges must share the exact frozen input and evidence")
                frozen = json.loads((workdir / "frozen_events.json").read_text())
                input_data = judges[0][2]
                event_list = input_data["events"] if task == "focused" else input_data["event_manifest"]["events"]
                self.assertEqual([e["event_id"] for e in event_list],
                                 [e["event_id"] for e in frozen["events"]])
                expected_sha = pipeline_inputs.digest(input_data)
                for name in ("judge_gpt", "judge_deepseek"):
                    receipt = json.loads((workdir / f"{name}_receipt.json").read_text())
                    self.assertEqual(receipt["input_sha256"], expected_sha)
                previous = (workdir / "judge_records.jsonl").read_bytes()
                replay_transport = SyntheticTransport()
                replay = self.workflow(task, replay_transport, workdir=workdir)
                self.assertEqual(replay.run()["status"], "COMPLETE")
                self.assertFalse(replay_transport.calls, "cached replay must make no model calls")
                self.assertEqual((workdir / "judge_records.jsonl").read_bytes(), previous)
                self.assert_results(workdir, expected)

    def test_cached_results_can_be_read_without_a_model_transport(self):
        # A completed immutable run contains all stage outputs and receipts.
        workdir = self.root / "offline-replay"
        self.workflow("focused", SyntheticTransport(), workdir=workdir).run()
        replay = self.workflow("focused", None, workdir=workdir)
        self.assertEqual(replay.run()["status"], "COMPLETE")
        self.assert_results(workdir, 15.0)

    def test_extended_judges_receive_all_frozen_records_without_relaxing_d1(self):
        class EvidenceTransport(SyntheticTransport):
            def run(self, name, model, payload, prompt, schema, tool, task):
                result = super().run(name, model, payload, prompt, schema, tool, task)
                if name == "evidence":
                    final_id = payload["evidence_catalog"]["candidates"][-1]["evidence_id"]
                    result["ambiguities"] = [{
                        "fact_id": "unresolved-language-claim",
                        "statement": "A language-resource claim remains ambiguous",
                        "evidence": [final_id],
                    }]
                return result

        transport = EvidenceTransport()
        workdir = self.root / "extended-evidence"
        wf = self.workflow("extended", transport, workdir=workdir)
        self.assertEqual(wf.run(aggregate=False)["status"], "COMPLETE")
        semantic = json.loads((workdir / "frozen_semantic_evidence.json").read_text())
        judges = [call[2] for call in transport.calls if call[0].startswith("judge_")]
        self.assertEqual(len(judges), 2)
        self.assertEqual(judges[0], judges[1])
        payload = judges[0]
        frozen = payload["evidence_manifest"]["frozen_evidence"]
        for field in ("prompt_expected_facts", "article_observed_facts", "ambiguities"):
            self.assertTrue(semantic[field])
            self.assertEqual(frozen[field], semantic[field])
        self.assertEqual(frozen["narrative_system"], semantic["narrative_system"])
        self.assertEqual(frozen["character_profiles"],
                         [semantic["character_profiles"][slot] for slot in ("A", "B")])
        for name in ("judge_gpt", "judge_deepseek"):
            receipt = json.loads((workdir / f"{name}_receipt.json").read_text())
            self.assertEqual(receipt["input_sha256"], pipeline_inputs.digest(payload))

        schema = json.loads((ROOT / "evaluation/schemas/extended_scoring_input_schema.json").read_text())
        for field in ("prompt_expected_facts", "article_observed_facts", "ambiguities"):
            missing = copy.deepcopy(payload)
            del missing["evidence_manifest"]["frozen_evidence"][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_schema(schema, missing)
        invalid = copy.deepcopy(payload)
        invalid["evidence_manifest"]["frozen_evidence"]["ambiguities"][0]["evidence"] = ["T9999"]
        with self.assertRaises(ValueError):
            run_stage.validate_input_evidence("extended", invalid)

        units = {unit["evidence_id"]: unit for unit in payload["evidence_units"]}
        event = payload["event_manifest"]["events"][0]
        final_id = frozen["ambiguities"][0]["evidence"][0]
        self.assertGreater(units[final_id]["end_codepoint"], event["start_codepoint"])
        self.assertNotIn(final_id, event["d1_allowed_evidence_ids"])
        output = extended_score(payload, "gpt-5.6-sol")
        gate = output["event_judgments"][0]["d1"]["role"]
        gate["evidence_ids"] = [final_id]
        gate["evidence_strings"] = [units[final_id]["text"]]
        with self.assertRaises(ValueError):
            run_stage.validate_judge_output("extended", payload, output)

    def test_cached_artifact_and_input_identity_cannot_drift(self):
        workdir = self.root / "focused-work"
        self.workflow("focused", SyntheticTransport(), workdir=workdir).run()
        receipt_path = workdir / "judge_gpt_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["input_sha256"] = "0" * 64
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.workflow("focused", None, workdir=workdir).run()
        with self.assertRaises(ValueError):
            Workflow(task="focused", prompt_id="PCV4_238", article=self.root / "focused.txt",
                     workdir=workdir, transport=None, article_id="different-id",
                     generation_model="constructed-model")

    def test_failed_stage_never_produces_successful_final_records(self):
        for task, failing in (("focused", "judge_deepseek"), ("extended", "evidence")):
            with self.subTest(task=task):
                workdir = self.root / ("failure-" + task)
                with self.assertRaises(ValueError):
                    self.workflow(task, SyntheticTransport(fail_stage=failing),
                                  workdir=workdir).run()
                self.assertFalse((workdir / "judge_records.jsonl").exists())
                self.assertFalse((workdir / "results/article_scores.csv").exists())

    def test_extended_punctuation_runs_complete_with_exact_evidence_offsets(self):
        transport = SyntheticTransport()
        text = "？！\n林澈问：“Really?!”周岚喊：“Wait!!”\n末行🙂！？"
        wf = self.workflow("extended", transport, article_text=text)
        self.assertEqual(wf.run(aggregate=False)["status"], "COMPLETE")
        units = json.loads((wf.workdir / "frozen_evidence_units.json").read_text())
        article_units = [u for u in units if u["source"] == "ARTICLE"]
        self.assertEqual("".join(c for c in text if not c.isspace()),
                         "".join(c for u in article_units for c in u["text"] if not c.isspace()))
        for unit in article_units:
            self.assertEqual(text[unit["start_codepoint"]:unit["end_codepoint"]], unit["text"])

    def test_focused_wrong_proposal_identity_never_reaches_scoring(self):
        class WrongIdentityTransport(SyntheticTransport):
            def run(self, name, model, payload, prompt, schema, tool, task):
                result = super().run(name, model, payload, prompt, schema, tool, task)
                if name == "discovery":
                    result["event_proposal"]["normalized_story_text_sha256"] = "0" * 64
                return result

        transport = WrongIdentityTransport()
        workdir = self.root / "wrong-proposal"
        with self.assertRaises(ValueError):
            self.workflow("focused", transport, workdir=workdir).run()
        self.assertEqual([c[0] for c in transport.calls], ["discovery"])
        self.assertFalse((workdir / "frozen_events.json").exists())
        self.assertFalse((workdir / "judge_records.jsonl").exists())

    def test_extended_manifest_challenges_block_both_judges_and_retain_reason(self):
        for manifest in ("event_manifest", "evidence_manifest"):
            for judge in ("judge_gpt", "judge_deepseek"):
                class ChallengingTransport(SyntheticTransport):
                    def run(self, name, model, payload, prompt, schema, tool, task):
                        result = super().run(name, model, payload, prompt, schema, tool, task)
                        if name == judge:
                            result["manifest_challenges"][manifest] = {
                                "raised": True, "reason": "Frozen manifest requires correction"}
                        return result

                workdir = self.root / (manifest + "-" + judge)
                with self.subTest(manifest=manifest, judge=judge), self.assertRaises(ValueError):
                    self.workflow("extended", ChallengingTransport(), workdir=workdir).run()
                self.assertFalse((workdir / "judge_records.jsonl").exists())
                self.assertFalse((workdir / "results/article_scores.csv").exists())
                retained = json.loads((workdir / (judge + "_challenge.json")).read_text())
                self.assertEqual(retained["manifest_challenges"][manifest]["reason"],
                                 "Frozen manifest requires correction")
                self.assertTrue(retained["input_sha256"])

    def test_cached_manifest_challenge_is_rejected_before_export(self):
        workdir = self.root / "cached-challenge"
        with self.assertRaises(ValueError):
            self.workflow("extended", SyntheticTransport(fail_stage="judge_deepseek"),
                          workdir=workdir).run()
        output_path = workdir / "judge_gpt_output.json"
        output = json.loads(output_path.read_text())
        output["manifest_challenges"]["evidence_manifest"] = {
            "raised": True, "reason": "Unresolved evidence conflict"}
        output_path.write_text(json.dumps(output), encoding="utf-8")
        receipt_path = workdir / "judge_gpt_receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["output_sha256"] = pipeline_inputs.digest(output)
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unresolved evidence conflict"):
            self.workflow("extended", None, workdir=workdir).run()
        self.assertTrue((workdir / "judge_gpt_challenge.json").exists())
        self.assertFalse((workdir / "judge_records.jsonl").exists())

    def test_live_manifest_challenge_is_not_retried(self):
        transport = SyntheticTransport()
        self.workflow("extended", transport).run(aggregate=False)
        payload = next(c[2] for c in transport.calls if c[0] == "judge_gpt")
        output = extended_score(payload, "gpt-5.6-sol")
        for field in run_stage.EXTENDED_FIXED:
            output.pop(field)
        output["manifest_challenges"]["event_manifest"] = {
            "raised": True, "reason": "Invalid event boundary"}
        config = self.root / "judges.json"
        config.write_text(json.dumps({"models": [{"model_id": m} for m in
                         ("gpt-5.6-sol", "deepseek-v4-flash")]}), encoding="utf-8")
        live = LiveTransport(config, attempts=3)
        schema = json.loads((ROOT / "evaluation/schemas/extended_scoring_output_schema.json").read_text())
        response = {"choices": [{"message": {"tool_calls": [{"function": {
            "name": "submit_long_judge_scoring", "arguments": json.dumps(output)}}]}}]}
        with patch.object(run_stage, "call", return_value=response) as call:
            with self.assertRaisesRegex(ValueError, "Invalid event boundary"):
                live.run("judge_gpt", "gpt-5.6-sol", payload, "Judge", schema,
                         "submit_long_judge_scoring", "extended")
        self.assertEqual(call.call_count, 1)

    def test_live_focused_identity_mismatch_retries_discovery(self):
        wf = self.workflow("focused", SyntheticTransport())
        wf.run(offline_prepare=True)
        payload = json.loads((wf.workdir / "discovery_input.json").read_text())
        valid = focused_discovery(payload)
        wrong = copy.deepcopy(valid)
        wrong["metadata"]["prompt_id"] = wrong["event_proposal"]["prompt_id"] = "other-card"
        def response(result):
            return {"choices": [{"message": {"tool_calls": [{"function": {
                "name": "submit_discovery_output", "arguments": json.dumps(result)}}]}}]}
        config = self.root / "focused-judges.json"
        config.write_text(json.dumps({"models": [{"model_id": m} for m in
                         ("gpt-5.6-sol", "deepseek-v4-flash")]}), encoding="utf-8")
        live = LiveTransport(config, attempts=2)
        schema = json.loads((ROOT / "evaluation/schemas/focused_discovery_output_schema.json").read_text())
        with patch.object(run_stage, "call", side_effect=[response(wrong), response(valid)]) as call:
            with patch("evaluate.time.sleep"):
                actual = live.run("discovery", "gpt-5.6-sol", payload, "Discover", schema,
                                  "submit_discovery_output", "focused")
        self.assertEqual(call.call_count, 2)
        self.assertEqual(actual, valid)


if __name__ == "__main__":
    unittest.main()
