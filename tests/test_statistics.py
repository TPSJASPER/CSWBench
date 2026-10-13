"""Statistical analysis contracts on deterministic, independently calculated inputs."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from experiments.statistics import robustness as stats


class StatisticalAnalysis(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def table(self, name, rows):
        path = self.root / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def article(self, task, article_id, prompt, model, judge, score):
        return {"task": task, "article_id": article_id, "prompt_card_id": prompt,
                "model": model, "judge": judge, "total": score}

    def test_seed_bootstrap_preserves_matched_model_contrasts(self):
        rows = []
        for seed, base in (("s1", 10), ("s2", 12), ("s3", 14)):
            for model, offset in (("alpha", 1), ("beta", 0)):
                rows.append(self.article("focused", model + seed, seed, model, "dual", base + offset))
        scores = self.table("article.csv", rows)
        seeds = self.table("seeds.csv", [
            {"task": "focused", "prompt_card_id": seed, "seed_id": seed}
            for seed in ("s1", "s2", "s3")])
        stats.main(["bootstrap", "--scores", str(scores), "--seeds", str(seeds),
                    "--out", str(self.root / "results"), "--replicates", "100", "--seed", "7"])
        output = json.loads((self.root / "results/bootstrap.json").read_text())
        difference = output["model_pairs"][0]
        self.assertAlmostEqual(difference["difference"], 1)
        self.assertAlmostEqual(difference["ci_low"], 1)
        self.assertAlmostEqual(difference["ci_high"], 1)
        self.assertEqual(output["seed_clusters"], 3)

    def test_bootstrap_rejects_incomplete_seed_mapping(self):
        scores = self.table("score.csv", [
            self.article("focused", "a", "missing", "alpha", "dual", 10)])
        seeds = self.table("mapping.csv", [
            {"task": "focused", "prompt_card_id": "other", "seed_id": "s1"}])
        with self.assertRaisesRegex(ValueError, "missing Seed mapping"):
            stats.main(["bootstrap", "--scores", str(scores), "--seeds", str(seeds),
                        "--out", str(self.root / "out"), "--replicates", "20"])

    def test_tied_rank_statistics_and_absolute_agreement(self):
        self.assertEqual(stats.average_ranks([2, 2, 5]), [1.5, 1.5, 3])
        self.assertAlmostEqual(stats.spearman([1, 2, 3], [10, 20, 30]), 1)
        self.assertAlmostEqual(stats.kendall_w([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), 1)
        self.assertAlmostEqual(stats.icc_absolute_single([[1, 1], [2, 2], [3, 3]]), 1)
        self.assertEqual(stats.holm([.01, .02, .2]), [.03, .04, .2])

    def test_judge_ablation_and_human_agreement(self):
        rows = []
        for model, score in (("gpt-5.6-sol", 14), ("deepseek-v4-flash", 12),
                             ("third", 10)):
            for judge, adjustment in (("gpt", .2), ("deepseek", -.2), ("dual", 0)):
                rows.append(self.article("focused", "a_" + model, "p", model, judge,
                                         score + adjustment))
        scores = self.table("judge.csv", rows)
        human = self.table("human.csv", [
            {"task": "focused", "article_id": str(i), "human_1": i,
             "human_2": i, "human_3": i, "dual_total": i} for i in range(1, 5)])
        stats.main(["judges", "--scores", str(scores), "--human", str(human),
                    "--out", str(self.root / "out")])
        report = json.loads((self.root / "out/judge_robustness.json").read_text())
        self.assertAlmostEqual(report["focused"]["leave_self_judge_out"]["spearman"], 1)
        self.assertAlmostEqual(report["human_validation"]["focused"]["kendall_w"], 1)
        self.assertAlmostEqual(report["human_validation"]["focused"]["icc_a1"], 1)

    def test_source_balance_and_matched_annotator_contrast(self):
        cards = self.root / "cards.jsonl"
        with cards.open("w", encoding="utf-8") as handle:
            for pid, source in (("a", "ASCEND"), ("b", "SwitchLingua")):
                handle.write(json.dumps({"prompt_id": pid, "task_type": "focused_narrative",
                                         "source_dataset": source}) + "\n")
        rows = []
        panel = []
        for model, base in (("alpha", 10), ("beta", 8)):
            for pid, increment in (("a", 0), ("b", 2)):
                rows.append(self.article("focused", model + pid, pid, model, "dual",
                                         base + increment))
            for judge, increment in (("gpt", 0), ("qwen", .5)):
                panel.append({"task": "focused", "constructor": "gpt",
                              "annotator": judge, "model": model,
                              "article_id": model + "p", "total": base + increment})
        scores = self.table("scores.csv", rows)
        panels = self.table("panels.csv", panel)
        stats.main(["pipeline", "--scores", str(scores), "--cards", str(cards),
                    "--panels", str(panels), "--out", str(self.root / "out")])
        data = json.loads((self.root / "out/pipeline_sensitivity.json").read_text())
        self.assertAlmostEqual(data["source_balanced"]["focused"]["rank_agreement"]["spearman"], 1)
        self.assertAlmostEqual(data["within_constructor_annotator_contrast"][0]["qwen_minus_gpt"], .5)

    def test_interventions_follow_cluster_and_paired_units(self):
        counterfactual = []
        for i, group in enumerate(("HIGH", "HIGH", "LOW", "LOW")):
            for judge in ("gpt", "deepseek"):
                for order in ("first", "second"):
                    counterfactual.append({"event_id": f"E{i}", "narrative_id": f"N{i}",
                                           "group": group, "judge": judge, "order": order,
                                           "outcome": "cs" if group == "HIGH" else "zh"})
        cf = self.table("counterfactual.csv", counterfactual)
        swaps = []
        for narrative in ("A", "B", "C"):
            for judge in ("gpt", "deepseek"):
                for cond, dose in ([("BASELINE", 0)] +
                                   [(c, p) for c in ("CS", "ZH") for p in (2, 4, 6, 8)]):
                    x = 4.0 if cond == "BASELINE" else 4 - dose * (.1 if cond == "CS" else .05)
                    swaps.append({"narrative_id": narrative, "judge": judge,
                                  "condition": cond, "pairs": dose,
                                  "whole": x, "multilingual": x, "general": x})
        swap_file = self.table("swaps.csv", swaps)
        stats.main(["interventions", "--counterfactual", str(cf), "--swaps", str(swap_file),
                    "--out", str(self.root / "out"), "--replicates", "100", "--seed", "4"])
        data = json.loads((self.root / "out/interventions.json").read_text())
        self.assertAlmostEqual(data["counterfactual"]["high_minus_low"], 2)
        self.assertAlmostEqual(data["counterfactual"]["cluster_bootstrap"]["event"]["ci95"][0], 2)
        row = next(r for r in data["position_swaps"]
                   if r["dimension"] == "whole" and r["pairs"] == 8)
        self.assertAlmostEqual(row["zh_minus_cs"], .4)
        self.assertAlmostEqual(row["ci_low"], .4)
        self.assertTrue((self.root / "out/position_swaps.csv").exists())

    def test_incomplete_swap_design_is_rejected(self):
        rows = [{"narrative_id": "A", "judge": "gpt", "condition": "BASELINE",
                 "pairs": 0, "whole": 4, "multilingual": 4, "general": 4}]
        with self.assertRaisesRegex(ValueError, "incomplete swap"):
            stats.position_swap_analysis(self.table("incomplete.csv", rows), 40, 1)

    def test_invalid_data_does_not_produce_successful_results(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            stats.number("nan")
        with self.assertRaisesRegex(ValueError, "constant"):
            stats.spearman([2, 2, 2], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
