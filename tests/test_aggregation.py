"""Independent arithmetic examples for the released /20 aggregation rule."""
import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "aggregation"))
import aggregate as scores

L = [f"L{i}" for i in range(1, 6)]
M = ["M-F1", "M-F2A", "M-F2B", "M-F3.1", "M-F3.2", "M-F3.3", "M-F3.4"]


def focused_event(d1, salience, d2, d3):
    return {
        "event_id": "synthetic",
        "d1_role": d1, "d1_world": "PASS", "salience": salience,
        "d2_subdimensions": {f"D2.{i}": d2.get(i, "NA") for i in range(1, 9)},
        "d3_subdimensions": {f"D3.{i}": d3.get(i, "IE") for i in range(1, 6)}
    }


def record(task, article, judge, events, whole=3, model="constructed-model"):
    dims = L if task == "focused" else M
    return {
        "task": task, "article_id": article, "prompt_card_id": "synthetic-card",
        "model": model, "judge": judge, "events": copy.deepcopy(events),
        "whole": {dimension: whole for dimension in dims}
    }


class ScoringAndAggregationContract(unittest.TestCase):
    def test_focused_event_first_and_salience_weighting(self):
        events = [
            focused_event("PASS", "S1", {1: 2, 2: 4}, {1: 5}),
            focused_event("FAIL", "S3", {1: 4, 2: 4}, {1: 1})
        ]
        row = scores.score_record(record("focused", "A", "gpt", events))
        # D1 = 5 * 1/2 = 2.5
        # D2 = (mean(2,4)*1 + mean(4,4)*3) / 4 = 3.75
        # D3 = (5*1 + 1*3) / 4 = 2; Whole = 15/5 = 3
        for dimension, expected in (("D1", 2.5), ("D2", 3.75),
                                    ("D3", 2.0), ("Whole", 3), ("total", 11.25)):
            self.assertAlmostEqual(row[dimension], expected)

    def test_extended_overall_scores_and_seven_dimension_whole(self):
        events = [
            {"event_id": "E001", "d1_role": "PASS", "d1_world": "PASS",
             "salience": "S1", "d2_overall": 1, "d3_overall": 5},
            {"event_id": "E002", "d1_role": "FAIL", "d1_world": "PASS",
             "salience": "S3", "d2_overall": 5, "d3_overall": 1}
        ]
        row = scores.score_record(record("extended", "B", "gpt", events))
        # (1 + 3*5)/4 = 4; (5 + 3*1)/4 = 2
        for name, expected in (("D1", 2.5), ("D2", 4), ("D3", 2),
                               ("Whole", 3), ("total", 11.5)):
            self.assertAlmostEqual(row[name], expected)

    def test_dual_judge_article_before_model_aggregation(self):
        first = [
            focused_event("PASS", "S1", {1: 2, 2: 4}, {1: 5}),
            focused_event("FAIL", "S3", {1: 4, 2: 4}, {1: 1})
        ]
        second = [
            focused_event("PASS", "S1", {i: 2 for i in range(1, 9)},
                          {i: 4 for i in range(1, 6)}),
            focused_event("PASS", "S3", {i: 2 for i in range(1, 9)},
                          {i: 4 for i in range(1, 6)})
        ]
        consistent = [focused_event("PASS", "S2",
                                    {i: 4 for i in range(1, 9)},
                                    {i: 4 for i in range(1, 6)})]
        input_records = [
            record("focused", "A", "gpt", first, whole=3),       # 11.25
            record("focused", "A", "deepseek", second, whole=4), # 15.0
            record("focused", "B", "gpt", consistent, whole=3),  # 16
            record("focused", "B", "deepseek", consistent, whole=3) # 16
        ]
        tables = scores.aggregate(input_records)
        dual = {r["article_id"]: r for r in tables["article"] if r["judge"] == "dual"}
        self.assertAlmostEqual(dual["A"]["total"], 13.125)
        self.assertAlmostEqual(dual["B"]["total"], 16)
        model = next(r for r in tables["model"] if r["judge"] == "dual")
        self.assertAlmostEqual(model["total"], 14.5625)
        self.assertEqual(model["articles"], 2)

    def test_missing_or_unscorable_inputs_cannot_contribute(self):
        valid = focused_event("PASS", "S1", {1: 3}, {1: 4})
        for events in (
            [],
            [{**valid, "d1_role": "NA"}],
            [{**valid, "salience": "NA"}],
            [{**valid, "d2_subdimensions": {f"D2.{i}": "IE" for i in range(1, 9)}}]
        ):
            with self.subTest(events=events), self.assertRaises(ValueError):
                scores.score_record(record("focused", "unscorable", "gpt", events))
        incomplete = record("extended", "missing-whole", "gpt", [{
            "event_id": "E001", "d1_role": "PASS", "d1_world": "PASS",
            "salience": "S1", "d2_overall": 3, "d3_overall": 3
        }])
        del incomplete["whole"]["M-F3.4"]
        with self.assertRaises(ValueError):
            scores.score_record(incomplete)
        with self.assertRaises(ValueError):
            scores.aggregate([record("focused", "solo", "gpt", [valid])])


if __name__ == "__main__":
    unittest.main()
