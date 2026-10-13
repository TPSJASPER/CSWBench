"""Published Prompt Cards must match the release's structural contract."""
import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "prompt_id": str, "task_type": str, "prompt_text": str,
    "source_dataset": str, "scene_domain": str, "participant_relation": str,
    "interaction_pattern": list, "preferred_style": list,
    "length_target_codepoints": int
}


class PublishedDataContract(unittest.TestCase):
    def test_released_prompt_cards_are_complete_and_consistent(self):
        manifest = json.loads((ROOT / "data/MANIFEST.json").read_text(encoding="utf-8"))
        stats = json.loads((ROOT / "DATASET_STATS.json").read_text(encoding="utf-8"))
        all_ids = set()
        actual_total = 0
        for task, expected_type, target in (
            ("focused", "focused_narrative", 700),
            ("extended", "extended_narrative", 4000),
        ):
            with self.subTest(task=task):
                path = ROOT / "data" / f"{task}.jsonl"
                raw = path.read_bytes()
                lines = raw.decode("utf-8").splitlines()
                self.assertTrue(lines)
                self.assertTrue(all(line.strip() for line in lines), "blank JSONL record")
                rows = [json.loads(line) for line in lines]
                fields = manifest["files"][path.name]
                digest = hashlib.sha256(raw).hexdigest()
                self.assertEqual(len(rows), fields["records"])
                self.assertEqual(len(raw), fields["size_bytes"])
                self.assertEqual(digest, fields["sha256"])
                self.assertEqual(digest, stats[task]["file_sha256"])
                self.assertEqual(len(rows), manifest["prompt_counts"][task])
                self.assertEqual(len(rows), stats[task]["records"])
                ids = [row["prompt_id"] for row in rows]
                self.assertEqual(len(set(ids)), len(ids))
                self.assertFalse(all_ids.intersection(ids), "Prompt ID reused across tasks")
                all_ids.update(ids)
                self.assertEqual(stats[task]["unique_prompt_ids"], len(set(ids)))
                for index, row in enumerate(rows):
                    with self.subTest(task=task, index=index):
                        self.assertEqual(set(row), set(REQUIRED))
                        for name, expected in REQUIRED.items():
                            self.assertIs(type(row[name]), expected, name)
                        self.assertTrue(row["prompt_id"].strip())
                        self.assertTrue(row["prompt_text"].strip())
                        self.assertEqual(row["task_type"], expected_type)
                        self.assertEqual(row["source_dataset"] in ("ASCEND", "SwitchLingua"), True)
                        self.assertTrue(row["scene_domain"].strip())
                        self.assertTrue(row["participant_relation"].strip())
                        self.assertGreaterEqual(row["length_target_codepoints"], 1)
                        self.assertEqual(row["length_target_codepoints"], target)
                        for name in ("interaction_pattern", "preferred_style"):
                            self.assertTrue(row[name])
                            self.assertTrue(all(isinstance(v, str) and v.strip() for v in row[name]))
                            self.assertEqual(len(row[name]), len(set(row[name])))
                self.assertEqual(
                    Counter(row["source_dataset"] for row in rows), stats[task]["source_dataset"])
                self.assertEqual(
                    (min(len(row["prompt_text"]) for row in rows),
                     max(len(row["prompt_text"]) for row in rows)),
                    (stats[task]["prompt_char_length_min"], stats[task]["prompt_char_length_max"]))
                actual_total += len(rows)
        self.assertEqual(actual_total, manifest["prompt_counts"]["total"])


if __name__ == "__main__":
    unittest.main()
