"""Generation requests, transport completion, and resumable identity are public contracts."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "generation"))
import generate


def response(payload, streamed=False):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def read(self): return json.dumps(payload).encode("utf-8")
        def __iter__(self): return iter(payload)
    return Response()


class GenerationContract(unittest.TestCase):
    def setUp(self):
        self.model = {
            "model_id": "fixture-model", "base_url": "https://example.invalid/v1",
            "api_key_env": "SYNTHETIC_KEY", "temperature": 0.7, "top_p": 0.7,
            "token_param": "max_completion_tokens", "max_tokens": 128,
            "reasoning_effort": "high", "enable_thinking": True
        }
        self.record = {"task_type": "focused_narrative", "prompt_text": "原样保留\n\n完整的 Prompt Card。"}

    def test_messages_and_parameters_preserve_full_prompt(self):
        focused = generate.build_body(self.record, self.model)
        self.assertEqual([m["role"] for m in focused["messages"]], ["system", "user"])
        self.assertEqual(focused["messages"][0]["content"], generate.FOCUSED_SYSTEM_INSTRUCTION)
        self.assertEqual(focused["messages"][1]["content"], self.record["prompt_text"])
        for key, expected in (
            ("model", "fixture-model"), ("temperature", 0.7), ("top_p", 0.7),
            ("max_completion_tokens", 128), ("reasoning_effort", "high"),
            ("enable_thinking", True)
        ):
            self.assertEqual(focused[key], expected)
        extended = generate.build_body(
            {**self.record, "task_type": "extended_narrative"}, self.model)
        self.assertEqual(extended["messages"], [{"role": "user", "content": self.record["prompt_text"]}])

    def test_http_completion_requires_text_and_model_identity(self):
        body = generate.build_body(self.record, self.model)
        def mock_http(req, timeout):
            sent = json.loads(req.data)
            self.assertEqual(sent, body)
            return response({
                "model": "fixture-model",
                "choices": [{"message": {"content": "一个完整故事。"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 8}
            })
        with patch.dict(os.environ, {"SYNTHETIC_KEY": "unused"}), patch.object(
            generate.urllib.request, "urlopen", side_effect=mock_http
        ):
            result = generate.call_model(body, self.model, 5)
        self.assertEqual(result["article_text"], "一个完整故事。")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["usage"], {"total_tokens": 8})
        bad = [
            {"model": "fixture-model", "choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
            {"model": "fixture-model", "choices": [{"message": {"content": "一半"}, "finish_reason": "length"}]},
            {"model": "unexpected-model", "choices": [{"message": {"content": "完整"}, "finish_reason": "stop"}]}
        ]
        for payload in bad:
            with self.subTest(payload=payload), patch.dict(os.environ, {"SYNTHETIC_KEY": "unused"}), patch.object(
                generate.urllib.request, "urlopen", return_value=response(payload)
            ), self.assertRaises(RuntimeError):
                generate.call_model(body, self.model, 5)

    def test_sse_requires_finish_reason_and_done_marker(self):
        body = {**generate.build_body(self.record, self.model), "stream": True}
        def stream(reason="stop", done=True, include_reason=True):
            data = [
                {"model": "fixture-model", "choices": [{"index": 0, "delta": {"content": "故事。"}}]},
                {"model": "fixture-model", "choices": [{"index": 0, "delta": {},
                                                      "finish_reason": reason if include_reason else None}]},
                {"model": "fixture-model", "choices": [], "usage": {"total_tokens": 3}}
            ]
            lines = [("data: " + json.dumps(e) + "\n\n").encode() for e in data]
            return lines + ([b"data: [DONE]\n\n"] if done else [])
        with patch.dict(os.environ, {"SYNTHETIC_KEY": "unused"}):
            with patch.object(generate.urllib.request, "urlopen", return_value=response(stream())):
                result = generate.call_model(body, self.model, 5)
                self.assertEqual(result["article_text"], "故事。")
                self.assertEqual(result["finish_reason"], "stop")
                self.assertEqual(result["usage"], {"total_tokens": 3})
            for lines in (stream(reason="length"), stream(done=False), stream(include_reason=False)):
                with self.subTest(lines=lines), patch.object(
                    generate.urllib.request, "urlopen", return_value=response(lines)
                ), self.assertRaises(RuntimeError):
                    generate.call_model(body, self.model, 5)

    def test_resumption_uses_request_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / "input.jsonl"
            cfg = root / "models.json"
            data.write_text(json.dumps({"prompt_id": "F-1", **self.record}, ensure_ascii=False) + "\n",
                            encoding="utf-8")
            config = {"models": [self.model]}
            cfg.write_text(json.dumps(config), encoding="utf-8")
            calls = []
            def fake_call(body, model, timeout):
                calls.append(body)
                return {"article_text": "测试正文", "finish_reason": "stop"}
            args = ["--task", "focused", "--data", str(data), "--config", str(cfg),
                    "--out", str(root / "output")]
            with patch.object(generate, "call_model", side_effect=fake_call), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(generate.main(args), 0)
                self.assertEqual(generate.main(args), 0)
                self.assertEqual(len(calls), 1, "identical completed request must not re-run")
                config["models"][0]["temperature"] = 0.6
                cfg.write_text(json.dumps(config), encoding="utf-8")
                self.assertEqual(generate.main(args), 0)
            self.assertEqual(len(calls), 2, "changed decoding request must be regenerated")
            rows = [json.loads(line) for line in (root / "output/focused_generations.jsonl").read_text().splitlines()]
            self.assertEqual([row["status"] for row in rows], ["ok", "ok"])
            self.assertNotEqual(rows[0]["request_sha256"], rows[1]["request_sha256"])


if __name__ == "__main__":
    unittest.main()
