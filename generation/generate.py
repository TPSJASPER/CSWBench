#!/usr/bin/env python3
"""Generate CSWBench narratives from the released prompt set.

This is the public generation entry point. It uses the published request
protocol:

  * Focused Narrative: one system message that fixes the output contract, then the prompt
    text as the user message.
  * Extended Narrative: the prompt text as the single user message.

It writes one JSONL record per (prompt, model) so that evaluation/ can consume the result.

Usage
-----
    # inspect the requests without calling anything
    python generate.py --task focused --config ../configs/models.example.json --dry-run

    # generate with one model
    python generate.py --task focused --config my_models.json --models gpt-5.6-sol

    # generate everything
    python generate.py --task extended --config my_models.json --out ../outputs

No API key is ever read from this repository or written to any output file. Keys are read
from the environment variables named in the config file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "data"

# The fixed output contract for Focused Narrative.
FOCUSED_SYSTEM_INSTRUCTION = (
    "只输出小说正文，不输出标题、说明、分析、计数过程、Markdown围栏或任何正文以外的内容。"
)

# Extended Narrative was generated with the prompt text as the single user message.
TASK_SYSTEM_INSTRUCTION = {
    "focused_narrative": FOCUSED_SYSTEM_INSTRUCTION,
    "extended_narrative": None,
}

# Some reasoning models prepend a <think>…</think> block to the streamed text.
# Remove a leading block before storing the article text.
THINK_BLOCK = re.compile(r"^\s*<think\b[^>]*>[\s\S]*?</think\s*>\s*", re.IGNORECASE)


def strip_think_block(text: str) -> str:
    """Remove a leading <think>…</think> block, leaving the article body."""
    return THINK_BLOCK.sub("", text, count=1)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_messages(record: dict, system_instruction: str | None) -> list[dict]:
    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": record["prompt_text"]})
    return messages


def build_body(record: dict, model_cfg: dict) -> dict:
    system_instruction = model_cfg.get(
        "system_instruction", TASK_SYSTEM_INSTRUCTION.get(record["task_type"])
    )
    token_param = model_cfg.get("token_param", "max_tokens")
    body = {
        "model": model_cfg["model_id"],
        "messages": build_messages(record, system_instruction),
        "stream": bool(model_cfg.get("stream", False)),
        "temperature": float(model_cfg.get("temperature", 0.7)),
        "top_p": float(model_cfg.get("top_p", 0.7)),
        token_param: int(model_cfg.get("max_tokens", 30000)),
    }
    if "reasoning_effort" in model_cfg:
        body["reasoning_effort"] = model_cfg["reasoning_effort"]
    if "enable_thinking" in model_cfg:
        body["enable_thinking"] = bool(model_cfg["enable_thinking"])
    body.update(model_cfg.get("extra_body", {}))
    return body


def _read_stream(response) -> tuple[str, str | None, str, dict | None]:
    """Collect SSE text, returned model, finish reason and optional usage."""
    chunks: list[str] = []
    returned_model = None
    finish_reason = None
    usage = None
    saw_done = False
    for raw in response:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            saw_done = True
            break
        if not data:
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise RuntimeError("malformed generation SSE event") from exc
        returned_model = event.get("model") or returned_model
        if event.get("usage") is not None:
            usage = event["usage"]
        for choice in event.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            piece = (choice.get("delta") or {}).get("content")
            if isinstance(piece, str) and piece:
                chunks.append(piece)
            reason = choice.get("finish_reason")
            if reason is not None:
                if finish_reason is not None and finish_reason != reason:
                    raise RuntimeError("conflicting generation SSE finish reasons")
                finish_reason = reason
    if not saw_done:
        raise RuntimeError("generation SSE ended before [DONE]")
    if finish_reason is None:
        raise RuntimeError("generation SSE ended without an explicit finish_reason")
    return "".join(chunks), returned_model, finish_reason, usage


def call_model(body: dict, model_cfg: dict, timeout: int) -> dict:
    """POST one request and return the article text plus transport metadata.

    Errors are raised and recorded by the caller; this transport does not retry failed requests.
    """
    url = model_cfg["base_url"].rstrip("/") + "/chat/completions"
    api_key = os.environ.get(model_cfg["api_key_env"], "")
    if not api_key:
        raise RuntimeError(
            f"environment variable {model_cfg['api_key_env']} is not set; "
            "export your own key before running"
        )
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    started = time.time()
    returned_model = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if body.get("stream"):
                text, returned_model, finish_reason, usage = _read_stream(response)
            else:
                payload = json.loads(response.read().decode("utf-8"))
                returned_model = payload.get("model")
                choice = (payload.get("choices") or [{}])[0]
                text = (choice.get("message") or {}).get("content") or ""
                finish_reason = choice.get("finish_reason")
                usage = payload.get("usage")
                if not text:
                    raise RuntimeError(f"empty content in response: {json.dumps(payload)[:400]}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    if model_cfg.get("strip_think_block", True):
        text = strip_think_block(text)
    if not text.strip():
        raise RuntimeError("empty narrative after response cleanup")
    if finish_reason == "length":
        raise RuntimeError("generation truncated by output-token limit")
    accepted_models = model_cfg.get("accepted_returned_models", [body["model"]])
    if returned_model is not None and returned_model not in accepted_models:
        raise RuntimeError(f"returned model {returned_model!r} does not match configured identity {body['model']!r}")
    return {
        "article_text": text,
        "finish_reason": finish_reason,
        "usage": usage,
        "model_returned": returned_model,
        "model_returned_matches_request": (
            returned_model is None or returned_model == body["model"]
        ),
        "latency_seconds": round(time.time() - started, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=["focused", "extended"])
    parser.add_argument("--config", required=True, help="model config JSON (see ../configs/models.example.json)")
    parser.add_argument("--models", nargs="*", help="subset of model_id values to run (default: all)")
    parser.add_argument("--data", type=Path, default=None, help="override the prompt JSONL path")
    parser.add_argument("--out", type=Path, default=HERE / "outputs", help="output directory")
    parser.add_argument("--dry-run", action="store_true", help="print requests without calling any API")
    parser.add_argument("--concurrency", type=int, default=1, help="parallel requests (default 1)")
    parser.add_argument("--timeout", type=int, default=900, help="per-request timeout in seconds")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N prompts")
    args = parser.parse_args(argv)

    task_type = "focused_narrative" if args.task == "focused" else "extended_narrative"
    data_path = args.data or (DEFAULT_DATA / f"{args.task}.jsonl")
    records = [r for r in load_jsonl(data_path) if r["task_type"] == task_type]
    if args.limit:
        records = records[: args.limit]

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    models = [m for m in config["models"] if not args.models or m["model_id"] in args.models]
    if not models:
        parser.error("no models selected")

    seen, unique_models = set(), []
    for model in models:
        if model["model_id"] not in seen:
            seen.add(model["model_id"])
            unique_models.append(model)
    models = unique_models

    def request_hash(record, model):
        request_body = build_body(record, model)
        return hashlib.sha256(json.dumps(request_body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    import hashlib

    plan = [(record, model) for record in records for model in models]
    print(f"task={task_type} prompts={len(records)} models={len(models)} calls={len(plan)}", file=sys.stderr)

    if args.dry_run:
        record, model = plan[0]
        body = build_body(record, model)
        preview = {
            "endpoint": model["base_url"].rstrip("/") + "/chat/completions",
            "api_key_env": model["api_key_env"],
            "roles": [m["role"] for m in body["messages"]],
            "user_message_is_prompt_text": body["messages"][-1]["content"] == record["prompt_text"],
            "body_keys": sorted(k for k in body if k != "messages"),
        }
        if len(records) * len(models) > 1:
            print(f"(dry run: {len(plan)} calls planned; showing the first)", file=sys.stderr)
        print(json.dumps({"plan": preview, "first_prompt_id": record["prompt_id"]}, ensure_ascii=False, indent=2))
        print("\n--- first request body (messages truncated) ---", file=sys.stderr)
        trimmed = dict(body)
        trimmed["messages"] = [
            {**m, "content": m["content"][:200] + ("…" if len(m["content"]) > 200 else "")}
            for m in body["messages"]
        ]
        print(json.dumps(trimmed, ensure_ascii=False, indent=2), file=sys.stderr)
        return 0

    for model in models:
        if "<" in model["base_url"] or ">" in model["base_url"]:
            parser.error(f"{model['model_id']}: base_url still holds a placeholder; edit your config first")

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.task}_generations.jsonl"
    done = set()
    if out_path.exists():
        for row in load_jsonl(out_path):
            if row.get("status") == "ok":
                if row.get("request_sha256"):
                    done.add((row["prompt_id"], row["model_id"], row["request_sha256"]))
        print(f"resuming: {len(done)} (prompt, model) pairs already present", file=sys.stderr)

    lock = threading.Lock()
    written = 0

    def run(record: dict, model: dict) -> dict:
        body = build_body(record, model)
        base = {
            "prompt_id": record["prompt_id"],
            "task_type": record["task_type"],
            "model_id": model["model_id"],
            "prompt_text_sha256": hashlib.sha256(record["prompt_text"].encode("utf-8")).hexdigest(),
            "request_sha256": request_hash(record, model),
        }
        try:
            result = call_model(body, model, args.timeout)
            return {**base, "status": "ok", **result}
        except Exception as exc:  # noqa: BLE001 - the failure is recorded, not raised
            return {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"}

    with out_path.open("a", encoding="utf-8", newline="\n") as sink:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = {}
            for record, model in plan:
                if (record["prompt_id"], model["model_id"], request_hash(record, model)) in done:
                    continue
                futures[pool.submit(run, record, model)] = (record, model)
            for future in as_completed(futures):
                result = future.result()
                record, _ = futures[future]
                result["prompt_text_sha256"] = hashlib.sha256(
                    record["prompt_text"].encode("utf-8")
                ).hexdigest()
                with lock:
                    sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                    sink.flush()
                    written += 1
                    if written % 25 == 0:
                        print(f"  {written} written", file=sys.stderr)

    print(f"wrote {written} records to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
