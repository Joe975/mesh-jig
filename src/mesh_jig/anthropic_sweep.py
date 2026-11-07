"""Native Messages transport for explicit effort and a conservative shared spend cap.

No retries: a failed request retains its entire worst-case reservation. At verified Anthropic
5.5 model rates, a full 1M input context plus the output cap is reserved
before dispatch. This intentionally stops short of the cap when headroom is low.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
import time
import urllib.request

from . import gauntlet, llm

RATES = {"claude-opus-5-5": (4., 20.), "claude-sonnet-5-5": (2., 10.),
         "claude-haiku-5-5": (.5, 2.5)}  # Haiku's maximum (long-context) rates for reservations


class Budget:
    def __init__(self, path, limit, model="claude-opus-5-5"):
        if not math.isfinite(limit) or limit <= 0 or model not in RATES:
            raise ValueError("budget must be finite and positive, with verified model pricing")
        self.model = model
        self.path, self.limit = path, limit
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.active = set()
        self.entries = []
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("model", "claude-opus-5-5") != model:
                raise ValueError("budget ledger belongs to another model")
            self.entries = old["entries"]
            if self.committed() > limit:
                raise ValueError("existing reservations exceed budget")

    def committed(self):
        return sum(e["charged_or_reserved_usd"] for e in self.entries)

    def save(self):
        gauntlet.write_json(self.path, {"limit_usd": self.limit, "model": self.model,
                            "charged_or_reserved_usd": round(self.committed(), 8),
                            "entries": self.entries})

    def reserve(self, max_tokens, label, *, wait=False):
        # 1.1M conservatively covers either decimal or binary interpretation of 1M.
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        input_rate, output_rate = RATES[self.model]
        amount = input_rate * 1.1 + max_tokens * output_rate / 1_000_000
        with self.changed:
            while wait and self.active and self.committed() + amount > self.limit:
                self.changed.wait(timeout=30)
            if self.committed() + amount > self.limit:
                raise llm.LLMError("budget stop: insufficient room for worst-case request")
            index = len(self.entries)
            self.entries.append({"label": label, "state": "reserved", "max_tokens": max_tokens,
                                 "charged_or_reserved_usd": amount})
            self.active.add(index)
            self.save()
            return index

    def settle(self, index, usage):
        # No cache writes requested; unexpected ones must retain the reservation.
        if usage.get("cache_creation_input_tokens"):
            raise llm.LLMError("unexpected cache write: retain full budget reservation")
        if "input_tokens" not in usage or "output_tokens" not in usage:
            raise llm.LLMError("missing usage: retain full budget reservation")
        from .gauntlet_sweep import estimate_cost
        cached = usage.get("cache_read_input_tokens", 0)
        amount = estimate_cost({"prompt_tokens": usage["input_tokens"] + cached,
                                "completion_tokens": usage["output_tokens"],
                                "prompt_tokens_details": {"cached_tokens": cached}}, self.model)
        if amount is None:
            raise llm.LLMError("unpriced usage: retain full budget reservation")
        with self.lock:
            if amount > self.entries[index]["charged_or_reserved_usd"]:
                raise llm.LLMError("provider usage exceeded request reservation")
            self.entries[index].update(state="completed", charged_or_reserved_usd=amount, usage=usage)
            self.active.discard(index)
            self.save()
            self.changed.notify_all()

    def failed(self, index):
        with self.changed:
            if self.entries[index]["state"] == "reserved":
                self.entries[index]["state"] = "unknown_charge_reserved"
            self.active.discard(index)
            self.save()
            self.changed.notify_all()


def identity(message):
    value = {k: message.get(k) for k in ("content", "tool_calls")}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def text_blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks = []
    for part in content or []:
        if part["type"] == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image_url":
            url = part["image_url"]["url"]
            if not url.startswith("data:"):
                raise ValueError("sweep images must be inline data URLs")
            header, data = url.split(",", 1)
            blocks.append({"type": "image", "source": {"type": "base64",
                          "media_type": header[5:].split(";")[0], "data": data}})
        else:
            raise ValueError(f"unsupported content part {part['type']}")
    return blocks


def payload(messages, schemas, history, model, max_tokens, effort="high"):
    system, converted = [], []
    for message in messages:
        role = message["role"]
        if role == "system":
            system.extend(text_blocks(message["content"]))
            continue
        if role == "tool":
            role = "user"
            blocks = [{"type": "tool_result", "tool_use_id": message["tool_call_id"],
                       "content": message.get("content") or ""}]
        elif role == "assistant":
            blocks = copy.deepcopy(history.get(identity(message)))
            if blocks is None:
                blocks = text_blocks(message.get("content"))
                for call in message.get("tool_calls") or []:
                    f = call["function"]
                    blocks.append({"type": "tool_use", "id": call["id"], "name": f["name"],
                                   "input": json.loads(f["arguments"])})
        else:
            blocks = text_blocks(message.get("content"))
        if not blocks:
            continue
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"].extend(blocks)
        else:
            converted.append({"role": role, "content": blocks})
    out = {"model": model, "system": system, "messages": converted, "max_tokens": max_tokens,
           "stream": True, "thinking": {"type": "adaptive"}}
    if effort is not None:
        out["output_config"] = {"effort": effort}
    if schemas:
        out["tools"] = [{"name": t["function"]["name"],
                         "description": t["function"].get("description", ""),
                         "input_schema": t["function"]["parameters"]} for t in schemas]
    return out


def read_stream(response):
    blocks, partial, usage = {}, {}, {}
    model, stopped = None, False
    for raw in response:
        if not raw.startswith(b"data:"):
            continue
        data = raw[5:].strip()
        if not data:
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError as error:
            raise llm.LLMError("invalid Anthropic event JSON; no automatic retry") from error
        kind = event["type"]
        if kind == "error":
            raise llm.LLMError("Anthropic stream error: " + str(event.get("error")))
        if kind == "message_start":
            usage.update(event["message"].get("usage", {}))
            model = event["message"]["model"]
        elif kind == "content_block_start":
            blocks[event["index"]] = event["content_block"]
        elif kind == "content_block_delta":
            index, delta = event["index"], event["delta"]
            for field in ("text", "thinking", "signature"):
                if field in delta:
                    blocks[index][field] = blocks[index].get(field, "") + delta[field]
            if "partial_json" in delta:
                partial[index] = partial.get(index, "") + delta["partial_json"]
        elif kind == "message_delta":
            usage.update(event.get("usage", {}))
        elif kind == "message_stop":
            stopped = True
            break
    if not stopped:
        raise llm.LLMError("incomplete Anthropic stream; no automatic retry")
    for index, value in partial.items():
        # An argument-free tool can emit an empty delta; its start block already
        # supplies the input object. Nonempty malformed arguments must still fail.
        if value == "":
            continue
        try:
            blocks[index]["input"] = json.loads(value)
        except json.JSONDecodeError as error:
            raise llm.LLMError(f"invalid Anthropic tool input JSON at block {index}; no automatic retry") from error
    return [blocks[i] for i in sorted(blocks)], usage, model


def chat_for(api, budget, label, *, max_tokens=65536, opener=urllib.request.urlopen,
             log=lambda text: None, effort="high"):
    if api.model not in RATES or api.url != "https://api.anthropic.com/v1":
        raise ValueError("native transport requires a supported direct Anthropic 5.5 model")
    if effort not in (None, "low", "medium", "high"):
        raise ValueError("native adaptive thinking effort must be low, medium or high")
    if budget is not None and budget.model != api.model:
        raise ValueError("budget model must match request model")
    history = {}

    def chat(messages, schemas, left_s=None):
        body = payload(messages, schemas, history, api.model, max_tokens, effort)
        request = urllib.request.Request(api.url + "/messages", data=json.dumps(body).encode(), method="POST",
                  headers={"Content-Type": "application/json", "x-api-key": api.key,
                           "anthropic-version": "2023-06-01"})
        ticket = budget.reserve(max_tokens, label, wait=True) if budget is not None else None
        start = time.monotonic()
        log(f"native {api.model} {effort or 'default'} call; budget reservation {ticket}")
        # One request only. Any transport failure leaves its reservation intact.
        try:
            timeout = api.timeout_s if left_s is None else max(1.0, min(api.timeout_s or left_s, left_s))
            with opener(request, timeout=timeout) as response:
                blocks, raw_usage, served = read_stream(response)
            if served != api.model:
                raise llm.LLMError(f"unexpected serving model {served}")
            if budget is not None:
                budget.settle(ticket, raw_usage)
        except Exception:
            if budget is not None:
                budget.failed(ticket)
            raise
        message = {"role": "assistant", "content": "".join(b["text"] for b in blocks if b["type"] == "text")}
        calls = [{"id": b["id"], "type": "function", "function": {"name": b["name"],
                  "arguments": json.dumps(b["input"])}} for b in blocks if b["type"] == "tool_use"]
        if calls:
            message["tool_calls"] = calls
            history[identity(message)] = copy.deepcopy(blocks)
        cached = raw_usage.get("cache_read_input_tokens", 0)
        usage = {"prompt_tokens": raw_usage["input_tokens"] + cached,
                 "completion_tokens": raw_usage["output_tokens"],
                 "prompt_tokens_details": {"cached_tokens": cached},
                 "native_usage": raw_usage, "served_model": served}
        log(f"native call completed in {time.monotonic()-start:.1f}s")
        return message, usage
    return chat
