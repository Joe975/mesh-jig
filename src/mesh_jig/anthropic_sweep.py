"""Native Messages transport for explicit effort and a conservative shared spend cap.

No retries: a failed request retains its entire worst-case reservation. At verified Anthropic
5.5 model rates, a full 1M input context (including any cold cache-write premium) plus the output cap is reserved
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

from . import gauntlet, llm, metrics

RATES = {"claude-fable-5-2": (8., 40.), "claude-opus-5-5": (4., 20.), "claude-sonnet-5-5": (2., 10.),
         "claude-haiku-5-5": (.5, 2.5)}  # Haiku's maximum (long-context) rates for reservations
# What every model in RATES can write in one reply, thinking included. At high effort the thinking alone has taken
# 50,000 to 63,000 tokens before a script was written, so a cap of 65,536 leaves a reply almost no room.
MAX_OUTPUT_TOKENS = 128_000
CUT = "max_tokens"      # the stop reason of a reply that ran into the output cap


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

    def reserve(self, max_tokens, label, *, wait=False, cache=False, input_tokens=1_100_000):
        # 1.1M conservatively covers either decimal or binary interpretation of 1M.
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if type(input_tokens) is not int or not 1<=input_tokens<=1_100_000:
            raise ValueError('invalid input reservation bound')
        input_rate, output_rate = RATES[self.model]
        # Any input may be a cold five-minute cache write, billed at 1.25x.
        amount = (input_rate * input_tokens * (1.25 if cache else 1) + max_tokens * output_rate) / 1_000_000
        with self.changed:
            while wait and self.active and self.committed() + amount > self.limit:
                self.changed.wait(timeout=30)
            if self.committed() + amount > self.limit:
                raise llm.LLMError("budget stop: insufficient room for worst-case request")
            index = len(self.entries)
            self.entries.append({"label": label, "state": "reserved", "max_tokens": max_tokens,
                                 "charged_or_reserved_usd": amount, "cache_ttl": "5m" if cache else None,
                                 'input_tokens_bound':input_tokens})
            self.active.add(index)
            self.save()
            return index

    def settle(self, index, usage):
        from .gauntlet_sweep import estimate_cost
        normalized = normalize_usage(usage, cache_ttl=self.entries[index].get("cache_ttl"))
        amount = estimate_cost(normalized, self.model)
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


def normalize_usage(raw, *, cache_ttl=None):
    """Native input, cache reads and cache writes are disjoint, not overlapping totals.

    An omitted write-TTL breakdown is known only when our request explicitly chose
    the five-minute policy. Unknown or inconsistent usage cannot settle a reservation.
    """
    if cache_ttl and any(k not in raw for k in ("cache_read_input_tokens", "cache_creation_input_tokens")):
        raise llm.LLMError("missing cache usage: retain full budget reservation")
    values = [metrics.number(raw.get(k, 0)) for k in
              ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens", "output_tokens")]
    if "input_tokens" not in raw or "output_tokens" not in raw or any(v is None for v in values):
        raise llm.LLMError("missing usage or invalid token counts: retain full budget reservation")
    fresh, read, write, output = values
    breakdown = raw.get("cache_creation")
    if breakdown is not None:
        if not isinstance(breakdown, dict):
            raise llm.LLMError("invalid cache write breakdown")
        five = metrics.number(breakdown.get("ephemeral_5m_input_tokens", 0))
        hour = metrics.number(breakdown.get("ephemeral_1h_input_tokens", 0))
        if five is None or hour is None or five + hour != write:
            raise llm.LLMError("inconsistent cache write breakdown")
    elif write and cache_ttl != "5m":
        raise llm.LLMError("unknown cache write TTL: retain full budget reservation")
    else:
        five, hour = write, 0
    return {"prompt_tokens": fresh + read + write, "completion_tokens": output,
            "prompt_tokens_details": {"cached_tokens": read, "cache_write_tokens": write,
                                      "cache_write_tokens_5m": five, "cache_write_tokens_1h": hour},
            "native_usage": raw}


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


def payload(messages, schemas, history, model, max_tokens, effort="high", *, cache=True):
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
    if cache:
        # Four breakpoints at most: system, opening, previous request, current request.
        # The previous endpoint survives even a tool result with >20 image blocks.
        users = [m for m in converted if m["role"] == "user"]
        targets = ([system] if system else [])
        if users:
            indices = {0, *range(max(0, len(users) - 2), len(users))}
            targets.extend(users[i]["content"] for i in sorted(indices))
        for blocks in targets:
            for block in reversed(blocks):
                if block["type"] in ("text", "image", "tool_result", "tool_use"):
                    block["cache_control"] = {"type": "ephemeral", "ttl": "5m"}
                    break
    return out


def read_stream(response):
    """-> (content blocks, usage, serving model, stop reason). A reply cut at the output cap (`CUT`) can end in the
    middle of a tool call: that call's input is left as it started and the caller runs none of the reply's calls.
    Broken tool input in a reply that was not cut is still an error."""
    blocks, partial, usage = {}, {}, {}
    model, stopped, stop_reason = None, False, None
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
            stop_reason = (event.get("delta") or {}).get("stop_reason") or stop_reason
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
            if stop_reason == CUT:
                continue
            raise llm.LLMError(f"invalid Anthropic tool input JSON at block {index}; no automatic retry") from error
    return [blocks[i] for i in sorted(blocks)], usage, model, stop_reason


def chat_for(api, budget, label, *, max_tokens=65536, opener=urllib.request.urlopen,
             log=lambda text: None, effort="high", cache=True, count_tokens=False):
    """The chat function over the native Messages API. `usage["mesh_jig_completion"]` carries the endpoint's own
    stop reason, as `llm.assemble_chunks` does for the other transport. A reply cut at the output cap comes back
    with its text and no tool call (`llm.cut_short(usage)` says so): a call that may have lost its end is not run,
    and the reply's thinking is not kept for the next request, which never sees the cut turn."""
    if api.model not in RATES or api.url != "https://api.anthropic.com/v1":
        raise ValueError("native transport requires a supported direct Anthropic 5.5 model")
    if effort not in (None, "low", "medium", "high"):
        raise ValueError("native adaptive thinking effort must be low, medium or high")
    if budget is not None and budget.model != api.model:
        raise ValueError("budget model must match request model")
    history = {}

    def chat(messages, schemas, left_s=None):
        body = payload(messages, schemas, history, api.model, max_tokens, effort, cache=cache)
        request = urllib.request.Request(api.url + "/messages", data=json.dumps(body).encode(), method="POST",
                  headers={"Content-Type": "application/json", "x-api-key": api.key,
                           "anthropic-version": "2023-06-01"})
        bound=1_100_000
        if budget is not None and count_tokens:
            # Count the exact native input, then allow 10% plus 1024 tokens for estimate variance.
            # A failed count never dispatches a generation request. No silent fallback above the cap.
            counted={k:v for k,v in body.items() if k in ('model','system','messages','tools','thinking','tool_choice')}
            counter=urllib.request.Request(api.url+'/messages/count_tokens',data=json.dumps(counted).encode(),method='POST',headers=dict(request.headers))
            with opener(counter,timeout=api.timeout_s) as response:count=json.loads(response.read(65536)).get('input_tokens')
            if type(count) is not int or not 0<=count<=1_000_000:raise llm.LLMError('invalid provider input token count')
            bound=min(1_100_000,math.ceil(count*1.1)+1024)
        ticket = budget.reserve(max_tokens, label, wait=True, cache=cache,input_tokens=bound) if budget is not None else None
        start = time.monotonic()
        log(f"native {api.model} {effort or 'default'} call; budget reservation {ticket}")
        # One request only. Any transport failure leaves its reservation intact.
        try:
            timeout = api.timeout_s if left_s is None else max(1.0, min(api.timeout_s or left_s, left_s))
            # Reserve first: a budget refusal is not a dispatched request. Preserve measured
            # usage even if serving-model validation or budget settlement subsequently fails.
            with metrics.request(model=api.model, provider="anthropic",
                                 role="implementor" if schemas else "reviewer", retry_index=0) as measured:
                with opener(request, timeout=timeout) as response:
                    blocks, raw_usage, served, stop_reason = read_stream(response)
                measured.usage(raw_usage, dialect="anthropic")
                if served != api.model:
                    raise llm.LLMError(f"unexpected serving model {served}")
                if budget is not None:
                    budget.settle(ticket, raw_usage)
        except Exception:
            if budget is not None:
                budget.failed(ticket)
            raise
        message = {"role": "assistant", "content": "".join(b["text"] for b in blocks if b["type"] == "text")}
        calls = [] if stop_reason == CUT else [
            {"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
            for b in blocks if b["type"] == "tool_use"]
        if calls:
            message["tool_calls"] = calls
            history[identity(message)] = copy.deepcopy(blocks)
        usage = normalize_usage(raw_usage, cache_ttl="5m" if cache else None)
        usage.update(served_model=served,
                     mesh_jig_completion={"finish_reason": stop_reason, "ambiguous": False})
        log(f"native call completed in {time.monotonic()-start:.1f}s"
            + (f"; cut at the {max_tokens}-token output cap" if stop_reason == CUT else ""))
        return message, usage
    return chat
