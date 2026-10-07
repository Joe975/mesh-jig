"""A minimal OpenAI-compatible chat client (urllib only) for the two steps that call a model: the pairwise judge and
`mesh-jig agent`.

Each is any vision model behind an OpenAI-compatible endpoint, named by three environment variables under its own
prefix (MESH_JIG_JUDGE_*, MESH_JIG_AGENT_*):

    <PREFIX>_URL     default https://openrouter.ai/api/v1
    <PREFIX>_MODEL   the judge defaults to deepseek/deepseek-v4.1-flash (the model it was validated on); the agent
                     has no default
    <PREFIX>_KEY     else the provider's own variable, chosen by the URL's host (KEY_BY_HOST), else OPENAI_API_KEY

Unset variables are filled from an env file first: $MESH_JIG_ENV_FILE, else `.env` in the directory mesh-jig is run
from, when it exists (KEY=VALUE lines; `.env.example` is the template). Variables already in the environment win,
nothing is ever exported to a shell, and no file outside that is looked for.
"""
from __future__ import annotations

import base64
import http.client
import io
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

DEFAULT_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4.1-flash"
JUDGE_PREFIX = "MESH_JIG_JUDGE"
AGENT_PREFIX = "MESH_JIG_AGENT"
# the key a provider's own endpoint takes when <PREFIX>_KEY is unset; any other host falls back to OPENAI_API_KEY
KEY_BY_HOST = (("openrouter.ai", "OPENROUTER_API_KEY"), ("api.anthropic.com", "ANTHROPIC_API_KEY"),
               ("generativelanguage.googleapis.com", "GEMINI_API_KEY"))
EFFORTS = ("none", "low", "medium", "high")
ENV_FILE_VAR = "MESH_JIG_ENV_FILE"
DEFAULT_ENV_FILE = ".env"          # relative: the directory mesh-jig is run from
# OpenAI's reasoning families reject `temperature` and want `max_completion_tokens`
REASONING_MODEL_RE = re.compile(r"^(gpt-5|o[1-9])")
IMAGE_MAX_DIM = 1536
IMAGE_QUALITY = 88


class LLMError(RuntimeError):
    pass


@dataclass
class Backend:
    name: str
    url: str
    model: str
    key: str | None = None
    timeout_s: float = 300.0

    def describe(self) -> str:
        return f"{self.model} @ {self.url}"


def load_env_file(path: str | None = None, env=os.environ) -> list[str]:
    """Fill UNSET env vars from a KEY=VALUE file (blanks and comments ignored, empty values skipped). Existing env
    vars win. Returns the names it set."""
    path = path or env.get(ENV_FILE_VAR) or DEFAULT_ENV_FILE
    if not os.path.exists(path):
        return []
    set_names = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v and not env.get(k):
                env[k] = v
                set_names.append(k)
    return set_names


def provider_key_var(url: str) -> str:
    """The provider's own key variable for an endpoint URL (KEY_BY_HOST), else OPENAI_API_KEY."""
    return next((var for host, var in KEY_BY_HOST if host in url), "OPENAI_API_KEY")


def resolve_backend(env=os.environ, *, load_file: bool = True, prefix: str = JUDGE_PREFIX, url: str | None = None,
                    model: str | None = None) -> Backend:
    """A backend from the environment under `prefix` (the judge's by default); `url` / `model` override the variables.
    LLMError when no model is named (the agent has no default) or no key is configured for a hosted endpoint."""
    if load_file:
        load_env_file(env=env)
    who = "the judge" if prefix == JUDGE_PREFIX else "the agent"
    url = (url or env.get(f"{prefix}_URL") or DEFAULT_URL).rstrip("/")
    model = model or env.get(f"{prefix}_MODEL") or (DEFAULT_MODEL if prefix == JUDGE_PREFIX else "")
    if not model:
        raise LLMError(f"{who} needs a model: set {prefix}_MODEL or pass --model")
    openrouter = "openrouter.ai" in url
    local = any(h in url for h in ("localhost", "127.0.0.1"))
    key_var = provider_key_var(url)
    key = env.get(f"{prefix}_KEY") or env.get(key_var)
    if not key and not local:
        raise LLMError(f"{who} needs an API key for {url}: set {prefix}_KEY (or {key_var}); see .env.example")
    return Backend(name="openrouter" if openrouter else "local" if local else "openai-compatible", url=url, model=model,
                   key=key)


def reasoning_extra(backend: Backend) -> dict:
    """Keep a thinking model's reasoning budget low: left alone, several spend their whole output budget thinking and
    return an empty reply. OpenRouter has one unified control; other endpoints are left at their defaults."""
    if backend.name == "openrouter":
        return {"reasoning": {"effort": "low"}}
    return {}


def agent_output_limit(backend: Backend, requested: int | None = None, *, opener=urllib.request.urlopen,
                       log: Callable[[str], None] = lambda s: None) -> int | None:
    """Use OpenRouter's advertised output capacity instead of its smaller implicit reply default.

    Explicit caps win. Other endpoints, unknown models and unavailable metadata keep their own defaults.
    Look up once before the run, never per turn; reasoning and visible output share this budget.
    """
    if requested is not None or backend.name != "openrouter":
        return requested
    try:
        with opener(backend.url + "/models", timeout=10) as response:
            models = json.loads(response.read())["data"]
        model = next((m for m in models if m.get("id") == backend.model), {})
        limit = (model.get("top_provider") or {}).get("max_completion_tokens")
        if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
            return limit
    except (OSError, ValueError, KeyError, TypeError) as error:
        log(f"could not read the model's output capacity ({error}); using the endpoint default")
        return None
    log("model output capacity not advertised; using the endpoint default")
    return None


def effort_extra(backend: Backend, effort: str | None) -> dict:
    """A chosen reasoning effort in the form the endpoint takes: OpenRouter's unified `reasoning`, else the
    `reasoning_effort` field of OpenAI's chat API (which OpenAI-compatible endpoints copy). None sends nothing."""
    if not effort:
        return {}
    if effort not in EFFORTS:
        raise LLMError(f"effort must be one of {EFFORTS}, got {effort!r}")
    return {"reasoning": {"effort": effort}} if backend.name == "openrouter" else {"reasoning_effort": effort}


CACHE_MARK = {"type": "ephemeral"}      # Anthropic's five-minute cache; every read renews it


def wants_cache_marks(backend: Backend) -> bool:
    """Whether the endpoint caches only what a request marks. Anthropic's models do, and through OpenRouter the mark
    is a `cache_control` on a content part; the other families cache a repeated prefix on their own."""
    return backend.name == "openrouter" and backend.model.startswith("anthropic/")


def _marked(message: dict) -> dict:
    """A copy of the message with a cache breakpoint on its last text part (a string content becomes one part); the
    message itself when it has no text to carry one."""
    content = message.get("content")
    if isinstance(content, str) and content:
        parts = [{"type": "text", "text": content}]
    elif isinstance(content, list):
        parts = [dict(c) for c in content]
    else:
        return message
    for part in reversed(parts):
        if part.get("type") == "text":
            part["cache_control"] = dict(CACHE_MARK)
            return dict(message, content=parts)
    return message


def cache_marks(messages: list[dict]) -> list[dict]:
    """The conversation with cache breakpoints for an endpoint that caches only what is marked, as a copy.

    Three marks, of the four Anthropic allows: the end of the opening message (the system prompt, the brief and the
    reference views: read from the cache by every call), the end of the previous request, and the end of this one.
    The previous request's mark sits exactly where that request wrote the cache, so this request reads everything up
    to there and writes only the turn since; a single moving mark relies on the cache's 20-block look-back, and one
    turn here (an evaluation's pictures, each a caption and an image) can be longer than that. The marks hold only
    while the conversation's earlier messages are left alone: rewriting one (pruning its pictures) moves every later
    byte out of the cache."""
    ends = [i for i in range(len(messages) - 1) if messages[i + 1].get("role") == "assistant"] + [len(messages) - 1]
    marks = {ends[0], *ends[-2:]}
    return [_marked(m) if i in marks else m for i, m in enumerate(messages)]


def image_to_data_url(path: str, max_dim: int = IMAGE_MAX_DIM, quality: int = IMAGE_QUALITY) -> str:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if max(im.size) > max_dim:
        s = max_dim / max(im.size)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


RETRY_STATUS = (408, 409, 425, 429, 500, 502, 503, 504)


def _post(url: str, payload: dict, key: str | None, timeout_s: float, attempts: int = 4,
          sleep=time.sleep, opener=urllib.request.urlopen) -> dict:
    """POST with retry on transport faults (429 / 5xx / unreachable / timeout): exponential backoff 2, 4, 8 s with
    jitter, `attempts` tries in all. A 4xx outside the retryable set raises at once."""
    data = json.dumps(payload).encode("utf-8")
    last: Exception | None = None
    for i in range(max(1, attempts)):
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if key:
            req.add_header("Authorization", f"Bearer {key}")
        try:
            with opener(req, timeout=timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:800]
            last = LLMError(f"HTTP {e.code} from {url}: {body}")
            if e.code not in RETRY_STATUS:
                raise last from e
        except urllib.error.URLError as e:
            last = LLMError(f"cannot reach {url}: {e.reason}")
        except TimeoutError:
            last = LLMError(f"timeout after {timeout_s}s from {url}")
        if i + 1 < attempts:
            sleep(2.0 ** (i + 1) + random.uniform(0, 1))
    raise LLMError(f"{last} (after {attempts} attempts)")


def shape_payload(backend: Backend, messages: list[dict], temperature: float | None, max_tokens: int | None) -> dict:
    """The base chat payload, adapted to what the endpoint accepts (see REASONING_MODEL_RE). A None temperature or
    max_tokens is left out, so the endpoint's own default applies."""
    payload: dict = {"model": backend.model, "messages": messages}
    reasoning = REASONING_MODEL_RE.match(backend.model)
    if max_tokens is not None:
        payload["max_completion_tokens" if reasoning else "max_tokens"] = max_tokens
    if temperature is not None and not reasoning:
        payload["temperature"] = temperature
    return payload


def chat(backend: Backend, messages: list[dict], temperature: float | None = 0.3, max_tokens: int | None = 4096,
         json_mode: bool = False, extra: dict | None = None, tools: list[dict] | None = None, poster=_post) -> dict:
    """Returns {"text": str, "message": the assistant message (with any tool_calls), "usage": dict, "raw": dict}."""
    payload = shape_payload(backend, messages, temperature, max_tokens)
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    if tools:
        payload["tools"] = tools
    if extra:
        payload.update(extra)
    raw = poster(backend.url + "/chat/completions", payload, backend.key, backend.timeout_s)
    try:
        msg = raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise LLMError(f"malformed chat response: {str(raw)[:400]}") from e
    return {"text": msg.get("content") or "", "message": msg, "usage": raw.get("usage") or {}, "raw": raw}


# -- streamed, with an idle watchdog: the agent's calls --------------------------------------------------------------
#
# A request timeout cannot tell a long reasoning reply from a hung upstream; the gap between stream chunks can.
# Measured on an earlier tool loop: one call took 1210 s, two whole client timeouts and then an answer at normal
# speed, while good replies ran up to 234 s.

CHAT_IDLE_S = 90.0          # no chunk for this long and the attempt is dropped
REASONING_TEXT_KEYS = ("text", "summary", "data")     # the fields of a reasoning_details item that arrive in pieces
CHAT_RETRIES = 2            # attempts after the first
TRANSPORT_SLACK_S = 30.0    # the socket timeout sits this far past the idle limit: it only reaps a dropped attempt
CHAT_BEAT_S = 60.0          # a call still streaming says so in the log this often


class ChatStalled(LLMError):
    """A streamed attempt that stopped producing chunks (`idle`), or ran past its cap."""

    def __init__(self, message: str, idle: bool):
        super().__init__(message)
        self.idle = idle


class OutOfTime(LLMError):
    """The time the caller gave one call, its retries included, ran out."""


def replies(chunk: dict) -> bool:
    """Whether a stream chunk carries part of the reply (text or a tool call) and not only thinking."""
    deltas = [c.get("delta") or {} for c in chunk.get("choices") or []]
    return any(d.get("content") or d.get("tool_calls") for d in deltas)


def sse_chunks(lines: Iterable) -> Iterator[dict]:
    """The JSON chunks of an OpenAI-style event stream. Only `data:` lines count: blank separators and `: keep-alive`
    comments are skipped, `[DONE]` ends it, and a chunk that carries an `error` raises."""
    for raw in lines:
        line = (raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw).strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError as e:
            raise LLMError(f"broken stream chunk: {data[:200]!r}") from e
        if not isinstance(chunk, dict):
            continue
        if chunk.get("error"):
            raise LLMError(f"error in the stream: {str(chunk['error'])[:400]}")
        yield chunk


def assemble_chunks(chunks: Iterable[dict], on_chunk: Callable[[dict], None] | None = None) -> tuple[dict, dict]:
    """Stream chunks -> (assistant message, usage) in the shape the blocking call returns: content concatenated, tool
    calls merged by `index` (id and name from the first delta that has them, arguments concatenated), usage from the
    last chunk that carries one. The readable `reasoning` deltas are dropped; OpenRouter's `reasoning_details` are
    kept, merged by `index` (text pieces concatenated, the signature from the chunk that has it): a model that
    thought before a tool call wants those blocks back, unmodified, with the tool's result."""
    content: list[str] = []
    calls: dict[int, dict] = {}
    thoughts: dict[int, dict] = {}
    usage: dict = {}
    for chunk in chunks:
        if on_chunk:
            on_chunk(chunk)
        if chunk.get("usage"):
            usage = dict(chunk["usage"])
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            for n, rd in enumerate(delta.get("reasoning_details") or []):
                slot = thoughts.setdefault(int(rd.get("index", n) or 0), {})
                for k, v in rd.items():
                    if k in REASONING_TEXT_KEYS and isinstance(v, str):
                        slot[k] = slot.get(k, "") + v
                    elif v is not None or k not in slot:
                        slot[k] = v
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(int(tc.get("index") or 0), {"id": "", "type": "function",
                                                                    "function": {"name": "", "arguments": ""}})
                slot["id"] = slot["id"] or tc.get("id") or ""
                fn = tc.get("function") or {}
                slot["function"]["name"] = slot["function"]["name"] or fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
    text = "".join(content)
    msg: dict = {"role": "assistant", "content": text}
    if calls:
        msg.update(content=text or None, tool_calls=[calls[i] for i in sorted(calls)])
    if thoughts:
        msg["reasoning_details"] = [thoughts[i] for i in sorted(thoughts)]
    return msg, usage


def consume_with_watchdog(open_stream: Callable[[], tuple[Iterable[dict], Callable[[], None] | None]], *,
                          idle_s: float, max_s: float, poll_s: float | None = None, beat_s: float = 0.0,
                          log: Callable[[str], None] = lambda s: None) -> tuple[dict, dict]:
    """Run one streamed attempt in a thread and give up on it from this one. The stream's socket read blocks, and
    keep-alive comments reset a transport read timeout without yielding a chunk, so the only reliable judge of a hung
    upstream is the time since the last parsed chunk. The request itself (`open_stream`) runs in the thread too, so a
    hang before the first byte counts as idle. On a stall the stream is closed from a third thread (closing a
    response waits on the lock its blocked read holds) and the attempt abandoned: ChatStalled, for the caller to
    retry. `max_s` caps one attempt even while chunks trickle in. Every `beat_s` (0 for never) an attempt still
    running logs how long it has run and whether it has begun its reply: a model that thinks for ten minutes and a
    hung one look the same from outside otherwise. -> (message, usage)."""
    state: dict = {"closer": None, "result": None, "error": None, "chunks": 0, "reply": 0}
    done = threading.Event()
    t0 = time.monotonic()
    last = [t0]
    beat = t0

    def on_chunk(chunk: dict) -> None:
        last[0] = time.monotonic()
        state["chunks"] += 1
        state["reply"] += replies(chunk)

    def work() -> None:
        try:
            chunks, state["closer"] = open_stream()
            state["result"] = assemble_chunks(chunks, on_chunk)
        except BaseException as e:  # handed to the waiting thread
            state["error"] = e
        finally:
            done.set()

    threading.Thread(target=work, daemon=True).start()
    poll = poll_s if poll_s is not None else max(0.01, min(1.0, idle_s / 10))
    while not done.wait(poll):
        now = time.monotonic()
        idle = now - last[0] >= idle_s
        why = (f"no stream chunk for {idle_s:g} s" if idle else
               f"still streaming at the {round(max_s, 1):g} s cap" if now - t0 >= max_s else "")
        if why:
            if state["closer"]:
                threading.Thread(target=state["closer"], daemon=True).start()
            raise ChatStalled(f"{why} ({state['chunks']} chunks in)", idle)
        if beat_s and now - beat >= beat_s:
            beat = now
            said = ("no chunk yet" if not state["chunks"] else
                    f"{state['chunks']} chunks, " + (f"{state['reply']} of them the reply" if state["reply"]
                                                     else "all of them thinking"))
            log(f"  the model call is {now - t0:.0f} s in: {said}")
    if state["error"] is not None:
        raise state["error"]
    if not state["chunks"]:
        raise LLMError("the stream ended without a reply")
    return state["result"]


def chat_stream(backend: Backend, messages: list[dict], max_tokens: int | None = None, extra: dict | None = None,
                tools: list[dict] | None = None, *, idle_s: float = CHAT_IDLE_S, retries: int = CHAT_RETRIES,
                opener=urllib.request.urlopen, sleep=time.sleep, log: Callable[[str], None] = lambda s: None,
                poll_s: float | None = None, budget_s: float | None = None, beat_s: float = CHAT_BEAT_S) -> dict:
    """`chat` as a stream under the watchdog: an attempt with no chunk for `idle_s` is dropped and retried at once
    instead of holding the caller for the whole of `backend.timeout_s`, which here caps one attempt. No temperature
    is sent. Transport faults back off as `_post` does; a 4xx outside the retryable set raises at once.

    `budget_s` is the time the whole call may take, retries included (None for no limit): an attempt is capped at
    what is left of it, and when it is used up the call raises OutOfTime instead of trying again. A caller with a
    deadline hands down what it has left, so no attempt is cut short only to be repeated past the deadline.
    Returns {"text": str, "message": the assistant message (with any tool_calls), "usage": dict}."""
    payload = shape_payload(backend, messages, None, max_tokens)
    if tools:
        payload["tools"] = tools
    if extra:
        payload.update(extra)
    payload.update(stream=True, stream_options={"include_usage": True})
    url = backend.url + "/chat/completions"
    data = json.dumps(payload).encode("utf-8")

    def open_stream():
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "text/event-stream")
        if backend.key:
            req.add_header("Authorization", f"Bearer {backend.key}")
        resp = opener(req, timeout=idle_s + TRANSPORT_SLACK_S)
        return sse_chunks(resp), resp.close

    last: Exception | None = None
    t0 = time.monotonic()

    def left() -> float:
        return float("inf") if budget_s is None else budget_s - (time.monotonic() - t0)

    def out_of_time() -> OutOfTime:
        return OutOfTime(f"the {budget_s:.0f} s left for the model call ran out" + (f": {last}" if last else ""))

    for i in range(retries + 1):
        if left() <= 0:
            raise out_of_time()
        delay = 2.0 ** (i + 1) + random.uniform(0, 1)
        try:
            msg, usage = consume_with_watchdog(open_stream, idle_s=idle_s, max_s=min(backend.timeout_s, left()),
                                               poll_s=poll_s, beat_s=beat_s, log=log)
            return {"text": msg.get("content") or "", "message": msg, "usage": usage}
        except ChatStalled as e:
            last, delay = e, 0.0
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:800]
            last = LLMError(f"HTTP {e.code} from {url}: {body}")
            if e.code not in RETRY_STATUS:
                raise last from e
        except urllib.error.URLError as e:
            last = LLMError(f"cannot reach {url}: {e.reason}")
        except (OSError, http.client.HTTPException) as e:
            last = LLMError(f"the stream from {url} broke: {type(e).__name__}: {e}")
        except LLMError as e:
            last = e
        if left() <= 0:
            raise out_of_time()
        if i < retries:
            log(f"  model call attempt {i + 1} failed ({last}); retrying" + (f" in {delay:.0f} s" if delay else ""))
            if delay:
                sleep(delay)
    hint = ("; an endpoint that thinks in silence before its first chunk needs a longer idle limit"
            if isinstance(last, ChatStalled) and last.idle else "")
    raise LLMError(f"{last} (after {retries + 1} attempts{hint})")


def extract_json(text: str) -> dict:
    """Tolerant JSON extraction: strips code fences and takes the outermost {...}."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        raise LLMError(f"no JSON object in response: {text[:300]!r}")
    return json.loads(t[start:end + 1])
