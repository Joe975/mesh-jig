"""Four fresh workflow arms with request-level accounting; private outputs only."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from . import agent, cpu, evaluate, gauntlet, project, sheet

ARMS = {"gauntlet": "none", "mesh-jig": None,
        "checks-builder": "builder", "checks-both": "both"}
PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
MODELS = (gauntlet.MODEL, "claude-sonnet-5-5", "claude-opus-5-5")


def estimate_cost(usage, model=gauntlet.MODEL):
    """USD list-price estimate, per request (not invoice); no caching requested.

    Compatibility API prompt_tokens includes cached tokens. Refuse unknown cache
    writes rather than guessing their TTL. Reasoning is already in completion_tokens.
    """
    if "prompt_tokens" not in usage or "completion_tokens" not in usage:
        return None
    details = usage.get("prompt_tokens_details") or {}
    if details.get("cache_write_tokens") or usage.get("cache_creation_input_tokens"):
        return None
    prompt, output = usage["prompt_tokens"], usage["completion_tokens"]
    cached = details.get("cached_tokens", 0) or 0
    if not 0 <= cached <= prompt or output < 0:
        return None
    if model == gauntlet.MODEL:
        multiplier = 5 if prompt > 100_000 else 1
        input_rate, output_rate, cache_rate = .10 * multiplier, .50 * multiplier, .01 * multiplier
    elif model == "claude-sonnet-5-5":
        input_rate, output_rate, cache_rate = 2., 10., .10
    elif model == "claude-opus-5-5":
        input_rate, output_rate, cache_rate = 4., 20., .20
    else:
        return None
    return round(((prompt - cached) * input_rate + cached * cache_rate + output * output_rate) / 1_000_000, 8)


def effort_control(model, effort):
    'The compatibility endpoint ignores reasoning_effort. Do not pretend it sets effort.'
    if effort is None:
        return "provider default"
    if model == "claude-sonnet-5-5" and effort == "high":
        return "documented provider default high; compatibility reasoning_effort is ignored"
    if model == "claude-opus-5-5" and effort == "high":
        return "native Messages API output_config.effort=high"
    raise ValueError("explicit effort is supported only for Sonnet 5.5 high (its documented default)")


class Meter:
    def __init__(self, chat, root, model=gauntlet.MODEL, *, price_fn=estimate_cost):
        self.chat, self.root = chat, root
        self.model = model
        self.price_fn = price_fn
        self.started = time.monotonic()
        self.calls = []

    def __call__(self, messages, schemas, *left):
        started = time.monotonic()
        role = "implementor" if schemas else "reviewer"
        event = {"role": role, "started_s": round(started - self.started, 3)}
        try:
            message, usage = self.chat(messages, schemas, *left)
            event.update(usage=usage, estimated_usd=self.price_fn(usage, self.model),
                         tool_names=[t.get("function", {}).get("name") for t in message.get("tool_calls") or []])
            return message, usage
        except Exception as error:
            event.update(error=str(error), estimated_usd=None)
            raise
        finally:
            event["wall_s"] = round(time.monotonic() - started, 3)
            event["elapsed_s"] = round(time.monotonic() - self.started, 3)
            self.calls.append(event)
            gauntlet.write_json(self.root / "accounting.json", self.summary())

    def summary(self):
        known = [c["estimated_usd"] for c in self.calls if c.get("estimated_usd") is not None]
        return {"cost_kind": "estimated", "model": self.model,
                "pricing_url": PRICING_URL, "pricing_date": "2026-10-09",
                "estimated_usd": round(sum(known), 6), "unpriced_calls": len(self.calls) - len(known),
                "api_wall_s": round(sum(c["wall_s"] for c in self.calls), 3),
                "prompt_tokens": sum(c.get("usage", {}).get("prompt_tokens", 0) for c in self.calls),
                "completion_tokens": sum(c.get("usage", {}).get("completion_tokens", 0) for c in self.calls),
                "calls": self.calls}


def trajectory(proj, result, meter):
    rows = []
    rounds = {r["name"]: r for r in result.get("rounds", [])}
    names = list(rounds) if rounds else [a["name"] for a in result.get("attempts", []) if a.get("ok")]
    for name in names:
        path = proj.attempts_dir / name / "eval.json"
        if not path.is_file():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        row = {"name": name, "numeric": rec.get("numeric"),
               "failed_gates": evaluate.failed_gates(rec),
               "gate_count": len(evaluate.acceptance_gates(rec)),
               "evaluation_wall_s": rec.get("wall_s")}
        if name in rounds:
            end = rounds[name]["calls_end"]
            row.update(elapsed_s=rounds[name]["elapsed_s"], round_wall_s=rounds[name]["wall_s"])
        else:
            # The next model call begins only after tool execution has finished.
            # This records an upper bound including other tools in the same reply.
            index = names.index(name)
            following = next((i for i, c in enumerate(result.get("calls", []))
                              if c.get("evaluations", 0) >= index + 1), None)
            end = following if following is not None else len(meter.calls)
            row["elapsed_s"] = (meter.calls[following]["started_s"] if following is not None
                                else result.get("wall_s"))
            row["timing_note"] = "at next API call after evaluation; includes same-turn tools"
        priced = [c.get("estimated_usd") for c in meter.calls[:end]]
        row["cumulative_estimated_usd"] = round(sum(v for v in priced if v is not None), 6)
        row["unpriced_calls"] = sum(v is None for v in priced)
        rows.append(row)
    return rows


def sweep(source, out, *, iterations=10, workers=4, model=gauntlet.MODEL, effort=None,
          idle_s=180, call_timeout_s=600, budget_usd=None):
    if not 1 <= iterations <= 10 or not 1 <= workers <= 4:
        raise ValueError("iterations must be 1..10; workers must be 1..4")
    if model not in MODELS:
        raise ValueError("no verified pricing for the requested model")
    if idle_s <= 0 or call_timeout_s < idle_s:
        raise ValueError("call timeout must be at least the positive idle timeout")
    effort_source = effort_control(model, effort)
    if model == "claude-opus-5-5" and (effort != "high" or budget_usd is None):
        raise ValueError("Opus requires explicit high effort and a budget")
    if budget_usd is not None and (model != "claude-opus-5-5" or not 0 < budget_usd <= 50):
        raise ValueError("budget is supported only for Opus and must be in (0, 50]")
    api = gauntlet.backend(model)
    api.timeout_s = call_timeout_s
    if out.exists():
        raise ValueError("use a fresh output directory")
    gauntlet.clone_seed(source, out / "seed")
    seed = project.load(out / "seed")
    if not seed.references:
        sheet.write_refs(seed)
        seed = project.load(seed.root)
    if seed.problems():
        raise ValueError("; ".join(seed.problems()))
    hashes = None
    for arm in ARMS:
        copied = gauntlet.clone_seed(seed, out / arm)
        if hashes is not None and copied != hashes:
            raise ValueError("seeds differ")
        hashes = copied
    repo = Path(__file__).resolve().parents[2]
    source_hashes = {p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for folder in ("src", "skill") for p in (repo / folder).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    manifest = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                "model": model, "endpoint": gauntlet.URL, "iterations": iterations,
                "workers": workers, "max_tokens": gauntlet.MAX_TOKENS, "max_turns_per_iteration": 24,
                "effort": effort or "provider default", "effort_control": effort_source,
                "idle_s": idle_s, "call_timeout_s": call_timeout_s,
                "cache_requested": False,
                "seed_sha256": hashes, "harness_sha256": source_hashes, "arms": {}}
    gauntlet.write_json(out / "sweep.json", manifest)
    budget = None
    if budget_usd is not None:
        from .anthropic_sweep import Budget
        budget = Budget(out / "budget.json", budget_usd)
        manifest["budget_usd"] = budget_usd
        manifest["transport"] = "native Messages; no automatic retries"
    start = time.monotonic()

    def run_arm(arm):
        proj = project.load(out / arm)
        def log(text):
            with (proj.root / "run.log").open("a", encoding="utf-8") as f:
                f.write(text + "\n")
        if budget is not None:
            from .anthropic_sweep import chat_for
            chat = chat_for(api, budget, arm, max_tokens=gauntlet.MAX_TOKENS, log=log)
        else:
            chat = agent.chat_for(api, max_tokens=gauntlet.MAX_TOKENS, idle_s=idle_s, log=log)
        meter = Meter(chat, proj.root, model)
        print(f"START {arm}", flush=True)
        if arm == "mesh-jig":
            result = asdict(agent.run(proj, meter, attempts=iterations, max_turns=24 * iterations,
                                     model=model, url=gauntlet.URL,
                                     out=proj.root / "agent" / "control", log=log))
        else:
            result = gauntlet.run(proj, meter, iterations=iterations, checks=ARMS[arm], model=model, log=log)
        result["effort"] = effort or "provider default"
        result["effort_control"] = effort_source
        account = meter.summary()
        result["accounting"] = {k: v for k, v in account.items() if k != "calls"}
        result["trajectory"] = trajectory(proj, result, meter)
        gauntlet.write_json(proj.root / "result.json", result)
        print(f"END {arm}: {len(result['trajectory'])} evaluations; {result['wall_s']}s; "
              f"estimated ${account['estimated_usd']:.4f}; {result.get('error', '')}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_arm, arm): arm for arm in ARMS}
        for future in as_completed(futures):
            arm = futures[future]
            try:
                manifest["arms"][arm] = future.result()
            except Exception as error:
                manifest["arms"][arm] = {"error": str(error), "done": False}
            gauntlet.write_json(out / "sweep.json", manifest)
    manifest["wall_s"] = round(time.monotonic() - start, 3)
    manifest["status"] = "complete" if all(
        len(r.get("trajectory", [])) == iterations and not r.get("error")
        for r in manifest["arms"].values()) else "incomplete"
    gauntlet.write_json(out / "sweep.json", manifest)
    return manifest


def main(argv=None):
    cpu.govern()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model", choices=MODELS, default=gauntlet.MODEL)
    parser.add_argument("--effort", choices=("high",), default=None)
    parser.add_argument("--idle-seconds", type=float, default=180)
    parser.add_argument("--call-timeout-seconds", type=float, default=600)
    parser.add_argument("--budget-usd", type=float)
    args = parser.parse_args(argv)
    result = sweep(project.load(args.source.resolve()), args.out.resolve(),
                   iterations=args.iterations, workers=args.workers, model=args.model, effort=args.effort,
                   idle_s=args.idle_seconds, call_timeout_s=args.call_timeout_seconds, budget_usd=args.budget_usd)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
