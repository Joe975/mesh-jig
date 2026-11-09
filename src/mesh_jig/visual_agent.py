"""The default API workflow: validate each build, then get a fresh visual critique."""
from __future__ import annotations

import math
import uuid

from . import agent, anthropic_sweep, gauntlet, llm
from .gauntlet_sweep import Meter, estimate_cost


def run(proj, backend, *, attempts=5, checks="none", max_turns=0, max_tokens=None,
        max_cost=0, max_wall=0, effort=None, idle_s=900, log=print, review_effort=None, keep_better=False):
    """`review_effort`, when given, is the reviewer's own effort; the builder keeps `effort`. A reviewer call is the
    one offered no tools."""
    if attempts < 1 or max_turns < 0 or any(not math.isfinite(v) or v < 0 for v in (max_cost, max_wall)):
        raise ValueError("attempts must be positive; turn, spend and time limits must be finite and nonnegative")
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max-tokens must be positive")
    if checks not in ("none", "builder", "both"):
        raise ValueError("checks must be none, builder or both")
    native = backend.url == gauntlet.URL and backend.model in anthropic_sweep.RATES
    reported = backend.name == "openrouter"    # it prices every reply itself, in usage["cost"]
    if max_cost and not (native or reported):
        raise ValueError("a spend cap needs direct Anthropic Haiku/Sonnet/Opus 5.5 (reserved) or OpenRouter "
                         "(the cost it reports); pricing is not verified for this endpoint/model")
    # the native models all write up to 128,000 tokens a reply; elsewhere the limit is the endpoint's own business
    max_tokens = max_tokens or (anthropic_sweep.MAX_OUTPUT_TOKENS if native else gauntlet.MAX_TOKENS)
    out = agent.run_dir(proj, backend.model)
    out = out.with_name(out.name + "_visual_" + uuid.uuid4().hex[:8])
    budget = anthropic_sweep.Budget(out / "budget.json", max_cost, backend.model) if max_cost and native else None
    def chat_at(level):
        if native:
            return anthropic_sweep.chat_for(backend, budget, "visual-agent", max_tokens=max_tokens,
                                            effort=level, log=log)
        return agent.chat_for(backend, effort=level, max_tokens=max_tokens, idle_s=idle_s, log=log)

    chat = chat_at(effort)
    if review_effort and review_effort != effort:
        builder, reviewer = chat, chat_at(review_effort)

        def chat(messages, tools, *rest):
            return (builder if tools else reviewer)(messages, tools, *rest)
    meter = Meter(chat, out, backend.model,
                  price_fn=estimate_cost if native else reported_cost if reported else lambda usage, model: None)
    # outside the meter, so a call the cap refuses is never made and never counted
    capped = reported_cap(meter, max_cost) if max_cost and not native else meter
    log(f"visual review: {backend.describe()}, {attempts} iterations, checks={checks}"
        + (f", reviewer effort {review_effort}" if review_effort else "")
        + (f", cost cap ${max_cost:.2f}" if max_cost else "") + f" -> {out}")
    if checks != "none":
        log("Experimental: numeric checks are diagnostic inputs, not a visual quality verdict.")
    result = gauntlet.run(proj, capped, iterations=attempts, checks=checks, model=backend.model,
                          url=backend.url, max_tokens=max_tokens, out=out,
                          max_total_turns=max_turns, max_wall=max_wall, log=log, keep_better=keep_better)
    account = meter.summary()
    result.update(effort=effort, review_effort=review_effort or effort, provider=backend.provider,
                  accounting={k: v for k, v in account.items() if k != "calls"},
                  max_cost=max_cost, max_wall=max_wall,
                  cost_usd=account["estimated_usd"] if not account["unpriced_calls"] else None,
                  cost_kind=("unpriced" if account["unpriced_calls"] else "reported" if reported else "estimated"),
                  prompt_tokens=account["prompt_tokens"], completion_tokens=account["completion_tokens"])
    if budget:
        budget.save()
        result["budget"] = {"limit_usd": max_cost, "charged_or_reserved_usd": budget.committed()}
    gauntlet.write_json(out / "report.json", result)
    return result, out


def reported_cost(usage, model=None):
    """The cost the endpoint reported for one reply, or None when it reported none."""
    try:
        cost = float(usage.get("cost") or 0.0)
    except (TypeError, ValueError):
        return None
    return cost if cost > 0 else None


def reported_cap(chat, max_cost):
    """Hold `max_cost` against the cost the endpoint reports with each reply, by the numeric loop's rule
    (`agent.over_cost`): no further call once the dearest call so far would no longer fit. A reply with no cost
    stops the run before the next call, as the cap can no longer be held."""
    spent, dearest, unknown = 0.0, 0.0, False

    def capped(messages, tools, *rest):
        nonlocal spent, dearest, unknown
        if unknown:
            raise gauntlet.BudgetStop(f"stopped: the endpoint reported no cost for a call, so the cost cap "
                                      f"(${max_cost:.2f}) cannot be held")
        stop = agent.over_cost(spent, dearest, max_cost)
        if stop:
            raise gauntlet.BudgetStop(stop)
        message, usage = chat(messages, tools, *rest)
        cost = reported_cost(usage or {})
        if cost is None:
            unknown = True
        else:
            spent, dearest = spent + cost, max(dearest, cost)
        return message, usage
    return capped


def summary(result):
    def candidate(key):
        value = result.get(key)
        return f"{value['name']} ({value['numeric_score']:.4f})" if value else "none"
    return (f"visual review: {'complete' if result['done'] else 'INCOMPLETE'}; "
            f"{result['evaluations']} evaluated, {result['iterations_completed']} reviewed\n"
            f"Highest numeric score: {candidate('best_numeric')}\nLatest: {candidate('latest')}\n"
            + (f"Kept by the comparisons (--keep-better): {candidate('kept')}\n" if result.get("keep_better") else "")
            + "Visual quality: inspect the saved renders and review.md; no label proves a visual winner."
            + (f"\nStopped: {result['error']}" if result.get("error") else ""))
