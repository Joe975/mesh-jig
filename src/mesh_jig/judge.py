"""Pairwise judge: which of two candidates is closer to the reference, decided inside ONE call.

Why pairwise. A judge that returns an absolute score drifts: the same four renders scored 0.517 / 0.529 / 0.496 /
0.521 across four batches, and one candidate's unchanged renders scored 0.498 and 0.572 on another judge - 0.074
apart, enough on its own to move a headline result from +20 % to +1 %. Within a batch the standard error was about
0.011, so the drift is not sampling noise; the calls in one batch share whatever the provider is doing at that
moment. A batch cannot be compared with another batch.

A forced choice removes the quantity that drifts. The model never emits a number that has to mean the same thing
tomorrow; it says which of two images in front of it is closer, and whatever mood the provider is in applies to
both sides of that comparison equally. What survives is the ordering, which is what a search actually needs.

Two things this has to get right to be worth anything:

- **Position bias.** A model asked to choose between slot A and slot B does not choose evenly. So every
  comparison runs half its samples with the candidates swapped, and the aggregate maps each answer back through
  the assignment it was given. `self_bias` measures what is left by comparing a candidate WITH ITSELF, where the
  honest answer is 50/50 and anything else is pure bias.
- **Ties.** A judge forced to pick between two near-identical builds invents a reason. `tie` is allowed and
  counts as half a win to each side, so "cannot tell them apart" is reported rather than converted into a
  spurious ordering.
"""

from __future__ import annotations

import random
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from . import silhouette
from .llm import Backend, LLMError, chat, extract_json, image_to_data_url, reasoning_extra

HEIGHT = 420            # composite panel height in px
GAP = 24

SYSTEM = (
    "You are a strict, consistent art director comparing two low-poly game models against the concept art they "
    "were both built from. You judge shape, proportion, part placement and colour zones, and nothing else. You "
    "answer which candidate is closer to the reference. Reply with a single JSON object and no prose outside it."
)

RULES = """\
Both candidates are flat vertex-colour low-poly meshes built to the same contract, rendered on the same plain \
stage from the same camera. They differ only in the model. Never count as a difference between them: surface \
texture, weathering, panel lines, decals, markings, text, emblems, bevel detail, material sheen, shading or \
lighting style, background, camera framing or image size.

For each view, decide which candidate is a more faithful low-poly translation of the reference: whose parts are \
in the right place, whose proportions and outline match, whose colour zones sit where the reference's do.

Answer "tie" only when you genuinely cannot tell which is closer - not to be safe. If one is even slightly \
closer, say which."""


def composite(reference: str | Path, a: str | Path, b: str | Path, view: str, height: int = HEIGHT) -> Image.Image:
    """A | REFERENCE | B for one view, each cropped to its subject and scaled to one height, labelled.

    The reference sits BETWEEN the candidates on purpose. With it on the left the first live check measured a
    position bias of 0.125-0.277 - the judge picked the far panel four times out of five whatever was in it -
    because only one candidate was ever next to the thing it was being compared against. This layout gives both
    the same adjacency; the swap still runs, so the remaining bias cancels instead of having to be trusted.
    """
    panels = [("A", silhouette.subject_panel(a, height)), ("REFERENCE", silhouette.subject_panel(reference, height)),
              ("B", silhouette.subject_panel(b, height))]
    label = 28
    width = sum(p.width for _n, p in panels) + GAP * (len(panels) - 1)
    out = Image.new("RGB", (width, height + label), (255, 255, 255))
    d = ImageDraw.Draw(out)
    x = 0
    for name, panel in panels:
        out.paste(panel, (x, label))
        d.text((x + 6, 6), f"REFERENCE {view}" if name == "REFERENCE" else f"CANDIDATE {name}", fill=(0, 0, 0))
        if x:
            d.line([(x - GAP // 2, 0), (x - GAP // 2, out.height)], fill=(0, 0, 0), width=2)
        x += panel.width + GAP
    return out


def write_composites(reference: dict[str, str], a: dict[str, str], b: dict[str, str], out_dir: Path,
                     tag: str = "") -> dict[str, str]:
    """{view: composite path} for every view all three have."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for view, ref in reference.items():
        if view in a and view in b:
            p = out_dir / f"cmp{('_' + tag) if tag else ''}_{view}.png"
            composite(ref, a[view], b[view], view).save(p)
            out[view] = str(p)
    return out


def output_spec(views: list[str]) -> str:
    keys = ", ".join(f'"{v}"' for v in views)
    return (f'Reply with ONLY this JSON object, one entry per view, keyed exactly {keys}:\n'
            '{"views": {"<view>": {"why": "<the single difference that decides it>", "closer": "A"|"B"|"tie"}}}\n'
            "Give the reason first, then the verdict for that view. Judge each view on its own evidence. "
            "Output the JSON object immediately, with no reasoning before it: the reason belongs in the \"why\" "
            "field. A reply that reasons in prose first runs out of tokens and is discarded.")


def build_messages(composites: dict[str, str], brief: str, encoder=image_to_data_url) -> list[dict]:
    content: list[dict] = []
    if brief:
        content.append({"type": "text", "text": "WHAT THE MODEL MUST SHOW (brief):\n" + brief.strip()})
    content.append({"type": "text", "text": RULES})
    for view, path in composites.items():
        content.append({"type": "text", "text": f"VIEW '{view}': candidate A, then the REFERENCE, then candidate B. "
                                                "The two candidates are equally far from the reference panel."})
        content.append({"type": "image_url", "image_url": {"url": encoder(path)}})
    content.append({"type": "text", "text": output_spec(list(composites))})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def parse(d: dict, views: list[str]) -> dict:
    """{"views": {view: {"closer": "A"|"B"|"tie", "why": str}}}; every view must answer or the sample is dropped."""
    got = {str(k).strip().lower(): v for k, v in (d.get("views") or {}).items()}
    out = {}
    for v in views:
        e = got.get(v.lower())
        if not isinstance(e, dict) or "closer" not in e:
            raise ValueError(f"pairwise: no verdict for view {v!r} ({sorted(got)})")
        c = str(e["closer"]).strip().upper()
        if c not in ("A", "B", "TIE"):
            raise ValueError(f"pairwise: verdict {c!r} for {v!r} is not A, B or tie")
        out[v] = {"closer": c, "why": str(e.get("why") or "").strip()}
    return {"views": out}


def tally(samples: list[dict], views: list[str]) -> dict:
    """Win rate for the FIRST candidate, mapping every answer back through the swap it was shown with.

    Each sample carries `swapped`: when true the first candidate was shown in slot B, so an "A" verdict is a win
    for the second one. A tie counts half to each side, and `position_bias` reports how often the slot-A panel
    won regardless of which candidate sat in it - the number `self_bias` exists to drive to 0.5.
    """
    wins = ties = total = slot_a = 0
    per_view: dict[str, list[float]] = {v: [] for v in views}
    per_sample: list[float] = []
    for s in samples:
        swapped = bool(s.get("swapped"))
        this_call: list[float] = []
        for v in views:
            c = s["views"][v]["closer"]
            total += 1
            if c == "TIE":
                ties += 1
                per_view[v].append(0.5)
                this_call.append(0.5)
                continue
            if c == "A":
                slot_a += 1
            first_won = (c == "B") if swapped else (c == "A")
            wins += 1 if first_won else 0
            per_view[v].append(1.0 if first_won else 0.0)
            this_call.append(1.0 if first_won else 0.0)
        if this_call:
            per_sample.append(statistics.fmean(this_call))
    decided = total - ties
    rate = statistics.fmean(per_sample) if per_sample else 0.5
    lo, hi = interval(per_sample)
    return {"win_rate": round(rate, 4), "ci": [round(lo, 4), round(hi, 4)], "decisive": bool(lo > 0.5 or hi < 0.5),
            "wins": wins, "losses": decided - wins, "ties": ties, "judgements": total, "calls": len(per_sample),
            "position_bias": round(slot_a / decided, 4) if decided else 0.5,
            "views": {v: round(statistics.fmean(x), 4) for v, x in per_view.items() if x},
            "margin": round(2 * rate - 1, 4)}


def interval(per_sample: list[float], z: float = 1.96) -> tuple[float, float]:
    """95 % interval for the win rate, with the CALL as the unit of evidence - not the view.

    This is not fussiness. Treating each view as an independent judgement and using a binomial interval declared
    the very first live self-comparison DECISIVE at 0.312 - on two copies of the same renders. The four views in
    one call share a reply, a mood and whatever the provider is doing that second, so they are nowhere near
    independent, and a binomial interval over them understates the spread badly enough to invert a result.

    The interval is the WIDER of two estimates, so neither failure mode can hide:

    - a Wilson interval on the number of CALLS, which stays sensible when every call happens to agree (two
      identical answers have zero observed variance, and an interval built from that alone would claim
      certainty from two samples);
    - the observed spread between calls, which is the larger of the two whenever the judge is overdispersed -
      exactly what drift looks like from inside one batch.
    """
    n = len(per_sample)
    if n < 2:
        return 0.0, 1.0
    rate = statistics.fmean(per_sample)
    d = 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / d
    half = z * ((rate * (1 - rate) / n + z * z / (4 * n * n)) ** 0.5) / d
    lo, hi = centre - half, centre + half
    se = statistics.pstdev(per_sample) * (n / (n - 1)) ** 0.5 / (n ** 0.5)
    return max(0.0, min(lo, rate - z * se)), min(1.0, max(hi, rate + z * se))


def compare(backend: Backend, reference: dict[str, str], a: dict[str, str], b: dict[str, str], brief: str,
            out_dir: Path, *, samples: int = 12, chat_fn: Callable = chat, encoder=image_to_data_url,
            retries: int = 1, temperature: float = 0.3, max_tokens: int = 8000, seed: int = 0) -> dict:
    """Is `a` closer to the reference than `b`? Half the samples run with the two swapped.

    `temperature` is deliberately above zero: identical calls at temperature 0 would return one verdict `samples`
    times and the win rate would be 0 or 1 with no way to see indecision.
    """
    samples = max(2, samples)
    half = samples // 2
    order = [False] * half + [True] * (samples - half)
    random.Random(seed).shuffle(order)

    comps = {False: write_composites(reference, a, b, out_dir, "ab"),
             True: write_composites(reference, b, a, out_dir, "ba")}
    if not comps[False]:
        raise ValueError("pairwise: no view is present in the reference and both candidates")
    views = list(comps[False])
    msgs = {k: build_messages(v, brief, encoder) for k, v in comps.items()}

    def one(i: int) -> dict:
        swapped = order[i]
        err = ""
        usage = {"total_tokens": 0, "cost": 0.0}
        for _ in range(retries + 1):
            try:
                out = chat_fn(backend, msgs[swapped], temperature=temperature, max_tokens=max_tokens,
                              json_mode=True, extra=reasoning_extra(backend))
            except LLMError as e:
                err = f"LLMError: {e}"
                continue
            u = out.get("usage") or {}
            usage["total_tokens"] += int(u.get("total_tokens") or 0)
            usage["cost"] += float(u.get("cost") or 0.0)
            try:
                res = parse(extract_json(out["text"]), views)
                res["swapped"] = swapped
                return {"result": res, "usage": usage}
            except (ValueError, KeyError, TypeError, LLMError) as e:
                err = f"{type(e).__name__}: {e}"
        return {"error": err, "usage": usage}

    with ThreadPoolExecutor(max_workers=max(1, min(samples, 10))) as pool:
        runs = list(pool.map(one, range(samples)))
    good = [r["result"] for r in runs if "result" in r]
    if not good:
        raise ValueError("pairwise: every sample failed: " + "; ".join(r.get("error", "") for r in runs))
    res = tally(good, views)
    res["samples"] = len(good)
    res["errors"] = [r["error"] for r in runs if "error" in r]
    res["usage"] = {"total_tokens": sum(r["usage"]["total_tokens"] for r in runs),
                    "cost": round(sum(r["usage"]["cost"] for r in runs), 8)}
    res["model"] = backend.model
    res["reasons"] = {v: [s["views"][v]["why"] for s in good[:3] if s["views"][v]["why"]] for v in views}
    return res


def better(res: dict) -> bool:
    """The promotion rule: the first candidate won, and DECISIVELY. `decisive` is false whenever the interval around
    the win rate still contains a draw, so a candidate the judge cannot separate from its incumbent is refused rather
    than kept on a lean."""
    return bool(res["decisive"] and res["win_rate"] > 0.5)


def self_bias(backend: Backend, reference: dict[str, str], candidate: dict[str, str], brief: str, out_dir: Path,
              **kw) -> dict:
    """Compare a candidate with ITSELF. The honest answer is 50/50; whatever comes back is the judge's own bias.

    This is the cheapest check in the file and the one that decides whether any other number here means
    anything: the two panels are the same image, so every verdict other than a tie is invented.
    """
    return compare(backend, reference, candidate, candidate, brief, out_dir, **kw)


def summary(res: dict, name_a: str = "A", name_b: str = "B") -> str:
    """The verdict as the agent reads it."""
    lines = [f"pairwise {name_a} vs {name_b}: win rate {res['win_rate']:.2f} (95 % interval {res['ci'][0]:.2f}-{res['ci'][1]:.2f}, "
             f"{res['calls']} calls, {res['ties']} ties of {res['judgements']} judgements, position bias {res['position_bias']:.2f}) "
             f"-> {'DECISIVE' if res['decisive'] else 'not decisive'}"
             + (f", {name_a} is closer" if better(res) else f", {name_b} is closer" if res["decisive"] else "")]
    for view, why in (res.get("reasons") or {}).items():
        if why:
            lines.append(f"- {view} ({res['views'].get(view, 0.5):.2f}): " + " | ".join(why))
    if res.get("errors"):
        lines.append(f"{len(res['errors'])} sample(s) failed: {res['errors'][0][:200]}")
    return "\n".join(lines)


__all__ = ["composite", "write_composites", "build_messages", "parse", "tally", "interval", "compare", "better",
           "self_bias", "summary"]
