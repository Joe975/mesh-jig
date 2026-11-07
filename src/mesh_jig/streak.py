'mesh-jig streak: whether a builder that was told to keep going should stop yet.'
from __future__ import annotations

import json
from pathlib import Path

from . import comparison_context, evaluate as evaluate_mod, project as project_mod

PATIENCE = 3
MARGIN = 0.002       # what `verdict` and the command count with; the functions below take the margin they are given


def running_best(scores: list[float]) -> list[float]:
    """The best score so far, after each attempt."""
    out: list[float] = []
    for s in scores:
        out.append(max(s, out[-1]) if out else s)
    return out


def flat_streak(scores: list[float], margin: float = 0.0) -> int:
    """How many attempts at the end have not beaten, by more than `margin`, the last attempt that did. The first
    attempt always counts as an improvement."""
    streak, mark = 0, None
    for s in scores:
        if mark is None or s > mark + margin:
            streak, mark = 0, s
        else:
            streak += 1
    return streak


def rule_held_at(scores: list[float], patience: int = PATIENCE, margin: float = 0.0) -> int | None:
    """How many attempts had been made when `patience` in a row had first not improved. None when that never
    happened."""
    for n in range(1, len(scores) + 1):
        if flat_streak(scores[:n], margin) >= patience:
            return n
    return None


def _attempt_records(project: Path) -> tuple[list[dict], int]:
    """(every measured attempt in the order it was measured, how many failed builds were left out). An attempt is
    {"name", "score", "failed": the names of the gates it fails, "gates": how many it has}."""
    measured, failed = [], 0
    for f in (Path(project) / project_mod.ATTEMPTS).glob(f"*/{evaluate_mod.EVAL_FILE}"):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(rec, dict):
            continue
        score = (rec.get("numeric") or {}).get("numeric_score")
        if not rec.get("ok") or score is None:
            failed += 1
            continue
        gates = evaluate_mod.acceptance_gates(rec)
        measured.append({"name": f.parent.name, "started": str(rec.get("started") or ""), "score": float(score),
                         "failed": [str(g.get("name") or "?") for g in gates if not g.get("passed")], "gates": len(gates),
                         "_record": rec})
    return sorted(measured, key=lambda a: (a["started"], a["name"])), failed


def attempts(project: Path) -> tuple[list[dict], int]:
    """Public measured-attempt list and failed-build count, without internal provenance fields."""
    measured, failed = _attempt_records(project)
    for attempt in measured:
        attempt.pop("_record", None)
    return measured, failed


def verdict(project: Path, patience: int = PATIENCE, margin: float = MARGIN, max_attempts: int = 0) -> dict:
    """What the rule says of a project as it stands: {"stop", "why", "made", "left_out", "best", "streak", "since":
    the attempts after the last improvement, "clean": the highest scoring attempt that fails no gate, when the best
    fails one}."""
    all_tried, left_out = _attempt_records(project)
    tried = []
    excluded = 0
    if all_tried:
        latest = all_tried[-1]["_record"]
        for attempt in all_tried:
            if comparison_context.compatible(latest, attempt["_record"]):
                attempt.pop("_record", None)
                tried.append(attempt)
            else:
                excluded += 1
    measured_total = len(all_tried)
    latest_kind = comparison_context.classify(latest)[0] if all_tried else None
    scores = [a["score"] for a in tried]
    streak = flat_streak(scores, margin)
    best = max(tried, key=lambda a: a["score"]) if tried else None        # the earliest, when two tie
    passing = [a for a in tried if a["gates"] and not a["failed"]]
    clean = max(passing, key=lambda a: a["score"]) if best and best["failed"] and passing else None
    did = f"gained more than {margin:g}" if margin else "improved on the best"
    if tried and streak >= patience:
        stop, why = True, f"{patience} evaluated attempts in a row have not {did}"
    elif max_attempts and measured_total >= max_attempts:
        stop, why = True, f"the ceiling of {max_attempts} evaluated attempts is reached"
    else:
        stop, why = False, (f"{streak} in a row {'has' if streak == 1 else 'have'} not {did}; stop at {patience}"
                            if tried else "comparisons unavailable for the latest successful evaluation"
                            if latest_kind == "invalid" else "nothing has been measured yet")
    result = {"stop": stop, "why": why, "made": len(tried), "left_out": left_out, "best": best, "streak": streak,
            "since": tried[len(tried) - streak:] if streak else [], "clean": clean, "patience": patience,
            "margin": margin, "max_attempts": max_attempts, "measured_total": measured_total}
    if latest_kind == "legacy":
        result["provenance_note"] = "historical full evaluation provenance unavailable for legacy records."
    elif latest_kind == "invalid":
        result["provenance_note"] = "comparisons unavailable: the latest successful record has invalid or unsupported provenance."
    if excluded:
        result["excluded_incompatible"] = excluded
        if latest_kind == "legacy":
            result["compatibility_note"] = (f"{excluded} measured attempt{'s' if excluded != 1 else ''} excluded: readiness target/cap provenance differs or is unknown.")
        elif latest_kind == "invalid":
            result["compatibility_note"] = (f"{excluded} measured attempt{'s' if excluded != 1 else ''} excluded: latest comparison provenance is invalid or unsupported.")
        else:
            result["compatibility_note"] = (f"{excluded} measured attempt{'s' if excluded != 1 else ''} excluded: evaluation contexts differ; legacy provenance is unavailable.")
    return result


def text(v: dict) -> str:
    lines = [f"evaluated attempts: {v['made']}" + (f" ({v['left_out']} failed build{'s' if v['left_out'] != 1 else ''} not counted)"
                                                    if v["left_out"] else "")]
    if v.get("excluded_incompatible"):
        lines.append(v["compatibility_note"])
        lines.append(f"successful measured attempts overall: {v['measured_total']} (the max-attempts ceiling counts all of them)")
    best = v["best"]
    if best:
        lines.append(f"best: {best['name']} {best['score']:.4f}")
        if v["since"]:
            n, did = len(v["since"]), f"gained more than {v['margin']:g}" if v["margin"] else "improved on the best"
            lines.append(f"the last {n} {'has' if n == 1 else 'have'} not {did}: "
                         + ", ".join(f"{a['name']} {a['score']:.4f}" for a in v["since"]))
        if best["gates"]:
            lines.append(f"gates: the best fails {len(best['failed'])} of {best['gates']}"
                         + (f" ({', '.join(best['failed'])})" if best["failed"] else ""))
        if v["clean"]:
            lines.append(f"highest that passes every gate: {v['clean']['name']} {v['clean']['score']:.4f}")
        elif best["failed"]:
            lines.append("no attempt passes every gate")
    if v.get("provenance_note"):
        lines.append(v["provenance_note"])
    if v["stop"]:
        also = f", and {v['clean']['name']} as the highest that passes every gate" if v["clean"] else ""
        lines.append(f"STOP: {v['why']}." + (f" Report {best['name']}{also}." if best else ""))
    else:
        lines.append(f"KEEP GOING: {v['why']}.")
    return "\n".join(lines)


__all__ = ["PATIENCE", "MARGIN", "running_best", "flat_streak", "rule_held_at", "attempts", "verdict", "text"]
