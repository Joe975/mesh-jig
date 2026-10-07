"""What a reference is judged on, stated beside it: jig.json's `criteria` block.

    "criteria": {
      "views":   {"front": 1, "side": 1, "rear": 1},      the views the score is taken over, each with its weight
      "outline": {"mode": "standing", "calibration": {...}, "stations": [...]},      a calibrated outline table
      "gates":   {"outline_mm_max": 12, "zones_min": 0.5, "resemblance_min": 0.6},   limits on the measures
      "review":  {"parts": {...}, "regions": [...], ...}, named regions, parts and probes, each a gate
      "light":   "camera",                                how the build is lit when it is drawn: see mesh_jig.render
      "score":   {"overlap": 1, "colour": 1, "edges": 1, "regions": 1}       what the numeric score is a weighted mean of
    }

Every key is optional, and what a project leaves out is not judged, with one exception: with no `views` the score
falls back to top, side and rear when the project has them, else to every compared view, and the brief, `doctor`
and the feedback of an evaluation say that it is the fallback.

The settings are older than the block, and every tool reads them where they were first put (`scoring`, `review`,
`model.profile` and three gates in `model`). `fold` writes the block into those places, so nothing downstream reads
it twice and a jig.json written the old way reads as it did. One setting stated in both places is a project
problem. `block` is the other direction (the criteria a jig.json states, as the block) and `rewrite` moves a
jig.json's criteria into the block without changing what is measured.

`score` has no older place: it is folded to the top-level `score`, which `Project.score_mix` reads. A project that
states none is scored as every project was before the key: overlap and resemblance in equal parts.
"""

from __future__ import annotations

import math

from . import scoring

KEYS = ("views", "outline", "gates", "review", "light", "score")
# what a score may be a weighted mean of -> the field of an evaluation's `numeric` record that holds it
SCORE_TERMS = {"overlap": "ortho_iou", "zones": "zone", "colour": "colour", "edges": "edges", "regions": "regions"}
LIGHTS = ("stage", "camera")        # render.LIGHT_MODES; "stage" is what a project that states none is drawn under
# the gate as the block names it (after the ledger's column) -> the `model` field every tool reads
GATES = {"outline_mm_max": "profile_mae_max", "zones_min": "placement_min", "resemblance_min": "resemblance_min"}
# where each criterion was stated before the block, for the messages
OLDER = {"views": "scoring", "outline": "model.profile", "review": "review", "light": "model.light"}


def score_text(mix: dict) -> str:
    """A project's score in words: `the mean of overlap, colour, edges and regions`, weights named when they differ."""
    terms = [t for t in SCORE_TERMS if t in mix]
    even = len({mix[t] for t in terms}) == 1
    said = [t if even else f"{t} x{mix[t]:g}" for t in terms]
    return ("the mean of " if even else "the weighted mean of ") + (
        " and ".join([", ".join(said[:-1]), said[-1]]) if len(said) > 1 else said[0])


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def fold(data: dict) -> tuple[dict, list[str]]:
    """(jig.json with its `criteria` block written into the fields it stands for, what is wrong with the block).
    A part of the block that is wrong is reported and not folded."""
    block = data.get("criteria")
    if "score" in data:                 # the folded place of criteria.score, and nowhere a jig.json states it
        data = {k: v for k, v in data.items() if k != "score"}
    if block is None:
        return data, []
    if not isinstance(block, dict):
        return data, [f"criteria must be an object with any of {', '.join(KEYS)}"]
    problems = []
    unknown = sorted(set(block) - set(KEYS))
    if unknown:
        problems.append(f"criteria: unknown key(s) {unknown}; it takes {', '.join(KEYS)}")
    out = {k: v for k, v in data.items() if k != "criteria"}
    if "score" in block:
        mix = block["score"]
        if (not isinstance(mix, dict) or not mix or set(mix) - set(SCORE_TERMS)
                or any(not _number(w) or w < 0 for w in mix.values()) or not any(mix.values())):
            problems.append(f"criteria.score must be a mapping of any of {', '.join(SCORE_TERMS)} to weights, none "
                            'negative and one above 0, e.g. {"overlap": 1, "colour": 1, "edges": 1, "regions": 1}')
        else:
            out["score"] = {term: float(w) for term, w in mix.items() if w}
    model = dict(data["model"]) if isinstance(data.get("model"), dict) else None

    def twice(key: str, older: str) -> None:
        problems.append(f"criteria.{key} is also stated as {older}: state it once, in criteria")

    if "views" in block:
        if data.get("scoring") is not None:
            twice("views", OLDER["views"])
        try:
            scoring.weights({"views": block["views"]})
            out["scoring"] = {"views": block["views"]}
        except ValueError:
            problems.append('criteria.views must be a nonempty mapping of view names to positive weights, '
                            'e.g. {"front": 1, "side": 1, "rear": 1}')
    if "review" in block:
        if data.get("review") is not None:
            twice("review", OLDER["review"])
        out["review"] = block["review"]
    if "outline" in block and model is not None:
        if model.get("profile") is not None:
            twice("outline", OLDER["outline"])
        model["profile"] = block["outline"]
    if "light" in block:
        if model is not None and model.get("light") is not None:
            twice("light", OLDER["light"])
        if block["light"] not in LIGHTS:
            problems.append(f"criteria.light must be one of {', '.join(LIGHTS)}")
        elif model is not None:
            model["light"] = block["light"]
    gates = block.get("gates")
    if gates is not None and not isinstance(gates, dict):
        problems.append(f"criteria.gates must be an object with any of {', '.join(GATES)}")
    elif gates:
        for name, limit in gates.items():
            if name not in GATES:
                problems.append(f"criteria.gates: unknown gate {name!r}; it takes {', '.join(GATES)}")
            elif not _number(limit):
                problems.append(f"criteria.gates.{name} must be a number")
            elif model is not None:
                if model.get(GATES[name]) is not None:
                    twice(f"gates.{name}", f"model.{GATES[name]}")
                model[GATES[name]] = limit
    if model is not None:
        out["model"] = model
    return out, problems


def sources(data: dict) -> dict[str, str]:
    """criterion -> where this jig.json states it (`criteria.views`, or the older `scoring`), for those it states."""
    block = data.get("criteria") if isinstance(data.get("criteria"), dict) else {}
    model = data.get("model") if isinstance(data.get("model"), dict) else {}
    out = {}
    for key in ("views", "outline", "review"):
        older = model.get("profile") if key == "outline" else data.get(OLDER[key])
        if key in block:
            out[key] = f"criteria.{key}"
        elif older is not None:
            out[key] = OLDER[key]
    if "light" in block:
        out["light"] = "criteria.light"
    elif model.get("light") is not None:
        out["light"] = OLDER["light"]
    if "score" in block:
        out["score"] = "criteria.score"
    if block.get("gates"):
        out["gates"] = "criteria.gates"
    elif any(model.get(old) is not None for old in GATES.values()):
        out["gates"] = "model"
    return out


def named(message: str, older: str, stated: str) -> str:
    """A validator's message, which opens with the setting's older name (`profile.mode must be ...`), opening with
    where this jig.json states it (`criteria.outline.mode must be ...`)."""
    if stated.startswith("criteria.") and message.startswith(older) and not message[len(older):len(older) + 1].isalnum():
        return stated + message[len(older):]
    return message


def block(data: dict) -> dict:
    """The criteria a jig.json states, as a `criteria` block, wherever it states them. A `scoring` preset is written
    out as the views it selects."""
    folded, _problems = fold(data)
    model = folded.get("model") if isinstance(folded.get("model"), dict) else {}
    out: dict = {}
    try:
        weights = scoring.weights(folded.get("scoring"))
    except ValueError:
        weights = None
    if weights is not None:
        out["views"] = {v: int(w) if w == int(w) else w for v, w in weights.items()}
    if model.get("profile") is not None:
        out["outline"] = model["profile"]
    gates = {name: model[old] for name, old in GATES.items() if model.get(old) is not None}
    if gates:
        out["gates"] = gates
    if folded.get("review") is not None:
        out["review"] = folded["review"]
    if model.get("light") is not None:
        out["light"] = model["light"]
    if folded.get("score"):
        out["score"] = {term: int(w) if w == int(w) else w for term, w in folded["score"].items()}
    return out


def rewrite(data: dict, fallback_views: list[str] | tuple[str, ...] = ()) -> dict:
    """jig.json with every criterion in its `criteria` block and none left where it was. `fallback_views` are stated,
    at equal weight, when the project states none: the views it is scored on already, so no number moves."""
    stated = block(data)
    if "views" not in stated and fallback_views:
        stated = {"views": dict.fromkeys(fallback_views, 1), **stated}
    out = {k: v for k, v in data.items() if k not in ("criteria", "scoring", "review", "score")}
    if isinstance(out.get("model"), dict):
        out["model"] = {k: v for k, v in out["model"].items()
                        if k not in ("profile", "light") and k not in GATES.values()}
    if stated:
        out["criteria"] = stated
    return out
