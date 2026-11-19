"""Production decisions for reference-led projects. Legacy contracts use their original model fields."""
import json
import hashlib
import os
from pathlib import Path
import tempfile

COLOURS = ("vertex", "texture", "atlas", "multiply")
FILE = "production-decisions.json"
COLOUR_FIELDS = ("textured", "texture_mode", "texture_from", "vertex_colors")


def decision_file(project):
    # A one-off --target run must not consume the default run's open choices.
    if project.chosen_target and project.chosen_target != project.data.get("target"):
        key = hashlib.sha256(project.chosen_target.encode("utf-8")).hexdigest()
        return project.root / f"production-decisions-{key}.json"
    return project.root / FILE


def resolve(project, spec, target):
    policy = project.data["production"]
    if not isinstance(policy, dict) or set(policy) - {"colour"}:
        raise ValueError("production accepts colour; user budgets belong in model.max_triangles")
    file = decision_file(project)
    chosen = json.loads(file.read_text(encoding="utf-8")) if file.is_file() else {}
    if not isinstance(chosen, dict) or set(chosen) - {"colour", "max_triangles", "reason"}:
        raise ValueError("invalid production decision file")
    rows = {}
    cap = spec.get("max_triangles")
    if cap is not None:
        rows["max_triangles"] = {"value": cap, "source": "user"}
    elif target:
        rows["max_triangles"] = {"value": target["max_triangles"], "source": "preset " + target["name"]}
    colour = policy.get("colour")
    if any(key in spec for key in COLOUR_FIELDS):
        if colour is not None:
            raise ValueError("choose production.colour or the model colour fields, not both")
        from . import glb
        colour = glb.texture_mode(spec)
        if glb.texture_from(spec) == "script":
            colour = "texture"
    if colour is not None:
        rows["colour"] = {"value": colour, "source": "user"}
    for key in ("max_triangles", "colour"):
        if key not in rows:
            rows[key] = {"value": chosen.get(key), "source": "agent"}
    check_values({k: r["value"] for k, r in rows.items() if r["value"] is not None})
    return {"choices": rows, "reason": chosen.get("reason", "")}


def check_values(values):
    if "colour" in values and values["colour"] not in COLOURS:
        raise ValueError("colour must be one of " + ", ".join(COLOURS))
    if "max_triangles" in values:
        cap = values["max_triangles"]
        if type(cap) is not int or cap <= 0:
            raise ValueError("max_triangles must be a positive integer")


def effective(project, spec, target):
    from . import target as target_mod
    report = resolve(project, spec, target)
    rows = report["choices"]
    cap = rows["max_triangles"]["value"]
    if cap is not None:
        if spec.get("min_triangles", 0) > cap:
            raise ValueError("model.min_triangles exceeds the resolved production cap")
        spec["max_triangles"] = cap
        if target:
            target = dict(target, max_triangles=cap)
    colour = rows["colour"]["value"]
    if colour and not any(key in spec for key in COLOUR_FIELDS):
        if colour in ("vertex", "multiply"):
            spec["vertex_colors"] = True
        if colour != "vertex":
            spec.update(textured=True, texture_from="script" if colour == "texture" else "plan")
            if colour == "multiply":
                spec["texture_mode"] = "multiply"
    spec["production"] = report
    return target_mod.apply(spec, target)


def pending(report):
    return [key for key, row in report["choices"].items() if row["value"] is None]


def describe(report):
    lines = [f"{key}: {row['value'] if row['value'] is not None else 'pending'} ({row['source']})"
             for key, row in report["choices"].items()]
    if pending(report):
        lines.append("Inspect the reference, then call choose_production with only the pending choices "
                     "and a reason naming the hardest marking at delivery scale.")
    elif report["reason"]:
        lines.append(report["reason"])
    return "Production contract\n" + "\n".join(lines)


def choose(project, reason, colour=None, max_triangles=None):
    if "production" not in project.data:
        raise ValueError("this project uses a fixed model contract")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("give the reference or delivery reason for the decision")
    decision = {k: v for k, v in (("colour", colour), ("max_triangles", max_triangles)) if v is not None}
    check_values(decision)
    file = decision_file(project)
    decision["reason"] = reason.strip()
    if file.exists() and json.loads(file.read_text(encoding="utf-8")) == decision:
        return project.spec
    required = pending(project.spec["production"])
    if not required or set(required) != set(decision) - {"reason"}:
        raise ValueError("set exactly the pending choices: " + (", ".join(required) or "none"))
    if max_triangles is not None and project.spec.get("min_triangles", 0) > max_triangles:
        raise ValueError("choose a ceiling at least as high as model.min_triangles")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=project.root, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(decision, tmp, indent=2)
    try:
        # Publish a complete decision once, including when builders start concurrently.
        try:
            if os.name == "nt":
                os.rename(tmp_path, file)  # Windows rename refuses an existing destination.
            else:
                os.link(tmp_path, file)
        except FileExistsError:
            if json.loads(file.read_text(encoding="utf-8")) != decision:
                raise ValueError("another production decision is already recorded") from None
    finally:
        tmp_path.unlink(missing_ok=True)
    return project.spec
