"""What a model is for: the engine budget a build is held to and the limits its mesh is read against.

A target is a name and a set of limits. `max_triangles` is the contract's (a build over it fails, `glb.check`); the
rest are what `readiness` reads a built GLB against, beside the fidelity score and not in it. Two ship with the
package, for one character in Godot on a VR headset; a project's `targets` block changes a limit or adds a name,
and its `target` key (or `--target`) says which one a build is for. A project that names none is built and read
as before."""
from __future__ import annotations

import math

# One stylised character among others in a VR scene. The triangle and material limits follow VRChat's avatar
# performance ranks ("Good" on Quest, "Excellent" on PC), the most used public numbers for one VR character; the
# texture sizes are what those budgets usually carry. The surface limits are about what the rigging and texturing
# stages need, not about the headset, and are the same for both but for buried triangles, which the larger budget
# can better afford. None was measured on a headset here.
SURFACE = {
    "max_buried": 0.15,          # share of triangles no outside view can see: sunk inside another part
    "max_pieces": 6,             # connected pieces that hold nine tenths of the surface: a body and its clothes
    "max_open_edges": 0.01,      # share of edges with one face, or more than two: holes and fins
    "max_slivers": 0.02,         # share of triangles thinner than 1 to 20
    "min_symmetry": 0.80,        # share of vertices with a mirror twin across X = 0: weights and UVs mirror
}
BUILTIN = {
    "quest": {"for": "a standalone mobile VR headset (Quest), in Godot", "max_triangles": 10000, "max_materials": 1,
              "texture_px": 1024, **SURFACE},
    "pcvr": {"for": "PC VR, in Godot", "max_triangles": 32000, "max_materials": 2, "texture_px": 2048,
             **{**SURFACE, "max_buried": 0.25}},
}
WHOLE = ("max_triangles", "max_materials", "texture_px", "max_pieces")
SHARES = tuple(k for k in SURFACE if k not in WHOLE)
KEYS = ("for",) + WHOLE + SHARES


class TargetError(ValueError):
    pass


def stated(data: dict) -> dict:
    """The project's own `targets` block ({} when it has none or it is not an object)."""
    block = data.get("targets")
    return block if isinstance(block, dict) else {}


def names(data: dict) -> list[str]:
    return sorted(set(BUILTIN) | set(stated(data)))


def problems(data: dict, chosen: str | None = None) -> list[str]:
    """What is wrong with a project's `target` and `targets`, and with the name a command chose; [] when fine."""
    out = []
    block = data.get("targets")
    if block is not None and not isinstance(block, dict):
        return ["targets must be an object: a target's name to the limits it changes"]
    for name, limits in stated(data).items():
        if not isinstance(limits, dict):
            out.append(f"targets.{name} must be an object of limits")
            continue
        for key, value in limits.items():
            number = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            if key == "for":
                if not isinstance(value, str):
                    out.append(f"targets.{name}.for must be text")
            elif key in WHOLE:
                if not (number and value == int(value) and value > 0):
                    out.append(f"targets.{name}.{key} must be a whole number above 0")
            elif key in SHARES:
                if not (number and 0 <= value <= 1):
                    out.append(f"targets.{name}.{key} must be a share from 0 to 1")
            else:
                out.append(f"targets.{name}.{key} is not a limit a target has ({', '.join(KEYS[1:])})")
        if name not in BUILTIN and "max_triangles" not in limits:
            out.append(f"targets.{name} needs max_triangles: the limit a build for it is held to")
    for where, name in (("target", data.get("target")), ("--target", chosen)):
        if name is not None and (not isinstance(name, str) or name not in names(data)):
            out.append(f"{where} {name!r} is not a target: {', '.join(names(data))}")
    return out


def resolve(data: dict, chosen: str | None = None) -> dict | None:
    """The target a build is for: `chosen` (a command's `--target`), else the project's `target`, else None. The
    package's limits with the project's own written over them, and `name`; `source` says whose limits they are."""
    name = chosen or data.get("target")
    if name is None:
        return None
    bad = problems(data, chosen)
    if bad:
        raise TargetError("; ".join(bad))
    own = stated(data).get(name) or {}
    limits = {"max_materials": 1, **SURFACE, **BUILTIN.get(name, {}), **own}     # a target of the project's own
    source = ("the package's" if not own else "the project's" if name not in BUILTIN
              else "the package's, with the project's own " + ", ".join(sorted(own)))
    return {"name": name, "source": source, **limits}


def apply(spec: dict, target: dict | None) -> dict:
    """The model contract as a build for `target` is held to: its triangle ceiling, and the target itself under
    `target` for the tools that read the rest. A floor above the ceiling gives way to it."""
    if not target:
        return spec
    out = {**spec, "target": target, "max_triangles": int(target["max_triangles"])}
    if out.get("min_triangles", 0) > out["max_triangles"]:
        out["min_triangles"] = 0
    return out


def facts(target: dict) -> str:
    """The target as the brief says it to a builder."""
    textured = f", one {target['texture_px']} px texture at most when the contract is textured" if target.get("texture_px") else ""
    return (
        f"- TARGET: {target['name']}, {target.get('for') or 'a stated budget'} ({target['source']} limits). At most "
        f"{target['max_triangles']} triangles per GLB: a build over that fails. Spend them where an outline or a face "
        "shows them, not on smooth round parts.\n"
        "- Every evaluation also reads the mesh for the stages after this one (rigging, texturing, the engine). These "
        "are reported and counted beside the score; they do not fail a build:\n"
        f"    buried triangles at most {target['max_buried']:.0%}: a triangle no outside view can see, because the part "
        "is sunk inside another, is paid for and never drawn. Join parts where they meet; do not overlap whole shapes.\n"
        f"    few pieces: nine tenths of the surface in at most {target['max_pieces']} connected pieces. Head, torso "
        "and limbs are ONE skin, which a rig can weight; clothing and accessories may be pieces of their own. Join "
        "an arm at the shoulder and a leg at the hip only: below that, leave a gap between arm and flank and between "
        "the legs, or a rig cannot bend them without stretching the body (the feedback's `limbs` line reads this).\n"
        f"    closed surface: at most {target['max_open_edges']:.0%} of edges open or shared by more than two faces.\n"
        f"    slivers at most {target['max_slivers']:.0%}: triangles thinner than 1 to 20 shade and bend badly.\n"
        f"    symmetry at least {target['min_symmetry']:.0%}: build one side and mirror it across X = 0, so each "
        "vertex has a twin (a voxel remesh loses this).\n"
        f"    materials at most {target['max_materials']}{textured}.")
