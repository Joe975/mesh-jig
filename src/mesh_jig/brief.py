"""The project as the agent reads it before it writes a line: how a model is built here, what the contract holds it
to, the reference's outlines and colour zones as numbers, and where the pictures are.

Everything in it is derived from jig.json, brief.md and the reference views, so it never drifts from what the tools
check.
"""

from __future__ import annotations

from . import criteria, glb, outline, profile, render, silhouette
from .project import Project

BLENDER_GOTCHAS = """\
- Blender API slips that have failed builds before:
  - the Bevel modifier's clamp is `mod.use_clamp_overlap` (there is no `clamp_overlap`);
  - vectors take ONE sequence: `Vector((x, y, z))`, not `Vector(x, y, z)`; a float property (width, radius, offset)
    takes one float, never a tuple;
  - `def build():` must be at the top level of the script (not inside an if or another function);
  - bmesh.ops that remove or rebuild geometry (delete, dissolve, extrude) invalidate BMFace / BMVert references taken
    before them ("BMesh data of type BMFace has been removed"): use the faces the op returns, or re-read bm.faces.
- A failed build prints JIG_BUILD_ERROR and a traceback naming `File "<script>", line N`."""


def blender_axis(v) -> str:
    """A direction given in glTF axes, named in Blender's (Blender X, Y, Z = glTF X, -Z, Y): `+Y` when one axis carries
    it, the sign of each axis that carries a quarter of it otherwise (`-X +Y +Z`)."""
    b = (float(v[0]), -float(v[2]), float(v[1]))
    n = sum(c * c for c in b) ** 0.5 or 1.0
    return " ".join(("+" if c > 0 else "-") + axis for c, axis in zip(b, "XYZ") if abs(c) / n > 0.25)


def camera_facts(spec: dict) -> str:
    """Where each of the project's own cameras stands and how its picture is turned, in the axes the script is written
    in. The cameras are given in glTF axes in jig.json, and every run on the steampunk mech that was not told this
    spent its first attempt building the model facing away from the `front` camera."""
    said = []
    for shot in spec.get("shots") or []:
        if not (shot.get("cam_pos") and shot.get("cam_target")):
            continue
        _eye, (right, up, forward), _fov = render.camera(shot)
        said.append(f"{shot.get('name', '')} stands on the {blender_axis(-forward)} side (picture right is "
                    f"{blender_axis(right)}, picture up is {blender_axis(up)})")
    if not said:
        return ""
    return ("- Where each camera stands, in BLENDER axes (a camera on the +Y side sees the faces you build at +Y): "
            + "; ".join(said) + ".")


def weights_text(weights: dict) -> str:
    """Scoring weights as a reader sees them: `front 2, side 1`."""
    return ", ".join(f"{view} {float(w):g}" for view, w in weights.items())


def size_facts(spec: dict) -> str:
    """What to build at when the contract checks no size: the render scale is the one thing that says."""
    if any(spec.get(k) for k in ("length_m", "span_m", "height_m", "axis_extents_m")):
        return ""
    scale = render.stage_scale(spec)
    return (f"- The contract checks no size. Renders are drawn at stage_scale {scale:g}, so a model {1 / scale:g} m "
            "long or tall is one unit to the cameras: build at the size the brief gives, and near that if it gives "
            "none.")


def model_facts(spec: dict) -> str:
    """How a model is built and what the contract checks, in words. A model block without `forward` is not a craft:
    no nose, no wingspan, no mirror plane; the brief orients it."""
    craft = bool(spec.get("forward"))
    variants = glb.variant_names(spec)
    files = glb.expected_files(spec)
    length = spec.get("length_m")
    custom = [s.get("name", "") for s in spec.get("shots") or []]
    lines = [
        "HOW A MODEL IS BUILT HERE (identical for every attempt):",
        f"- You write ONE Blender Python script, {glb.script_name(spec)}, in the attempt directory. It defines `def build():` "
        "at top level and may import only bpy, bmesh, mathutils, math, random, numpy and openvdb (the last two ship in "
        "Blender's own Python). Anything else (other imports, open, exec, eval, getattr, type, dunder attributes, "
        "bpy.ops.wm / render / image and every importer and exporter, bpy.app, bpy.utils, a file reached through "
        "Blender (a bpy.data collection's load, an image's save, a filepath), and whatever in numpy or openvdb reads "
        "or writes a file) makes the runner refuse the script before running it. No picture is read into the model: "
        "its shape and colour come from the script. The runner wipes the scene, runs build(), then exports every "
        "object as a GLB.",
        (f"- The global VARIANT (a string, already defined; never assign it) names the fit being built: "
         f"{', '.join(repr(v) for v in variants)}. Share all geometry and branch only where the fits differ. Outputs: "
         f"{', '.join(files.values())}.") if variants != [""] else f"- Output: {', '.join(files.values())}.",
        "- The runner bakes modifiers and joins every mesh object into ONE mesh." if spec.get("join") else "",
        ("- Blender units are metres. Blender +Z is up and the NOSE POINTS ALONG BLENDER +Y (the export maps Blender +Y "
         "to glTF -Z, the GLB's forward). Mirror left/right across Blender X = 0. Centre the model on the origin.") if craft else
        ("- Blender units are metres and Blender +Z is up. The export maps Blender +Y to glTF -Z and Blender -Y to "
         "glTF +Z; the brief says which way the model faces. Centre the model on the origin."),
        (f"- {'Length nose-to-tail' if craft else 'Depth front to back'} along Blender Y: {length} m (tolerance "
         f"{spec.get('length_tolerance', 0.05):.0%}).") if length else "",
        (f"- {'Wingspan' if craft else 'Width'} across Blender X: {spec['span_m']} m (tolerance "
         f"{spec.get('span_tolerance', 0.10):.0%})." + (" It may be wider than the length." if craft else ""))
        if spec.get("span_m") else "",
        (f"- Standing height along Blender Z / GLB Y: {spec['height_m']} m "
         f"(tolerance {spec.get('height_tolerance', 0.05):.0%}).") if spec.get("height_m") else "",
        ("- GLB axis extents in metres: " + ", ".join(f"{axis} {float(m):g}" for axis, m in spec["axis_extents_m"].items())
         + ".") if spec.get("axis_extents_m") else "",
        size_facts(spec),
        (f"- Triangles per GLB: {spec.get('min_triangles', 0)} to {spec.get('max_triangles', 'any')}."
         if spec.get("max_triangles") or spec.get("min_triangles") else ""),
        ("- COLOUR IS VERTEX COLOUR. Materials are ignored. Give each mesh a corner colour attribute and paint its "
         "faces from a small palette:\n"
         "    attr = mesh.color_attributes.new(name='Color', type='FLOAT_COLOR', domain='CORNER')\n"
         "    for poly in mesh.polygons:\n"
         "        for li in poly.loop_indices:\n"
         "            attr.data[li].color = PALETTE[k]   # (r, g, b, 1.0), LINEAR, not the sRGB hex\n"
         f"  Fewer than {spec.get('min_colors', 1)} distinct colours fails. Build geometry with bmesh or mesh.from_pydata; "
         "keep faces outward (bmesh.ops.recalc_face_normals) and never leave zero-area faces.") if spec.get("vertex_colors") else
        "- Build geometry with bmesh or mesh.from_pydata; keep faces outward (bmesh.ops.recalc_face_normals) and never "
        "leave zero-area faces.",
        BLENDER_GOTCHAS,
        atlas_facts(spec),
    ]
    if custom:
        lines.append("- Every build is drawn on a plain light ground from fixed cameras: " + ", ".join(custom) + ". A render "
                     "is measured against the reference view of the same name; a camera with no reference view is drawn for "
                     "you to see and is not measured.")
        lines.append(camera_facts(spec))
    else:
        lines.append("- Every build is drawn on a plain light ground from four fixed cameras named like the reference views "
                     "(hero: three-quarter from front-left and above, nose lower-left; top: nose down; side: nose left; rear: "
                     "from astern), and each render is measured against its reference view."
                     + (" Each other variant is also drawn from the hero camera (hero_<variant>) for you to see; it is "
                        "not measured." if len(variants) > 1 else ""))
    return "\n".join(x for x in lines if x)


def atlas_facts(spec: dict) -> str:
    """The textured contract, when the model block asks for one: the plan to write, the UVs the script owes, and what
    is measured."""
    if not spec.get("textured"):
        return ""
    if glb.texture_from(spec) == "script":
        return painted_facts(spec)
    atlas = glb.atlas_name(spec)
    return (
        "- THIS IS A TEXTURED CONTRACT. Colour comes from ONE painted atlas. The script cannot read an image, so you "
        f"write a paint PLAN (JSON) beside the script and mesh-jig paints it to {atlas} and binds it at build time:\n"
        '    {"size": 1024, "background": "white", "ops": [{"op": "rect", "box": [u0, v0, u1, v1], "color": "<name>"}, '
        '{"op": "line", "points": [[u, v], ...], "color": "<name>", "width": 0.004}, {"op": "polygon", "points": [...]}, '
        '{"op": "ellipse", "box": [...]}, {"op": "stripes", "box": [...], "count": 6}, '
        '{"op": "speckle", "box": [...], "density": 0.02, "seed": 7}]}\n'
        "  Coordinates are 0..1 of the atlas, v DOWN; colours are the project's palette names or #rrggbb.\n"
        "  The SCRIPT must author a UV layer (mesh.uv_layers.new(); set uv for every loop) that maps each face onto the "
        "region of the atlas painted for it; a mesh without UVs fails the build.\n"
        "  Paint the reference's OWN markings where the reference has them: colour zones are compared with the reference "
        "cell by cell, and detail in the wrong place scores below no detail at all.")


def painted_facts(spec: dict) -> str:
    """The textured contract whose atlas the script paints (`texture_from: "script"`): the function it owes and what
    it is handed. What the runner does with it is mesh_jig.texel_bake."""
    return (
        "- THIS IS A TEXTURED CONTRACT, AND YOUR SCRIPT PAINTS THE TEXTURE. Author no UVs, no materials, no vertex "
        "colours and no image. Beside build(), define at top level:\n"
        "    def texture(name, position, normal):\n"
        "        ...\n"
        "        return colours\n"
        "  It is called once for each mesh object, after build(). `name` is the object's name. `position` is an (N, 3) "
        "numpy array of points on that object's surface, in the Blender metres you built it in; `normal` is the (N, 3) "
        "unit normals there. Return an (N, 3) array of sRGB colours 0..1, one row per position, or one (r, g, b) for "
        "the whole object.\n"
        "  The global PALETTE (already defined; never assign it) maps each palette name to its (r, g, b) in exactly "
        "those units: PALETTE['<name>'], with NO conversion. A texture is sRGB; only vertex colours are linear.\n"
        f"  The runner then unwraps every object into one {glb.texture_size(spec)} px atlas, asks texture about the "
        "point each texel shows, and binds the image. So a colour is a function of where a point is on the model: a "
        "marking is \"inside this ellipsoid\", \"above this height\", \"on the side that faces forward\" (normal[:, 1] > "
        "0.3), tested with numpy on the whole array at once (boolean masks, np.where, np.linalg.norm), never a Python "
        "loop over points. Name each object for what it is: texture is told the name. Use the same sizes and positions "
        "build() used, as shared constants, so the paint sits on the shape.\n"
        "  Paint the reference's OWN markings where the reference has them: colour zones are compared with the "
        "reference cell by cell, and detail in the wrong place scores below no detail at all. A soft edge between two "
        "paints (a blend over a centimetre or two), a seam, a fold and the lie of fur are worth painting. Light and "
        "shadow are not: the model is lit when it is drawn.")


def scored_views(project: Project) -> tuple[list[str], list[str]]:
    """(the views the overlap, resemblance and numeric score are taken over, the views that are compared only to be
    looked at), among the views that have both a camera and a reference: the project's `scoring` when it names one,
    else top, side and rear, else (none of those three) every compared view, as `evaluate.numeric_measures` does."""
    refs = project.references
    compared = [s["name"] for s in render.shots_for(project.spec) if s["name"] in refs]
    weights = project.scoring_weights
    scored = ([v for v in weights if v in compared] if weights is not None
              else [v for v in silhouette.ORTHO_VIEWS if v in compared] or compared)
    return scored, [v for v in compared if v not in scored]


# outline table mode -> (how a row of its feedback reads, what a baseline gets right first)
OUTLINE_READING = {
    "craft": ("your GLB's outline against the craft table above, at 11 stations along the length in each of top, "
              "side and rear. In the feedback `y+40:+7` means 7 mm too big at the station 40 mm ahead of centre; `+` "
              "is always \"yours is bigger\".",
              "the overall length, span and height"),
    "standing": ("your GLB's outline against the standing table above, at the heights it names. In the feedback a "
                 "station reads min/max/width, your model minus the reference, in mm from the fixed ground origin; a "
                 "station the reference has and yours lacks fails a gate.",
                 "the declared height and the widths at the named heights"),
    outline.MODE: ("your GLB's outline against the table above, which is read through the orthographic cameras. In "
                   "the feedback `z+364:+149/+177` means that at the station 364 mm above centre your model reaches "
                   "149 mm too far out at one end and 177 mm too far at the other (the line's header names the ends); "
                   "`-` means it falls short there.",
                   "the overall extents the outline table opens with"),
    "": ("none. There is no outline table here (no orthographic camera looks along one axis, and it is not a "
         "craft with a length), so the ledger's outline column stays `-`.",
         "the overall proportions of the reference views"),
}


def reading_facts(project: Project) -> str:
    """What the numbers of THIS project are: which views count, which outline table it has and how a row of its
    feedback reads, whether colour zones are measured, and what a baseline should get right first. The skill and the
    agent's system prompt say none of this themselves: it differs between a craft, a standing character and anything
    else, and a line that was true of the first subject was told to every builder since."""
    scored, looked = scored_views(project)
    if not scored:
        return ""
    prof = project.reference_profile()
    mode = (prof.get("mode") or "craft") if prof else ""
    reading, first = OUTLINE_READING[mode]
    weights = project.scoring_weights
    uneven = weights is not None and len(set(weights.values())) > 1
    limits = gate_facts(project.spec)
    mix = project.score_mix
    marked = [r["name"] for r in (project.review or {}).get("regions") or []]
    share = (lambda term: mix.get(term, 0) / sum(mix.values())) if mix else (lambda term: 0.0)
    lines = [
        "HOW THIS PROJECT IS MEASURED (it differs from subject to subject):",
        "- overlap, zones, resemblance and the numeric score are taken over " + ", ".join(scored)
        + (f" (weights: {weights_text(weights)})" if uneven else "") + (" ONLY." if looked else ".")
        + (f" {', '.join(looked)} {'is' if len(looked) == 1 else 'are'} compared for you to look at and "
           f"{'does' if len(looked) == 1 else 'do'} not count." if looked else "")
        + (" These are the fallback views: jig.json states no `criteria.views`." if weights is None else ""),
        "- overlap: silhouette IoU per view. A \"too flat or too wide\" line names a proportion problem.",
        ("- zones: the share of cells whose palette colour matches the reference's. Low zones with high overlap: the "
         "shape is right and the paint is in the wrong place." if project.zones else
         "- zones: not measured (there is no palette). Resemblance is edges in place alone."),
        "- colour: how close each cell's colour is to the reference's in the same place, 1 for the same colour and 0 "
        "for an unrelated one. It needs no palette, and where zones only ask which palette colour is nearest, this "
        "falls for the right colour in the wrong shade.",
        (f"- regions: the {len(marked)} marked region{'s' if len(marked) != 1 else ''} ({', '.join(marked)}), each "
         "read on its own, twice as fine as a whole view, for how close its colours are and whether its edges are in "
         "place. An evaluation lists them weakest first and draws each one enlarged beside the reference's "
         "(`region_<name>.png`).") if marked else "",
        ("- resemblance: zones plus edges in place. numeric score: the mean of overlap and resemblance; rank attempts "
         "on it.") if not mix else
        (f"- numeric score: {criteria.score_text(mix)}; rank attempts on it. Overlap is {share('overlap'):.0%} of it"
         + (f" and the marked regions {share('regions'):.0%}" if share("regions") else "")
         + ". resemblance (zones plus edges in place) is reported beside it."),
        "- outline error (mm): " + reading + " It is reported beside the score, not in it.",
        ("- light: every view is lit by white lights that turn with its camera, from the camera's upper left. A face "
         "that looks straight at a camera is drawn in exactly the colour it stores, so a part painted its palette "
         "colour reads as that colour in every view: never lighten or darken a paint to suit a render.")
        if project.spec.get("light") == "camera" else "",
        ("- gates this reference sets (each is reported as passed or FAILED beside the score; a failed gate does not "
         "fail the attempt): " + ", ".join(limits) + ".") if limits else "",
        (f"- Start with a baseline that only gets {first} right, then work part by part"
         + (" against the outline numbers" if mode else "") + " before touching colour: the outline is what every "
         "number sees first.") if not mix or share("overlap") >= 0.5 else
        (f"- Start with a baseline that only gets {first} right. Do not spend every attempt on the outline after "
         f"that: it is {share('overlap'):.0%} of this score. Once the overlap is close, the score is in "
         + ("the marked regions, " if share("regions") else "") + "the colours and the edges, part by part"
         + (", and the close-ups are where to read a region." if share("regions") else ".")),
    ]
    return "\n".join(x for x in lines if x)


def gate_facts(spec: dict) -> list[str]:
    """The limits the project sets on the measures, in words, in the order the ledger has its columns."""
    said = {"outline_mm_max": "outline error at most {} mm", "zones_min": "zones at least {}",
            "resemblance_min": "resemblance at least {}"}
    return [said[name].format(spec[old]) for name, old in criteria.GATES.items() if spec.get(old) is not None]


FALLBACK_VIEWS = "top, side and rear when the project has them, else every compared view"


def criteria_facts(project: Project) -> str:
    """What this reference is judged on, for whoever sets the project up: each line ends with where jig.json states
    it, or says that it states none and what is done instead. `mesh-jig criteria` and `doctor` print it. The project
    must have no problems (`Project.problems`): a criterion that is wrong cannot be read back."""
    said = criteria.sources(project.data)

    def where(key: str) -> str:
        return f" [{said[key]}]" if key in said else ""

    scored, looked = scored_views(project)
    weights = project.scoring_weights
    spec = project.spec
    lines = ["CRITERIA (what this reference is judged on; jig.json `criteria`):"]
    if weights is not None:
        lines.append("- views: " + ", ".join(f"{v} x{w:g}" for v, w in weights.items()) + where("views"))
    else:
        lines.append(f"- views: {', '.join(scored) or 'none'} [NOT STATED: the fallback, {FALLBACK_VIEWS}]")
    if looked:
        lines.append(f"  compared but not scored: {', '.join(looked)}")
    lines.append(f"- zones: measured against the palette's {len(project.palette)} colour(s) [palette]" if project.zones
                 else "- zones: not measured [no palette]")
    prof = project.reference_profile()
    mode = (prof.get("mode") or "craft") if prof else ""
    if mode == "standing":
        names = [s["name"] for s in spec["profile"]["stations"]]
        lines.append(f"- outline: the calibrated standing table, at {len(names)} named heights ({', '.join(names)})"
                     + where("outline"))
    elif mode == "craft":
        lines.append("- outline: the craft table, 11 stations in each of top, side and rear [model.length_m]")
    elif mode:
        unit = "millimetres" if prof.get("unit_mm") else "thousandths of the longest extent (the contract declares no size)"
        lines.append(f"- outline: read through the orthographic cameras ({', '.join(prof['views'])}), in {unit} "
                     "[automatic: no calibrated table stated]")
    else:
        lines.append("- outline: none [no orthographic camera looks along one axis, and no calibrated table stated]")
    limits = gate_facts(spec)
    lines.append("- gates: " + (", ".join(limits) + where("gates") if limits else
                                f"none stated (`criteria.gates` takes {', '.join(criteria.GATES)})"))
    review = project.review
    lines.append("- review: " + (", ".join(f"{len(v)} {k}" for k, v in review.items() if v) + where("review")
                                 if review else "none stated"))
    observations = [r["name"] for r in (review or {}).get("regions", [])
                    if "min_iou" not in r and "min_zone" not in r]
    if observations:
        lines.append("- WARNING: regions without acceptance limits are observations, not gates: " + ", ".join(observations))
    has_review_limits = any(v for k, v in (review or {}).items() if k != "regions") or any(
        "min_iou" in r or "min_zone" in r for r in (review or {}).get("regions", []))
    if not limits and not has_review_limits:
        lines.append("- WARNING: no explicit quality acceptance limits; a measured numeric score does not establish acceptance.")
    if spec.get("light") is not None:
        lines.append(f"- light: {spec['light']}" + where("light"))
    mix = project.score_mix
    lines.append(f"- score: {criteria.score_text(mix)}" + where("score") if mix else
                 "- score: overlap and resemblance in equal parts [NOT STATED: `criteria.score` takes "
                 f"{', '.join(criteria.SCORE_TERMS)}]")
    return "\n".join(lines)


def text(project: Project) -> str:
    """Everything an agent needs to start: the brief, the build rules, the measured outlines, the zone maps, the
    reference views and the commands."""
    spec = project.spec
    refs = project.references
    parts = [f"PROJECT {project.name} ({project.root})"]
    if project.brief.strip():
        parts.append("WHAT THE MODEL MUST SHOW (brief):\n" + project.brief.strip())
    parts.append(model_facts(spec))
    if project.scoring_weights is not None:
        parts.append(f"AGGREGATE SCORING VIEW WEIGHTS: {weights_text(project.scoring_weights)}. Hero only contributes if "
                     "explicitly selected.")
    if project.review is not None:
        import json
        parts.append("REGIONS / PARTS / VISIBILITY / CLEARANCE (fixed review contract; Blender metres):\n" +
                     json.dumps(project.review, indent=2))
    if project.palette:
        parts.append("PALETTE (sRGB hex; convert to linear for vertex colours): "
                     + ", ".join(f"{k} {v}" for k, v in project.palette.items()))
    prof = project.reference_profile()
    if prof:
        parts.append(profile.text(prof))
    elif refs:
        parts.append("No measured outlines: they need an orthographic camera that looks along one axis (a `model.shots` "
                     "entry with `ortho`) and a reference view of its name, or a `model.length_m` and reference views "
                     "named top, side and rear.")
    zones = project.zones
    if zones:
        zone_views = project.scoring_weights if project.scoring_weights is not None else profile.VIEWS
        maps = {v: profile.zone_map(refs[v], zones) for v in zone_views if v in refs}
        maps = {v: rows for v, rows in maps.items() if rows}
        if maps:
            parts.append(profile.zone_text(maps, zones, craft=bool(spec.get("forward"))))
    if refs:
        parts.append("REFERENCE VIEWS (look at them):\n" + "\n".join(f"- {v}: {p}" for v, p in refs.items()))
    reading = reading_facts(project)
    if reading:
        parts.append(reading)
    shots = [s["name"] for s in render.shots_for(spec)]
    unmatched = [v for v in refs if v not in shots]
    if unmatched:
        parts.append(f"Reference views with no camera of the same name (never measured): {unmatched}. The cameras are {shots}.")
    root = project.root.as_posix()
    parts.append(
        "THE LOOP:\n"
        f"- write {root}/attempts/<name>/{glb.script_name(spec)}"
        + (" and atlas_plan.json" if glb.texture_from(spec) == "plan" else "") + ", one idea per attempt, named for it;\n"
        f"- mesh-jig build {root} --script {root}/attempts/<name>/{glb.script_name(spec)}"
        + (f" --paint {root}/attempts/<name>/atlas_plan.json" if glb.texture_from(spec) == "plan" else "")
        + f", then mesh-jig preview {root} {root}/attempts/<name>\n"
        "  is a free look: the overlap, colour zones, resemblance and numeric score an evaluation of that build gives,\n"
        "  with nothing added to the ledger. Edit, build and preview an attempt until it is ahead of your best one;\n"
        f"- mesh-jig eval {root} --script {root}/attempts/<name>/{glb.script_name(spec)}"
        + (f" --paint {root}/attempts/<name>/atlas_plan.json" if glb.texture_from(spec) == "plan" else "")
        + ' --note "<what you changed>"\n'
        "  builds it, renders it, measures it and appends a line to attempts/ledger.md; look at the side-by-sides it names;\n"
        "- the numbers are deterministic and free: run eval on every attempt. Ask the judge only once the numbers have\n"
        "  moved (--judge --against <another attempt>), and treat only a DECISIVE verdict as a result.")
    return "\n\n".join(parts)
