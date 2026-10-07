"""A mesh-jig project: one directory holding the contract, the reference views and every attempt.

    <project>/
      jig.json            the contract (below)
      brief.md            what the model must show, in prose (the judge reads it; optional)
      refs/<view>.png     the reference views, named after the cameras: hero, top, side, rear
      attempts/<name>/    one directory per attempt: the script, its build, its renders, eval.json
      attempts/ledger.md  one line per attempt, appended by `mesh-jig eval`

jig.json:

    {"name": "ship",
     "brief": "brief.md",                       optional, default brief.md when it exists
     "refs": "refs",                            optional, default refs
     "palette": {"crimson": "#b53c2e", ...},    optional: the colours the reference is painted in; turns on the
                                                colour-zone measure and names colours for atlas plans
     "model": {...},                            the build contract, see mesh_jig.glb
     "criteria": {"views": {...}, "outline": {...}, "gates": {...}, "review": {...}},
                                                optional: what this reference is judged on, see mesh_jig.criteria
     "sheet": {"source": "sheet.png",           optional: cut refs/ from one design sheet (`mesh-jig sheet`)
               "mask_sample_x": 8,
               "crops": {"hero": {"box": [x0, y0, x1, y1], "mask": [[x0, y0, x1, y1], ...]}, ...}}}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from . import criteria, glb, outline, palette as palette_mod, profile, silhouette, scoring, standing, render, review

JIG_FILE = "jig.json"
ATTEMPTS = "attempts"
LEDGER = "ledger.md"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


class ProjectError(ValueError):
    pass


@dataclass
class Project:
    root: Path
    data: dict
    _profile: dict | None = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return str(self.data.get("name") or self.root.name)

    @property
    def stated(self) -> dict:
        """jig.json as every tool reads it: the `criteria` block written into the fields it stands for
        (`criteria.fold`). `data` is the file as written, which is what a command that rewrites it starts from."""
        return criteria.fold(self.data)[0]

    @property
    def spec(self) -> dict:
        """The model block with its defaults filled in: the contract every tool reads."""
        spec = dict(self.stated.get("model") or {})
        spec.setdefault("script", "model.py")
        spec.setdefault("glb", f"{self.name}_{{variant}}.glb" if spec.get("variants") else f"{self.name}.glb")
        return spec

    @property
    def refs_dir(self) -> Path:
        return self.root / str(self.data.get("refs") or "refs")

    @property
    def references(self) -> dict[str, str]:
        """view -> image path for every reference view on disk."""
        d = self.refs_dir
        if not d.is_dir():
            return {}
        return {p.stem: str(p) for p in sorted(d.iterdir())
                if p.suffix.lower() in IMAGE_SUFFIXES and not p.stem.startswith(silhouette.OVERLAY_PREFIX)}

    @property
    def brief(self) -> str:
        p = self.root / str(self.data.get("brief") or "brief.md")
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    @property
    def palette(self) -> dict[str, str]:
        return dict(self.data.get("palette") or {})

    @property
    def zones(self) -> tuple[palette_mod.Zone, ...]:
        return palette_mod.zones(self.palette)

    @property
    def length_mm(self) -> float:
        return float(self.spec.get("length_m") or 0) * 1000.0

    @property
    def attempts_dir(self) -> Path:
        return self.root / ATTEMPTS

    @property
    def ledger(self) -> Path:
        return self.attempts_dir / LEDGER

    def reference_profile(self) -> dict:
        """The reference outlines as numbers: the standing table when the contract calibrates one, the craft table
        (profile.measure) when it names a length and has top, side and rear views, else the table read through the
        cameras (outline.measure). {} when no view can be measured: no orthographic camera looks along an axis."""
        if self._profile is None:
            refs = self.references
            self._profile = {}
            if self.spec.get("profile") is not None:
                self._profile = standing.measure(refs, self.spec, self.spec["profile"])
            elif self.length_mm and profile.measurable(refs):
                try:
                    self._profile = profile.measure(refs, self.length_mm)
                except (ValueError, OSError):
                    self._profile = {}
            else:
                try:
                    self._profile = outline.measure(refs, self.spec)
                except (ValueError, KeyError, OSError):
                    self._profile = {}
        return self._profile

    @property
    def scoring_weights(self) -> dict[str, float] | None:
        return scoring.weights(self.stated.get("scoring"))

    @property
    def review(self) -> dict | None:
        """The named regions, parts and probes this reference is held to, when it states any."""
        return self.stated.get("review")

    @property
    def score_mix(self) -> dict[str, float] | None:
        """What the numeric score is a weighted mean of (`criteria.score`: term -> weight), when the project states
        it. None is the score every project had before the key: overlap and resemblance in equal parts."""
        return self.stated.get("score")

    def problems(self) -> list[str]:
        """What is wrong with the project as written; [] when every tool can run on it."""
        out = []
        model = self.data.get("model")
        if not isinstance(model, dict):
            return [f"{JIG_FILE} needs a `model` block (see mesh_jig.glb for its fields)"]
        spec = self.spec
        out += criteria.fold(self.data)[1]
        if len(glb.variant_names(spec)) > 1 and "{variant}" not in spec["glb"]:
            out.append(f"model.glb {spec['glb']!r} must contain {{variant}} when there is more than one variant")
        if spec.get("judge_variant") and spec["judge_variant"] not in glb.variant_names(spec):
            out.append(f"model.judge_variant {spec['judge_variant']!r} is not one of {glb.variant_names(spec)}")
        for key in ("length_m", "span_m", "height_m"):
            if key in spec and not (isinstance(spec[key], (int, float)) and not isinstance(spec[key], bool)
                                    and math.isfinite(spec[key]) and spec[key] > 0):
                out.append(f"model.{key} must be a positive number of metres")
        for key, positive in (("axis_extents_m", True), ("axis_tolerances", False)):
            values = spec.get(key, {})
            if not isinstance(values, dict) or any(a not in ("X", "Y", "Z") or isinstance(v, bool)
                    or not isinstance(v, (int, float)) or not math.isfinite(v) or (v <= 0 if positive else v < 0)
                    for a, v in values.items()):
                out.append(f"model.{key} needs finite {'positive' if positive else 'nonnegative'} values keyed by GLB X/Y/Z")
        if spec.get("height_m") and isinstance(spec.get("axis_extents_m"), dict) and "Y" in spec["axis_extents_m"]:
            if spec["height_m"] != spec["axis_extents_m"]["Y"]:
                out.append("model.height_m conflicts with axis_extents_m.Y")
        if "height_tolerance" in spec and (isinstance(spec["height_tolerance"], bool) or
                not isinstance(spec["height_tolerance"], (int, float)) or
                not math.isfinite(spec["height_tolerance"]) or spec["height_tolerance"] < 0):
            out.append("model.height_tolerance must be finite and nonnegative")
        if spec.get("texture_from") is not None:
            if spec["texture_from"] not in glb.TEXTURE_SOURCES:
                out.append(f"model.texture_from must be one of {', '.join(glb.TEXTURE_SOURCES)}")
            elif not spec.get("textured"):
                out.append("model.texture_from needs `textured`: it says where a textured contract's atlas comes from")
        if "texture_size" in spec:
            lo, hi = glb.TEXTURE_SIZES
            size = spec["texture_size"]
            if isinstance(size, bool) or not isinstance(size, int) or not lo <= size <= hi:
                out.append(f"model.texture_size must be a whole number of pixels from {lo} to {hi}")
        if glb.texture_from(spec) == "script" and glb.texture_mode(spec) == "multiply":
            out.append('model.texture_mode "multiply" needs vertex colours, and a script that paints its texture '
                       "authors none: leave texture_mode out")
        if spec.get("light") is not None and spec["light"] not in render.LIGHT_MODES:
            out.append(criteria.named(f"model.light must be one of {', '.join(render.LIGHT_MODES)}", "model.light",
                                      criteria.sources(self.data).get("light", "")))
        for shot in spec.get("shots") or []:
            if not all(k in shot for k in ("name", "cam_pos", "cam_target")):
                out.append(f"model.shots entry {shot!r} needs name, cam_pos and cam_target")
        # a message names the setting where this jig.json states it: `criteria.outline`, or the older `profile`
        said = criteria.sources(self.data)

        def checked(check, older: str, key: str) -> None:
            try:
                check()
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                out.append(criteria.named(str(e), older, said.get(key, "")))

        def views() -> None:
            weights = self.scoring_weights
            if weights is not None:
                names = {s["name"] for s in render.shots_for(spec)}
                missing = sorted(set(weights) - (set(self.references) & names))
                if missing:
                    raise ValueError(f"scoring: selected views lack a reference/camera: {missing}")

        checked(views, "scoring", "views")
        if spec.get("profile") is not None:
            checked(lambda: standing.validate(spec["profile"], spec, self.references), "profile", "outline")
        if self.review is not None:
            checked(lambda: review.validate(self.review, spec, self.references, self.zones), "review", "review")
        mix = self.score_mix or {}
        if "regions" in mix and not (isinstance(self.review, dict) and self.review.get("regions")):
            out.append("criteria.score counts `regions` and the project marks none: give criteria.review.regions, "
                       "or take `regions` out of the score")
        if "zones" in mix and not self.palette:
            out.append("criteria.score counts `zones` and the project has no palette: zones are read against one")
        try:
            palette_mod.zones(self.palette)
        except ValueError as e:
            out.append(f"palette: {e}")
        if not self.references:
            out.append(f"no reference views under {self.refs_dir} (name them after the cameras: hero, top, side, rear)")
        return out


def load(path: str | Path) -> Project:
    """The project at `path` (its directory or its jig.json)."""
    p = Path(path)
    root = p.parent if p.is_file() else p
    jig = root / JIG_FILE
    if not jig.is_file():
        raise ProjectError(f"no {JIG_FILE} in {root}: `mesh-jig init {root}` writes one")
    try:
        data = json.loads(jig.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ProjectError(f"{jig}: not JSON ({e})") from None
    if not isinstance(data, dict):
        raise ProjectError(f"{jig}: the top level must be an object")
    return Project(root=root.resolve(), data=data)
