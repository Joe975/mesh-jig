"""Write-once selection checkpoints and comparison sheets from existing artifacts.

No build, render, scoring or model call happens here. Visual choices are supplied by a
person or reviewer; the numerical winner is never substituted for a missing choice.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import tempfile

from PIL import Image, ImageDraw, ImageFont, ImageOps

from . import evaluate, glb, harness
from .project import Project

SCHEMA = 1
DIRECTORY = "checkpoints"
ROLES = ("latest", "numeric_best", "visual")
ROLE_LABELS = {"latest": "Latest", "numeric_best": "Numeric best", "visual": "Visual choice"}
SOURCES = ("human", "self-review", "reviewer", "pairwise")
NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}\Z")


def _encode(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _hash(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _name(value: str) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value) or value.endswith("."):
        raise ValueError("use a directory name such as round-05, with letters, numbers, dots, hyphens or underscores")
    # Windows device names are invalid even with a filename extension.
    if value.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                                       *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("reserved directory name")
    return value


def _inside(path: Path, root: Path) -> Path:
    path, root = path.resolve(), root.resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("artifact path escapes its directory")
    return path


def _relative(path: str) -> Path:
    if not isinstance(path, str):
        raise ValueError("invalid checkpoint artifact path")
    p = Path(path)
    if not path or p.is_absolute() or p.drive or ".." in p.parts or "\\" in path:
        raise ValueError("invalid checkpoint artifact path")
    return p


def _image(raw: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError("invalid saved image") from error


def _records(project: Project, through: str) -> list[dict]:
    rows = []
    for file in sorted(project.attempts_dir.glob("*/eval.json")):
        _inside(file, project.attempts_dir)
        try:
            raw = file.read_bytes()
            rec = json.loads(raw)
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read evaluation for {file.parent.name}") from error
        if not isinstance(rec, dict):
            raise ValueError(f"invalid evaluation for {file.parent.name}")
        if not rec.get("ok") or rec.get("stage") != "measured":
            continue
        name = _name(file.parent.name)
        numeric = rec.get("numeric") or {}
        if not isinstance(numeric, dict):
            raise ValueError(f"invalid numeric record for {name}")
        score = numeric.get("numeric_score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))
                                  or not math.isfinite(score)):
            raise ValueError(f"invalid numeric score for {name}")
        started = rec.get("started")
        if started is not None and not isinstance(started, str):
            raise ValueError(f"invalid start timestamp for {name}")
        fallback = datetime.fromtimestamp(file.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        stored_script = Path(rec.get("script") or project.spec["script"])
        script = _inside(stored_script if stored_script.is_absolute() else file.parent / stored_script, file.parent)
        if not script.is_file():
            raise ValueError(f"missing source script for {name}")
        rows.append({"name": name, "score": score, "started": started, "order": [started or fallback, name],
                     "script_sha256": _hash(script.read_bytes()),
                     "evaluation_sha256": _hash(raw), "record": rec, "raw": raw, "directory": file.parent})
    rows.sort(key=lambda row: row["order"])
    if through:
        stop = next((i for i, row in enumerate(rows) if row["name"] == _name(through)), None)
        if stop is None:
            raise ValueError("through must name a successful measured attempt")
        rows = rows[:stop + 1]
    if not rows:
        raise ValueError("no successful measured attempts; wait for evaluations to finish")
    return rows


def verify(directory: str | Path) -> dict:
    """Verify only the frozen copies, so later source edits do not alter an old checkpoint."""
    root = Path(directory).resolve()
    try:
        seal = json.loads(_inside(root / "seal.json", root).read_bytes())
        if not isinstance(seal, dict) or seal.get("schema_version") != SCHEMA or not isinstance(seal.get("files"), dict):
            raise ValueError("unsupported checkpoint seal")
        if "checkpoint.json" not in seal["files"] or "comparison.png" not in seal["files"]:
            raise ValueError("incomplete checkpoint seal")
        for name, expected in seal["files"].items():
            file = _inside(root / _relative(name), root)
            if _hash(file.read_bytes()) != expected:
                raise ValueError(f"checkpoint artifact changed: {name}")
        actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} - {"seal.json"}
        if actual != set(seal["files"]):
            raise ValueError("checkpoint file inventory changed")
        record = json.loads(_inside(root / "checkpoint.json", root).read_bytes())
        if not isinstance(record, dict) or record.get("schema_version") != SCHEMA:
            raise ValueError("unsupported checkpoint format")

        def artifacts(value):
            if isinstance(value, dict):
                if "path" in value and "sha256" in value:
                    path = _relative(value["path"]).as_posix()
                    if seal["files"].get(path) != value["sha256"]:
                        raise ValueError("artifact identity disagrees with checkpoint seal")
                for child in value.values():
                    artifacts(child)
            elif isinstance(value, list):
                for child in value:
                    artifacts(child)

        artifacts(record)
        return record
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ValueError("cannot verify checkpoint: missing or invalid files") from error


def _sheet(record: dict, blobs: dict[str, bytes], before: dict | None, before_root: Path | None) -> bytes:
    """Use saved pixels at their original aspect ratio; never manufacture missing views."""
    cell, gap, margin = 500, 24, 32
    columns = 3 if before else 4
    width = margin * 2 + columns * cell + (columns - 1) * gap
    panels = [(view, role) for view in record["views"] for role in ROLES] if before else [
        (view, None) for view in record["views"]]
    row_height = 390
    image = Image.new("RGB", (width, 170 + len(panels) * row_height + 60), "#f6f2eb")
    draw = ImageDraw.Draw(image)

    def text(x, y, value, size=25):
        # Wrap by measured width, including long attempt labels and review sources.
        font = ImageFont.load_default(size=size)
        lines, line = [], ""
        for character in str(value):
            candidate = line + character
            if line and draw.textlength(candidate, font=font) > cell - 12:
                lines.append(line)
                line = character
            else:
                line = candidate
        lines.append(line)
        if len(lines) > 2:
            lines[1] = lines[1][:-3] + "..."
        for index, line in enumerate(lines[:2]):
            draw.text((x, y + index * (size + 5)), line, fill="#27343c", font=font)

    def picture(entry, x, y, data=None):
        draw.rectangle((x, y, x + cell, y + 280), fill="#e2dfd8")
        if entry:
            raw = data if data is not None else blobs[entry["path"]]
            pic = ImageOps.contain(_image(raw), (cell, 280), Image.Resampling.LANCZOS)
            image.paste(pic, (x + (cell - pic.width) // 2, y + (280 - pic.height) // 2))
        else:
            text(x + 14, y + 120, "Image unavailable", 26)

    title = "Selection checkpoint: " + record["name"]
    title_size = 38
    while title_size > 18 and draw.textlength(title, font=ImageFont.load_default(size=title_size)) > width - margin * 2:
        title_size -= 1
    draw.text((margin, 25), title, fill="#27343c", font=ImageFont.load_default(size=title_size))
    scope = (f"Before {before['name']}: {before['scope']['successful_count']} successes / " if before else "")
    scope += f"After {record['name']}: {record['scope']['successful_count']} successes"
    draw.text((margin, 82), scope,
              fill="#27343c", font=ImageFont.load_default(size=25))
    draw.text((margin, 119), "Visual choice records a reviewer's preference. Numeric score measures reference fit.",
              fill="#27343c", font=ImageFont.load_default(size=24))
    for index, (view, role) in enumerate(panels):
        y = 160 + index * row_height
        reference = record["references"].get(view)
        text(margin, y, f"Reference / {view}" + (f" / {ROLE_LABELS[role]}" if role else ""))
        picture(reference, margin, y + 46)
        choices = [("Before", before, before_root, role), ("After", record, None, role)] if before else [
            (label, record, None, selected) for label, selected in
            (("Latest", "latest"), ("Numeric best", "numeric_best"), ("Visual choice", "visual"))]
        for col, (label, checkpoint, source_root, selected_role) in enumerate(choices, 1):
            x = margin + col * (cell + gap)
            name = checkpoint["selections"][selected_role]
            candidate = checkpoint["attempts"].get(name) if name else None
            entry = candidate["artifacts"]["renders"].get(view) if candidate else None
            text(x, y, label + (f" / {ROLE_LABELS[selected_role]}" if before else ""))
            data = (source_root / entry["path"]).read_bytes() if entry and source_root else None
            picture(entry, x, y + 46, data)
            score = candidate["numeric_score"] if candidate else None
            absent = "No explicit visual choice" if selected_role == "visual" else "No numeric score available"
            text(x, y + 334, f"{name or absent} / score {score if score is not None else 'unavailable'}", 23)
    footer = "Saved renders only. Numeric fit and visual preference are separate. Private working artifacts."
    if any("identit" in warning and "evaluat" in warning.lower() for warning in record["warnings"]):
        footer = "Evaluation versions differ or are unknown. Compare images; the scores may not be comparable."
    draw.text((margin, image.height - 40), footer,
              fill="#27343c", font=ImageFont.load_default(size=22))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def create(project: Project, name: str, *, visual_attempt: str = "", visual_source: str = "", reason: str = "",
           review_file: str = "", before: str = "", through: str = "", views: list[str] | None = None) -> dict:
    """Snapshot latest/numeric/explicit visual selections. Returns the verified portable record.

    Numeric ties keep the earliest measured attempt, matching evaluate.standing. Selection
    choices and review-source labels are assertions, not independent review attestation.
    """
    name = _name(name)
    root = project.root.resolve()
    base = _inside(root / DIRECTORY, root)
    destination = _inside(base / name, base)
    if destination.exists():
        raise ValueError("checkpoint already exists; choose a new name")
    if visual_attempt:
        _name(visual_attempt)
        if visual_source not in SOURCES or not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
            raise ValueError("visual choice requires a source (human/self-review/reviewer/pairwise) and a short reason")
    elif visual_source or reason or review_file:
        raise ValueError("review source, reason and evidence require an explicit visual attempt")
    rows = _records(project, through)
    by_name = {row["name"]: row for row in rows}
    if visual_attempt and visual_attempt not in by_name:
        raise ValueError("visual choice must be a successful measured attempt within this checkpoint's scope")
    scored = [row for row in rows if row["score"] is not None]
    best = max(scored, key=lambda row: row["score"]) if scored else None
    selections = {"latest": rows[-1]["name"], "numeric_best": best["name"] if best else None,
                  "visual": visual_attempt or None}
    blobs: dict[str, bytes] = {}
    warnings = []

    def capture(path: Path, target: str, *, image=False) -> dict:
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ValueError(f"missing or unreadable artifact: {path.name}") from error
        if image:
            _image(raw)
        blobs[target] = raw
        return {"path": target, "sha256": _hash(raw), "size_bytes": len(raw)}

    inputs = {"jig": capture(root / "jig.json", "inputs/jig.json")}
    brief = root / str(project.data.get("brief") or "brief.md")
    if brief.is_file():
        inputs["brief"] = capture(brief, "inputs/brief.md")
    references = {}
    for view, path in project.references.items():
        _name(view)
        references[view] = capture(Path(path), f"references/{view}{Path(path).suffix.lower()}", image=True)
    input_hash = _hash(_encode({"inputs": {k: v["sha256"] for k, v in inputs.items()},
                                "references": {k: v["sha256"] for k, v in references.items()}}))
    before_record, before_root = None, None
    if before:
        before_root = _inside(base / _name(before), base)
        before_record = verify(before_root)
        if before_record["inputs_sha256"] != input_hash:
            raise ValueError("before checkpoint has different inputs/references; do not present it as the same task")
    attempts = {}
    for selected in dict.fromkeys(n for n in selections.values() if n):
        row = by_name[selected]
        directory = row["directory"].resolve()
        rec = evaluate.read(directory)
        script = _inside(Path(rec.get("script") or directory / project.spec["script"]), directory)
        prefix = f"attempts/{selected}"
        artifacts = {"evaluation": capture(directory / "eval.json", prefix + "/eval.json"),
                     "script": capture(script, prefix + "/script" + script.suffix), "models": {}, "renders": {}}
        if artifacts["evaluation"]["sha256"] != row["evaluation_sha256"]:
            raise ValueError("evaluation changed while creating checkpoint; retry after it finishes")
        if artifacts["script"]["sha256"] != row["script_sha256"]:
            raise ValueError("script changed while creating checkpoint; retry after it finishes")
        for variant, filename in glb.expected_files(project.spec).items():
            model = _inside(directory / "out" / filename, directory)
            try:
                glb.stats(model)  # Reject a missing or unreadable model; do not remeasure reference fit.
            except (OSError, ValueError, KeyError, IndexError) as error:
                raise ValueError(f"missing or invalid model for {selected}: {model.name}") from error
            artifacts["models"][variant] = capture(model, prefix + "/models/" + model.name)
        renders = rec.get("renders") or evaluate.renders_of(directory)
        for view, path in renders.items():
            _name(view)
            file = _inside(Path(path), directory)
            if file.is_file():
                artifacts["renders"][view] = capture(file, prefix + f"/renders/{view}{file.suffix.lower()}", image=True)
            else:
                warnings.append(f"{selected}: missing saved render {view}")
        attempts[selected] = {"numeric_score": row["score"], "evaluation_harness": row["record"].get("harness"),
                              "artifacts": artifacts}
    versions = [row["record"].get("harness") for row in rows]
    if before_record:
        versions += [item.get("evaluation_harness") for item in before_record["attempts"].values()]
    if any(version is None for version in versions):
        warnings.append("Some evaluations have no recorded harness identity; score comparability is unknown")
    if len({_encode(version) for version in versions if version is not None}) > 1:
        warnings.append("Evaluation harness identities differ; scores may not be comparable")
    available = set(references) | {v for a in attempts.values() for v in a["artifacts"]["renders"]}
    if views is not None and (not views or len(set(views)) != len(views)):
        raise ValueError("views must be a nonempty list without duplicates")
    chosen_views = views if views is not None else [v for v in ("hero", "front", "side") if v in available]
    chosen_views = chosen_views or sorted(available)[:3] or ["hero"]
    for view in chosen_views:
        _name(view)
        if view not in references:
            warnings.append(f"no reference image for {view}")
        for selected, item in attempts.items():
            if view not in item["artifacts"]["renders"]:
                warnings.append(f"{selected}: no image for {view}; sheet uses a labelled placeholder")
    evidence = None
    if review_file:
        evidence = capture(_inside(root / review_file, root), "review/evidence" + Path(review_file).suffix)
    record = {"schema_version": SCHEMA, "name": name,
              "created_utc": datetime.now(timezone.utc).isoformat(), "selection_harness": harness.version(),
              "scope": {"successful_count": len(rows), "through": through or None,
                        "order_rule": "evaluation started (or eval file mtime if absent), then attempt directory name",
                        "distinct_script_count": len({row["script_sha256"] for row in rows}),
                        "candidates": [{k: row[k] for k in ("name", "score", "started", "order", "evaluation_sha256", "script_sha256")} for row in rows]},
              "selections": selections, "numeric_best_ties": [row["name"] for row in scored if row["score"] == best["score"]] if best else [],
              "numeric_tie_rule": "earliest in scope order", "visual_choice": {
                  "source": visual_source, "reason": reason.strip(), "evidence": evidence} if visual_attempt else None,
              "inputs": inputs, "inputs_sha256": input_hash, "references": references,
              "attempts": attempts, "views": chosen_views, "warnings": list(dict.fromkeys(warnings)),
              "before": {"name": before, "checkpoint_sha256": _hash((before_root / "checkpoint.json").read_bytes())} if before else None}
    blobs["comparison.png"] = _sheet(record, blobs, before_record, before_root)
    record["comparison"] = {"path": "comparison.png", "sha256": _hash(blobs["comparison.png"])}
    blobs["checkpoint.json"] = _encode(record)
    blobs["seal.json"] = _encode({"schema_version": SCHEMA, "files": {p: _hash(raw) for p, raw in blobs.items()}})
    # Validate everything before reserving an output. Staging stays inside the project.
    base.mkdir(exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=base)).resolve()
    try:
        for path, raw in blobs.items():
            target = staging / _relative(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        verify(staging)
        destination.mkdir()  # Exclusive reservation; an existing checkpoint is never overwritten.
        for path in staging.iterdir():
            shutil.move(str(path), str(destination / path.name))
    finally:
        _inside(staging, base)
        shutil.rmtree(staging)
    return verify(destination)
