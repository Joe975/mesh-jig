"""Create a fresh synthetic smoke project. No model backend, downloads or private inputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from mesh_jig import build, render


def prepare(root: Path) -> None:
    # Refuse even a partially prepared project; never overwrite an old attempt or reference.
    if root.exists():
        raise ValueError(f"{root} already exists; choose a new project directory")
    blender = build.find_blender()
    if not blender or not Path(blender).is_file() and shutil.which(blender) is None:
        raise ValueError("Blender not found; put it on PATH or set MESH_JIG_BLENDER in this shell")
    script = Path(__file__).with_name("model.py")
    spec = {
        "script": "model.py", "glb": "box.glb", "height_m": 1.0, "height_tolerance": 0.01,
        "single_mesh": True, "vertex_colors": True, "min_triangles": 12, "max_triangles": 12,
        "shots": [
            {"name": "front", "cam_pos": [0, 0.5, -3], "cam_target": [0, 0.5, 0], "ortho": 1.5},
            {"name": "side", "cam_pos": [-3, 0.5, 0], "cam_target": [0, 0.5, 0], "ortho": 1.5},
            {"name": "top", "cam_pos": [0, 3, 0], "cam_target": [0, 0.5, 0],
             "cam_up": [0, 0, 1], "ortho": 1.5},
        ],
    }
    root.mkdir(parents=True)
    config = {"name": "synthetic-box", "model": spec,
              "criteria": {"views": {"front": 1, "side": 1, "top": 1}}}
    (root / "jig.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (root / "brief.md").write_text(
        "A supplied blue box with a gold top, 1 m tall, 0.6 m wide and 0.4 m deep.\n"
        "References come from this same script: a pipeline self-check, not an AI modeling result.\n",
        encoding="utf-8",
    )
    truth = root / "reference-build"
    error = build.build_variants(script, truth, spec, blender=blender)
    if error:
        raise ValueError(error)
    result = render.render_views(spec, truth, root / "refs")
    if result["error"]:
        raise ValueError(result["error"])
    attempt = root / "attempts" / "a00"
    attempt.mkdir(parents=True)
    shutil.copyfile(script, attempt / "model.py")
    print(f"Prepared {root}: three reference views and supplied attempts/a00/model.py")
    print("No evaluation recorded yet. Run doctor, build, preview, then eval as in docs/QUICKSTART.md.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="new directory, normally local-work/smoke")
    args = parser.parse_args()
    try:
        prepare(args.project)
    except (OSError, ValueError) as exc:
        print(f"smoke preparation: {exc}")
        print("If preparation was interrupted, preserve the partial project and choose a new directory.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
