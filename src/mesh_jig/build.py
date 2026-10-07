"""Build a model script into its GLBs with Blender, then hold them to the contract.

Blender runs in the background on one file, blender_runner.py, which refuses a script that reaches outside bpy /
bmesh / mathutils / math / random / numpy / openvdb before it executes it. Nothing here renders.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import glb, script_check

RUNNER = Path(__file__).resolve().parent / "blender_runner.py"
BLENDER_ENV = "MESH_JIG_BLENDER"
OK_MARK = "JIG_BUILD_OK"
BUILD_TIMEOUT_S = 300
PALETTE_FILE = "palette.json"       # the project's palette as the runner hands it to a script that paints its texture


def _candidates() -> list[str]:
    if sys.platform == "win32":
        roots = [os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", "")]
        found = [p for r in roots if r for p in glob.glob(os.path.join(r, "Blender Foundation", "Blender*", "blender.exe"))]
        return sorted(found, reverse=True)          # the newest version folder first
    if sys.platform == "darwin":
        return ["/Applications/Blender.app/Contents/MacOS/Blender",
                os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender")]
    return ["/usr/bin/blender", "/usr/local/bin/blender", "/snap/bin/blender", "/opt/blender/blender"]


def find_blender(env=os.environ) -> str | None:
    """The Blender to run: $MESH_JIG_BLENDER, else `blender` on PATH, else the usual install places."""
    named = env.get(BLENDER_ENV)
    if named:
        return named
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    return next((p for p in _candidates() if os.path.isfile(p)), None)


def build_variants(script: str | Path, out_dir: str | Path, spec: dict, *, blender: str | None = None,
                   runner=subprocess.run, atlas: str | Path | None = None, palette: dict | None = None) -> str:
    """Build every variant's GLB from `script` into out_dir (in parallel) and run the contract gate, which writes
    out_dir/glb_stats.json. "" when clean, else the error text for the agent. The script is copied into out_dir
    under the contract's script name; `atlas` (the textured contract) is copied beside it under the contract's
    atlas name and bound by the runner. On a `texture_from: "script"` contract no atlas is given: the runner paints
    one from the script's `paint` (which is handed `palette` as PALETTE) and the judged variant's is left under the
    atlas name. With `blend` in the contract each variant's Blender file is saved beside its GLB."""
    script, out_dir = Path(script), Path(out_dir)
    if not script.is_file():
        return f"model script {script} does not exist"
    refused = script_check.check_script(script.read_text(encoding="utf-8"))
    if refused:
        return "model script refused: " + "; ".join(refused[:6])
    blender = blender or find_blender()
    if not blender:
        return (f"Blender not found: install it, put `blender` on PATH, or set {BLENDER_ENV} to the executable "
                "(`mesh-jig doctor` shows what was looked for)")
    out_dir.mkdir(parents=True, exist_ok=True)
    held = out_dir / glb.script_name(spec)
    if script.resolve() != held.resolve():
        shutil.copyfile(script, held)
    atlas_arg: list[str] = []
    if atlas is not None:
        bound = out_dir / glb.atlas_name(spec)
        if Path(atlas).resolve() != bound.resolve():
            shutil.copyfile(atlas, bound)
        atlas_arg = ["--atlas", str(bound)]
    files = glb.expected_files(spec)
    if not files:
        return "model block names no glb pattern"
    painted = glb.texture_from(spec) == "script"
    if painted and atlas is not None:
        return 'the contract\'s texture is painted by the script (`texture_from: "script"`): give no atlas or plan'
    if painted:
        (out_dir / PALETTE_FILE).write_text(json.dumps(palette or {}), encoding="utf-8")

    def one(item):
        variant, name = item
        argv = [blender, "--background", "--factory-startup", "--python", str(RUNNER), "--", "--script", str(held),
                "--out", str(out_dir / name), "--variant", variant] + atlas_arg
        if spec.get("join"):
            argv += ["--join", "1"]
        if painted:
            argv += ["--texture-out", str(out_dir / glb.painted_name(name)), "--texture-size",
                     str(glb.texture_size(spec)), "--palette", str(out_dir / PALETTE_FILE)]
        if spec.get("blend"):
            argv += ["--blend", str(out_dir / glb.blend_name(name))]
        try:
            r = runner(argv, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return f"variant {variant or 'default'}: Blender timed out after {BUILD_TIMEOUT_S} s"
        except OSError as e:
            return f"variant {variant or 'default'}: could not run Blender at {blender} ({e})"
        log = (r.stdout or "") + (r.stderr or "")
        return "" if OK_MARK in log else f"variant {variant or 'default'}: " + log[-2500:]

    with ThreadPoolExecutor(max_workers=len(files)) as pool:
        errs = [e for e in pool.map(one, files.items()) if e]
    if errs:
        return "\n".join(errs)
    if painted:                                 # the atlas the contract names is the judged variant's
        shutil.copyfile(out_dir / glb.painted_name(glb.judged_glb(spec)), out_dir / glb.atlas_name(spec))
    report = glb.write_report(out_dir, spec)
    return "contract: " + "; ".join(report["issues"]) if report["issues"] else ""
