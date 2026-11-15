"""The Godot side of a game-ready package: the files `game-build` writes for Godot, and the headless harness that
plays a scenario with the Godot addon (`game-run --runtime godot`).

`package/godot/` holds `character.tscn` (character.glb instanced under a MeshJigCharacter node, with
character.runtime.json and character.clips.json set on it, all by paths relative to the scene, so the package folder
can sit anywhere in a project), `character.clips.json` and `addons/mesh_jig_character/`, a copy of the fixed addon
(the same bytes for every character; it goes in the project's `res://addons/`).

character.clips.json is every joint track of the GLB's clips, key for key. The addon samples it as the reference does
(linear, slerp) instead of the Animation resources Godot's scene import makes: that import resamples every track at its
bake rate (30 Hz by default, which moved a 60 Hz rig's feet 2.3 mm) and dropped keys on the armour whatever the rate
(the walk's swing extreme: 2.8 degrees at 30 Hz, 1.1 at 120), with its optimizer off. See docs/GAME_READY.md, Godot.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import glb

GODOT_ENV = 'MESH_JIG_GODOT'
ENGINE = Path(__file__).resolve().parents[2]/'engines'/'godot'
ADDON = ENGINE/'addons'/'mesh_jig_character'
HARNESS = ENGINE/'harness'/'play_scenario.gd'
RUNTIME_NAME = 'godot-mesh-jig-character'
ADDON_PATH = 'res://addons/mesh_jig_character'
RATE_MAX = 240                       # the highest key rate looked for; past it the clips are resampled at 120 Hz
RUN_TIMEOUT_S = 600


def find_godot(env=os.environ) -> str | None:
    """The Godot to run: $MESH_JIG_GODOT, else `godot` or `godot4` on PATH. On Windows name the console build."""
    named = env.get(GODOT_ENV)
    if named:
        return named
    return shutil.which('godot') or shutil.which('godot4')


def addon_files() -> list[Path]:
    return sorted(p for p in ADDON.iterdir() if p.is_file() and p.suffix in ('.gd', '.cfg'))


def key_rate(path) -> int:
    """The clips' key rate: the smallest whole number of keys a second on which every key time of every animation
    falls (120 when none up to RATE_MAX does). The harness bakes the scene's own Animation resources at it; the addon
    plays character.clips.json, so this only matters to someone previewing clips in the editor."""
    doc, blob = glb.read_glb(path)
    times = np.unique(np.concatenate([glb.accessor(doc, blob, s['input']).ravel() for a in doc.get('animations', [])
                                      for s in a['samplers']] or [np.zeros(1)]))
    for rate in range(1, RATE_MAX+1):
        if np.abs(times*rate-np.round(times*rate)).max() < 1e-3:
            return max(rate, 1)
    return 120


def _short(x) -> float:
    """A float32 as its shortest decimal: the same float32 once read back."""
    return float(np.format_float_positional(np.float32(x), unique=True, trim='-'))


def clips_data(path) -> dict:
    """Every joint track of every named clip in the GLB, key for key: {clip: {joint: {translation|rotation|scale:
    {interpolation, times, values (flat, 3 or 4 a key; rotations x, y, z, w)}}}}."""
    from . import game_measure, terrain
    body = game_measure.Body(path)
    names = {i: n for n, i in body.node.items()}
    clips = {}
    for name, clip in sorted(body.clips.items()):
        joints = {}
        for node, kind, times, values, interpolation in clip.channels:
            if node in names:
                joints.setdefault(names[node], {})[kind] = {
                    'interpolation': interpolation, 'times': [_short(x) for x in np.asarray(times).ravel()],
                    'values': [_short(x) for x in np.asarray(values).ravel()]}
        clips[name] = joints
    return {'format': 'mesh-jig game clips', 'version': 1, 'glb_sha256': terrain.digest(path), 'clips': clips}


def rest_ground(path) -> float:
    """The rest pose's lowest vertex, model-space height (glTF y): what the character stands on."""
    from . import game_measure
    return float(game_measure.Body(path).rest[:, 2].min())


def scene_text(lift: float) -> str:
    """character.tscn: the GLB under a MeshJigCharacter, its runtime file set, the model raised by `lift` (minus the
    rest pose's lowest vertex height) so the node's origin is where the character stands. No uids: Godot fills them
    in when it saves the scene."""
    lift = round(float(lift), 6)+0.0
    return '\n'.join([
        '[gd_scene load_steps=5 format=3]',
        '',
        f'[ext_resource type="Script" path="{ADDON_PATH}/mesh_jig_character.gd" id="1_script"]',
        '[ext_resource type="PackedScene" path="../character.glb" id="2_model"]',
        '[ext_resource type="JSON" path="../character.runtime.json" id="3_runtime"]',
        '[ext_resource type="JSON" path="character.clips.json" id="4_clips"]',
        '',
        '[node name="Character" type="Node3D"]',
        'script = ExtResource("1_script")',
        'runtime_file = ExtResource("3_runtime")',
        'clips_file = ExtResource("4_clips")',
        '',
        '[node name="Model" parent="." instance=ExtResource("2_model")]',
        f'transform = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, {lift:g}, 0)',
        ''])


def write(package) -> list[str]:
    """Write the Godot files into a built package. Deterministic: the same package gives the same bytes."""
    package = Path(package).resolve()
    # Nested agent runs can exceed Win32's legacy 260-character path limit.
    if os.name == 'nt' and not str(package).startswith('\\\\?\\'):
        package = Path('\\\\?\\'+str(package))
    out = package/'godot'
    target = out/'addons'/'mesh_jig_character'
    target.mkdir(parents=True, exist_ok=True)
    written = []
    for f in addon_files():
        (target/f.name).write_bytes(f.read_bytes())
        written.append(f'godot/addons/mesh_jig_character/{f.name}')
    (out/'character.tscn').write_bytes(scene_text(-rest_ground(package/'character.glb')).encode('utf-8'))
    clips = json.dumps(clips_data(package/'character.glb'), sort_keys=True, separators=(',', ':'), allow_nan=False)
    (out/'character.clips.json').write_bytes((clips+'\n').encode('utf-8'))
    return written+['godot/character.tscn', 'godot/character.clips.json']


def exact_numbers(data):
    """The scenario with every number as the hex of its IEEE-754 double, little-endian (what the harness decodes):
    Godot's JSON parser is not correctly rounded, so the harness takes the numbers Python read."""
    if isinstance(data, dict):
        return {k: exact_numbers(v) for k, v in data.items()}
    if isinstance(data, list):
        return [exact_numbers(v) for v in data]
    if isinstance(data, (int, float)) and not isinstance(data, bool):
        return struct.pack('<d', float(data)).hex()
    return data


def run(package, scenario, out, godot=None, timeout=RUN_TIMEOUT_S, harness=HARNESS):
    """Play a scenario file with the Godot addon, headless, and write its trace to `out` (which must be new).
    `harness`: another copy of play_scenario.gd beside its addon (tests run a broken copy)."""
    package, scenario, out = Path(package).resolve(), Path(scenario).resolve(), Path(out).resolve()
    exe = godot or find_godot()
    if not exe:
        raise ValueError(f'Godot was not found: set {GODOT_ENV} to the Godot executable (on Windows the _console.exe)')
    if out.exists():
        raise ValueError('the trace file must be new')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        numbers = Path(tmp)/'exact.json'
        numbers.write_text(json.dumps(exact_numbers(json.loads(scenario.read_text(encoding='utf-8')))), encoding='utf-8')
        cmd = [exe, '--headless', '--xr-mode', 'off', '--script', Path(harness).as_posix(), '--', package.as_posix(),
               scenario.as_posix(), out.as_posix(), str(key_rate(package/'character.glb')), numbers.as_posix()]
        done = subprocess.run(cmd, cwd=Path(harness).parent, capture_output=True, text=True, timeout=timeout,
                              encoding='utf-8', errors='replace')
    errors = [line for line in (done.stdout+done.stderr).splitlines() if 'ERROR' in line and 'ObjectDB' not in line]
    if done.returncode != 0 or not out.is_file() or errors:
        raise RuntimeError(f'the Godot harness failed (exit {done.returncode}):\n'+(done.stdout+done.stderr)[-4000:])
    return out
