"""`game-preview`: the skinned character on the scenario's heightfield, drawn from a trace with the CPU renderer.

A film strip (a three-quarter row and a side row) and a GIF, one camera for the whole run so the root's travel and
the ground both show. It draws what the trace says; it measures nothing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import game_check, game_measure, game_scenario, render, rig_look


def ground_mesh(ground, heights):
    """The heightfield as triangles in glTF axes, split along the same diagonal the measurement uses."""
    tris, colours = [], []
    for j in range(len(ground.z)-1):
        for i in range(len(ground.x)-1):
            p = [np.array([ground.x[a], heights[b, a], ground.z[b]]) for a, b in ((i, j), (i+1, j), (i+1, j+1), (i, j+1))]
            colour = (.26, .32, .18) if (i+j) % 2 else (.36, .39, .23)
            # wound so the renderer, which keeps triangles facing the camera, sees the top
            tris += [[p[0], p[2], p[1]], [p[0], p[3], p[2]]]
            colours += [colour, colour]
    tris = np.array(tris)
    normals = np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return render.Mesh(tris, np.repeat(np.array(colours)[:, None], 3, axis=1), np.repeat(normals[:, None], 3, axis=1))


def character_mesh(skin, base, root, matrices):
    v = skin.vertices(skin.world(root, matrices))
    tris = v[skin.body.faces]
    normals = np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
    return render.Mesh(tris, base.colors, np.repeat(normals[:, None], 3, axis=1), base.uvs, base.tex_index, base.textures)


def combine(a, b):
    n = len(b.tris)
    return render.Mesh(np.concatenate((a.tris, b.tris)), np.concatenate((a.colors, b.colors)),
                       np.concatenate((a.normals, b.normals)),
                       np.concatenate((a.uvs, np.zeros((n, 3, 2)))) if a.uvs is not None else None,
                       np.concatenate((a.tex_index, np.full(n, -1))) if a.tex_index is not None else None, a.textures)


def preview(package, scenario, trace, out, *, phases=8, size=360):
    if not 2 <= phases <= 40 or not 64 <= size <= 1024:
        raise ValueError('preview phases must be 2..40 and size 64..1024')
    package, out = Path(package), Path(out)
    runtime = json.loads((package/'character.runtime.json').read_text(encoding='utf-8'))
    game_scenario.resolve(scenario, runtime)
    body = game_measure.Body(package/runtime['glb'])
    names = [j['name'] for j in runtime['skeleton']['joints']]
    times = game_scenario.frame_times(scenario)
    roots, mats = game_scenario.read_trace(trace, names, times)
    skin = game_check.Skin(body, names)
    doc, _ = game_measure.glb.read_glb(package/runtime['glb'])
    base = render.load(package/runtime['glb'], textured=bool(doc.get('images')), multiply=True)
    ground = game_scenario.Ground(scenario)
    patches = [e for e in scenario['events'] if e['type'] == 'terrain_patch']
    heights_at = []
    for t in times:
        g = game_scenario.Ground(scenario)
        for p in patches:
            if p['time'] <= t+1e-9:
                g.patch(p['rect'], p['offset'])
        heights_at.append(g.heights)
    # one camera for the run: the character's bounds over every frame and the ground under its path
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for k in range(0, len(times), max(1, len(times)//60)):
        v = skin.vertices(skin.world(roots[k], mats[k]))
        lo, hi = np.minimum(lo, v.min(axis=0)), np.maximum(hi, v.max(axis=0))
    lo_b, hi_b = rig_look.to_blender(lo), rig_look.to_blender(hi)
    box_lo, box_hi = np.minimum(lo_b, hi_b), np.maximum(lo_b, hi_b)
    out.mkdir(parents=True, exist_ok=True)
    views = ['three-quarter', 'right']
    picks = np.linspace(0, len(times)-1, phases).round().astype(int)
    sheet = Image.new('RGB', (size*phases, size*len(views)))
    for row, view in enumerate(views):
        shot = rig_look.shot(view, box_lo, box_hi)
        if view == 'right':
            shot['cam_pos'][1] += .12*np.linalg.norm(np.asarray(shot['cam_pos'])-shot['cam_target'])
        for col, k in enumerate(picks):
            frame = rig_look.draw(combine(character_mesh(skin, base, roots[k], mats[k]), ground_mesh(ground, heights_at[k])), shot, size)
            ImageDraw.Draw(frame).text((8, 8), f'{view} / {times[k]:.3f}s', fill=(20, 20, 20))
            sheet.paste(frame, (col*size, row*size))
    sheet.save(out/'filmstrip.png')
    shot = rig_look.shot('three-quarter', box_lo, box_hi)
    step = max(1, math.ceil(len(times)/120))
    frames = []
    for k in range(0, len(times), step):
        frame = rig_look.draw(combine(character_mesh(skin, base, roots[k], mats[k]), ground_mesh(ground, heights_at[k])), shot, size)
        ImageDraw.Draw(frame).text((8, 8), f'{times[k]:.3f}s', fill=(20, 20, 20))
        frames.append(frame)
    span = float(times[-1]-times[0]) or 1.0
    frames[0].save(out/'run.gif', save_all=True, append_images=frames[1:], duration=max(20, round(span/len(frames)*1000)), loop=0)
    return out/'filmstrip.png'
