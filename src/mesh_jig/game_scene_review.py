"""Bounded temporal pictures of runtime skin on its actual scenario ground.

CPU reference scenes, not browser screenshots or a second visual acceptance gate. All
paths and cameras are harness-owned; the proposal supplies no executable render code.
"""
from __future__ import annotations

import copy
import json
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import game_check, game_look, game_runtime, game_scenario as gs, glb, render, rig_look, terrain

SIZE = 224
MAX_STRESS_SECONDS = 8
VIEWS = {'side': np.array([1., .2, 0.]), 'quarter': np.array([.8, .35, .8])}


def frame_indices(trace, check):
    """Context plus three adjacent frames around the worst failed gate, at most seven."""
    frames = trace['frames']
    times = np.array([f['time'] for f in frames])
    gates = check.get('gates', {})
    failed = [g for g in gates.values() if not g['passed'] and g.get('worst_time_s') is not None]
    # Prefer a contact/trajectory failure over a cosmetic summary reading.
    bad = next((gates[k] for k in ('slip_mps', 'penetration_mm', 'joint_speed_mps')
                if k in gates and gates[k] in failed), failed[0] if failed else None)
    if bad is None:
        return sorted(set(np.linspace(0,len(frames)-1,min(7,len(frames))).round().astype(int).tolist()))
    ids = {0, len(frames)//3, 2*len(frames)//3, len(frames)-1}
    if bad:
        k = int(np.abs(times-bad['worst_time_s']).argmin())
        ids.update(range(max(0, k-1), min(len(frames), k+2)))
    return sorted(ids)


def stress_scenario(runtime, base):
    """A short scaled staircase, uneven ridge and wall; visual diagnostics, not new job gates.

    This is not the site's 24 m course. Depth and height follow this rig's reach so a
    miniature rig does not get a human-sized obstacle. The exact scene is saved.
    """
    spec = copy.deepcopy(base)
    reach = min(sum(l['segment_lengths_m']) for l in runtime['limbs'])
    speed = abs(spec['root'][-1][3]-spec['root'][0][3])/spec['duration']
    skeleton = np.array([j['rest_position'] for j in runtime['skeleton']['joints']])
    span = max(float(np.ptp(skeleton,axis=0).max()),reach)
    duration = min(MAX_STRESS_SECONDS,max(spec['duration'],4*span/speed))
    length = speed*duration
    spec['duration'] = duration
    start,end = spec['root'][0],spec['root'][-1]
    spec['root'] = [list(start),[duration,end[1],end[2],start[3]-length,end[4]]]
    width = max(.5, reach*3)
    # Thin ramps approximate risers while remaining a supported continuous heightfield.
    rows = [(0,0),(.15,0),(.151,.15),(.25,.15),(.251,.3),(.35,.3),(.351,0),
            (.5,0),(.56,.22),(.62,.06),(.68,.28),(.74,0),(.84,0),(.841,.5),(.9,.5),(.901,0),(1,0)]
    zs = [-length-span]+[-length*f for f,_ in reversed(rows)]+[span]
    heights = [[h*reach*.6*(1+side*.2) for side in (-1,0,1)] for _,h in reversed(rows)]
    heights = [[0,0,0]]+heights+[[0,0,0]]
    spec['terrain'] = {'x':[-width,0,width], 'z':zs, 'heights':heights}
    spec.pop('expect', None)
    spec['events'] = [e for e in spec['events'] if e['type']!='set_ground_enabled']
    return spec


def capture(package, cases, out):
    """Three matched fastest-gait scenes, two angles each, with exact stance labels.

    cases contain harness-owned scenario, trace and check objects. Saved metadata pins
    the runtime/GLB and reports diagnostics even when the candidate failed acceptance.
    """
    package, out = Path(package), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runtime = json.loads((package/'character.runtime.json').read_text())
    gaits = [(name,c['travel_speed_mps']) for name,c in runtime['clips'].items()
             if c.get('travel_speed_mps') and any(x['label']==name+'-slope' for x in cases)]
    if not gaits:
        raise ValueError('no measured gait scenario is available for scene review')
    gait = max(gaits,key=lambda item:item[1])[0]
    chosen = [x for label in (gait+'-playback',gait+'-slope') for x in cases if x['label']==label]
    base = next(x for x in cases if x['label']==gait+'-flat')
    stress = stress_scenario(runtime,base['scenario'])
    trace = game_runtime.run(package,stress)
    checked = game_check.check(package,stress,trace)
    chosen.append({'label':gait+'-obstacles','scenario':stress,'trace':trace,'check':checked})
    player = game_runtime.Runtime(package)
    skinner = game_check.Skin(player.body,player.names)
    doc,_ = glb.read_glb(package/runtime['glb'])
    base_mesh = render.load(package/runtime['glb'],textured=bool(doc.get('images')),multiply=True)
    pictures, summaries = [], []
    for number, case in enumerate(chosen):
        scenario, trace, checked = case['scenario'],case['trace'],case['check']
        # Validate provenance and matrix shape before drawing the trace.
        gs.read_trace(trace,player.names,gs.frame_times(scenario))
        expected = {'glb_sha256':terrain.digest(package/runtime['glb']),
                    'runtime_sha256':terrain.digest(package/'character.runtime.json'),
                    'scenario_sha256':gs.digest_json(scenario)}
        if trace['hashes'] != expected:
            raise ValueError('scene trace does not match its package/scenario')
        selected = frame_indices(trace,checked)
        states = {}
        def observe(r,t,root,world,ground):
            k = len(states)
            states[k] = {'planted':[l['name'] for l in r.legs if r.planted(l,r.votes[l['name']])],
                         'votes':r.votes, 'ground':copy.deepcopy(ground)}
        # Stance labels come from actual graph votes, including phase offsets/layers.
        replay = game_runtime.run(package,scenario,observe=observe)
        if replay != trace:
            raise ValueError('scene replay differs from checked runtime trace')
        meshes, centres, metadata = [], [], []
        for k in selected:
            f = trace['frames'][k]
            root = gs.matrix(f['root'])
            matrices = np.array([gs.matrix(m) for m in f['joints']])
            body = game_look.character_mesh(skinner,base_mesh,root,matrices)
            relative = body.tris-root[:3,3]
            centres.append(relative.reshape(-1,3))
            floor = states[k]['ground']
            meshes.append((game_look.combine(body,game_look.ground_mesh(floor,floor.heights)),root[:3,3]))
            metadata.append({'frame':k,'time':f['time'],'planted':states[k]['planted'],
                             'votes':states[k]['votes']})
        bounds = np.concatenate(centres)
        lo,hi = bounds.min(axis=0),bounds.max(axis=0)
        centre, span = (lo+hi)/2, max(float(np.max(hi-lo))*1.55,.1)
        sheet = Image.new('RGB',(SIZE*len(selected),2*(SIZE+58)),'white')
        ink = ImageDraw.Draw(sheet)
        for row,(name,direction) in enumerate(VIEWS.items()):
            for col,((mesh,root),meta) in enumerate(zip(meshes,metadata)):
                aim = root+centre
                shot = {'cam_pos':(aim+direction*(span+10)).tolist(),'cam_target':aim.tolist(),
                        'ortho':span,'light':'camera'}
                x,y = col*SIZE,row*(SIZE+58)
                sheet.paste(rig_look.draw(mesh,shot,SIZE),(x,y+58))
                votes = [v for leg in meta['votes'].values() for v in leg]
                clip,t,_ = max(votes,key=lambda v:v[2])
                phase = t/runtime['clips'][clip]['duration_s']
                ink.text((x+3,y+2),f'{case["label"][:24]} / {name} / {meta["frame"]}',fill='black')
                ink.text((x+3,y+17),f'{meta["time"]:.3f}s / {clip[:14]} phase {phase:.2f}',fill='black')
                planted = 'planted: '+(','.join(meta['planted']) or 'none')
                for line,text in enumerate(textwrap.wrap(textwrap.shorten(planted,width=64),width=33)):
                    ink.text((x+3,y+32+line*12),text,fill='black')
        path = out/f'scene-{number+1:02d}.png'
        sheet.save(path)
        pictures.append(path)
        stem = out/f'scene-{number+1:02d}'
        for suffix,value in (('scenario',scenario),('trace',trace),('check',checked)):
            stem.with_suffix('.'+suffix+'.json').write_text(json.dumps(value,allow_nan=False),encoding='utf-8',newline='\n')
        summaries.append({'label':case['label'],'frames':metadata,'hashes':expected,'passed':checked['passed'],
                          'diagnostic_only':case['label'].endswith('-obstacles')})
    (out/'scenes.json').write_text(json.dumps(summaries,allow_nan=False),encoding='utf-8',newline='\n')
    return pictures
