"""`game-build`: measure an animated GLB against its `character.json`, gate it, and write the package.

`package/` holds `character.glb` (the input, byte for byte), `character.runtime.json` (the neutral description in
glTF axes, everything measured filled in) and `character.report.json` (every gate and reading). A build that fails
a gate writes the report alone. See docs/GAME_READY.md.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from . import game, glb, skin, terrain
from . import game_measure as gm

r = gm.r


def _v(a):
    return [r(x) for x in a]


# --- the graph, evaluated at rest, for the poses at each end of a fade -------------------------------------------

def clip_pose(body, clip, loops, t):
    d = body.clips[clip].duration
    return body.full(clip, t % d if loops[clip] else min(t, d))


def node_clips(node):
    if node is None:
        return []
    kind = node['type']
    if kind == 'clip':
        return [node['clip']]
    if kind == 'blend1d':
        return [c for p in node['points'] for c in ([p['clip']] if 'clip' in p else node_clips(p['node']))]
    if kind == 'states':
        return [c for s in node['states'].values() for c in node_clips(s)]
    return list(node['clips'])


def node_pose(body, node, loops, params, t):
    """A resolved graph node's joint local transforms t seconds after it started, every parameter at its default;
    None for a node that adds nothing (a null state; one-shots when none is playing)."""
    if node is None or node['type'] == 'oneshots':
        return None
    if node['type'] == 'clip':
        return clip_pose(body, node['clip'], loops, t)
    if node['type'] == 'states':
        return node_pose(body, node['states'][node['start']], loops, params, t)
    points, x = node['points'], params[node['parameter']]
    values = [p['value'] for p in points]
    if any(v is None for v in values) or any(b <= a for a, b in zip(values, values[1:])) or x <= values[0]:
        i, w = 0, 0.0
    elif x >= values[-1]:
        i, w = len(points)-2, 1.0
    else:
        i = max(k for k in range(len(points)-1) if values[k] <= x)
        w = (x-values[i])/(values[i+1]-values[i])
    a, b = points[i], points[i+1]
    if node.get('sync'):
        ta, tb = gm.sync_times([body.clips[p['clip']].duration for p in (a, b)], [a['phase_offset'], b['phase_offset']],
                               [a['cycle'], b['cycle']], w, t)
        return gm.blend(body.full(a['clip'], ta), body.full(b['clip'], tb), w)
    pa, pb = ({'type': 'clip', 'clip': p['clip']} if 'clip' in p else p['node'] for p in (a, b))
    return gm.blend(node_pose(body, pa, loops, params, t), node_pose(body, pb, loops, params, t), w)


def fades(body, layers, masks, loops, params):
    """Each state transition and one-shot fade: the largest distance between the masked joints' positions at its
    two ends (joints outside the layer's mask held at rest), and that distance over the fade time."""
    out = []
    for k, layer in enumerate(layers):
        node, mask = layer['node'], masks[k]
        if node['type'] not in ('states', 'oneshots'):
            continue
        names = mask or body.names
        ids = [body.node[n] for n in names]

        def place(trs):
            pose = {i: dict(v) for i, v in body.rest_trs.items()}
            for i in (ids if mask else trs):
                pose[i] = trs[i]
            return body.joints(body.world(pose), names)

        def below_pose(t):
            pose = {i: dict(v) for i, v in body.rest_trs.items()}
            for lower, m in zip(layers[:k], masks[:k]):
                p = node_pose(body, lower['node'], loops, params, t)
                if p is not None:
                    for i in ([body.node[n] for n in m] if m else p):
                        pose[i] = p[i]
            return pose
        span = max([body.clips[c].duration for lower in layers[:k] for c in node_clips(lower['node'])] or [1.0])
        below = np.array([place(below_pose(float(t))) for t in gm.sample_times(span, True)])

        def at(n, t):
            return below if n is None else place(node_pose(body, n, loops, params, t))[None]

        def over(n):
            if n is None:
                return below
            d = max(body.clips[c].duration for c in node_clips(n))
            return np.array([place(node_pose(body, n, loops, params, float(t))) for t in gm.sample_times(d, True)])

        def apart(a, b):
            return max(float(np.linalg.norm(b-x, axis=2).max()) for x in a)
        if node['type'] == 'states':
            for tr in node['transitions']:
                src, dst = node['states'][tr['from']], node['states'][tr['to']]
                a = at(src, body.clips[src['clip']].duration) if tr.get('at_end') else over(src)
                d = apart(a, at(dst, 0.0))
                out.append({'name': f'{layer["name"]}: {tr["from"]} -> {tr["to"]}', 'fade_s': tr['fade_s'],
                            'distance_m': d, 'speed_mps': d/tr['fade_s']})
        else:
            for clip, f in sorted(node['clips'].items()):
                one = {'type': 'clip', 'clip': clip}
                for end, a, fade in (('in', at(one, 0.0), f['fade_in_s']),
                                     ('out', at(one, body.clips[clip].duration), f['fade_out_s'])):
                    d = apart(a, below)
                    out.append({'name': f'{layer["name"]}: {clip} fades {end}', 'fade_s': fade, 'distance_m': d,
                                'speed_mps': d/fade})
    return out


# --- measurement, gates and readings ------------------------------------------------------------------------------

def gate(items, limit, direction='max'):
    known = {k: v for k, v in items.items() if v is not None}
    if not known:
        return {'value': None, 'limit': limit, 'direction': direction, 'passed': True, 'worst': None, 'items': {}}
    worst = max(sorted(known), key=lambda k: known[k])
    value = known[worst]
    return {'value': r(value), 'limit': limit, 'direction': direction, 'passed': bool(value <= limit),
            'worst': worst, 'items': {k: r(v) for k, v in sorted(items.items())}}


def measure(spec, body, res):
    found, criteria, legs = res['found'], spec['criteria'], res['legs']
    gap = criteria.get('contact_gap_mm_max', gm.TOUCH_MM)/1000
    shortest = criteria.get('stance_s_min', 0.0)
    forward = res['forward']
    clips, reads, samples = {}, {}, {}
    for c in sorted(found['clips']):
        info, s, read = gm.clip_read(body, c, res['loops'][c], legs, gap, forward, res['roots'], shortest)
        clips[c], reads[c], samples[c] = info, (info, read), s
    order = list(legs)
    items = {k: {} for k in game.GATES}
    blends, layers = [], copy.deepcopy(spec['graph']['layers'])
    resolved = {where: node for where, node in game.graph({**spec, 'graph': {'layers': layers}})['blends']}
    for where, node in found['blends']:
        live = resolved[where]
        label = where.replace('graph.layers', 'layer').replace('.node', '')
        param = spec['parameters'][node['parameter']]
        for p in live['points']:
            if p['value'] == 'measured':
                p['value'], p['measured'] = r(clips[p['clip']]['travel_speed_mps']), True
            elif 'clip' in p:
                p['measured'] = False
        values = [p['value'] for p in live['points']]
        bad = sum(v is None or not param['min'] <= v <= param['max'] for v in values)
        bad += sum(a is not None and b is not None and b <= a for a, b in zip(values, values[1:]))
        items['points_in_order'][label] = bad
        entry = {'name': label, 'parameter': node['parameter'], 'sync': bool(node.get('sync')),
                 'points': [p.get('clip', 'node') for p in node['points']], 'values': values, 'pairs': []}
        if node.get('sync'):
            live['sync_clock'] = 'frequency_lerp'
            names = [p['clip'] for p in node['points']]
            ref, stepping, ph = gm.phases(reads, names, order) if legs else (None, [], {c: {'phase_offset': 0.0, 'relative': {}} for c in names})
            # a clip in which no leg touches down (an idle) is off the cycle: its own clock, no part in the frequency
            cycles = {c: c in stepping for c in names}
            for p in live['points']:
                p['phase_offset'], p['cycle'] = r(ph[p['clip']]['phase_offset']), cycles[p['clip']]
            entry.update(reference_leg=ref, phase_offsets={c: r(ph[c]['phase_offset']) for c in names},
                         on_cycle=cycles)
            for a, b in zip(names, names[1:]):
                pair = {'clips': [a, b]}
                if a in stepping and b in stepping:
                    pair['gait_phase_difference'] = gm.phase_difference(ph[a]['relative'], ph[b]['relative'])
                    items['gait_phase_difference'][f'{label}: {a}-{b}'] = pair['gait_phase_difference']
                    pair['order'] = {c: sorted(order, key=lambda l: (ph[c]['relative'][l] or [9])[0]) for c in (a, b)}
                speeds = {c: clips[c]['travel_speed_mps'] for c in (a, b)}
                if legs and None not in speeds.values():
                    slip = gm.blend_read(body, a, b, {c: ph[c]['phase_offset'] for c in (a, b)}, cycles, legs, gap, forward,
                                         speeds, shortest)
                    pair['blend'] = {'period_s': r(slip['period_s']), 'speed_mps': r(slip['speed_mps']),
                                     'slip_mps': {k: r(v) for k, v in slip['slip_mps'].items()}}
                    items['blend_slip_mps'][f'{label}: {a}-{b}'] = slip['worst_slip_mps']
                entry['pairs'].append(pair)
        blends.append(entry)
    for c in sorted(found['clips']):
        info = clips[c]
        if info['loop']:
            items['loop_seam_mm'][c] = info['loop_seam_mm']
        if c in found['locomotion']:
            items['root_travel_mm'][c] = info['root_travel_mm']
            if legs:
                items['legs_planted'][c] = sum(1 for l in info['legs'].values() if not l['stance_s'])
                for leg, x in info['legs'].items():
                    items['skate_mps'][f'{c}/{leg}'] = x['skate_mps']
    masks = res['masks']
    fade_rows = fades(body, layers, masks, res['loops'], {k: v['default'] for k, v in spec.get('parameters', {}).items()})
    for f in fade_rows:
        items['fade_joint_speed_mps'][f['name']] = f['speed_mps']
    gates = {}
    for name, (criterion, _) in game.GATES.items():
        if criterion is None:
            if items[name] or (name == 'legs_planted' and legs and found['locomotion']) or (name == 'points_in_order' and found['blends']):
                gates[name] = gate(items[name], 0)
        elif criterion in criteria:
            gates[name] = gate(items[name], criteria[criterion])
    # readings: never gates
    contact_ids = np.concatenate(list(legs.values())) if legs else np.zeros(0, int)
    missed = []
    for c in sorted(found['locomotion']):
        for t in gm.touching(body, samples[c], gap, contact_ids):
            missed.append({'clip': c, **{k: t[k] for k in ('joint', 'nodes', 'vertices', 'box', 'share')}})
    rigid, index = [], {n: k for k, n in enumerate(body.names)}
    for limb in spec.get('limbs', []):
        if len(limb['joints']) < 3:
            continue
        a = np.concatenate([gm.angles(samples[c].joints, [index[j] for j in limb['joints']]) for c in sorted(samples)])
        moved = float(np.ptp(a, axis=0).max())
        if moved < gm.BEND_DEG:
            rigid.append({'limb': limb['name'], 'bend_range_degrees': r(moved)})
    outside = []           # a clip's own bend outside the limb's declared range: what the IK will be held to
    for limb in spec.get('limbs', []):
        if limb['solver'] == 'none':
            continue
        for chain in zip(limb['joints'], limb['joints'][1:], limb['joints'][2:]):
            for c in sorted(samples):
                lo, hi = gm.bend_ranges(body, samples[c].joints, [chain])[tuple(chain)]
                if limb['solver'] == 'two_bone':
                    low, high = limb['bend_degrees']
                    if lo < low or hi > high:
                        outside.append({'limb': limb['name'], 'joint': chain[1], 'clip': c, 'inner_degrees': [r(lo), r(hi)],
                                        'declared': [low, high]})
                elif 180-lo > limb['bend_degrees_max']:
                    outside.append({'limb': limb['name'], 'joint': chain[1], 'clip': c, 'bend_degrees_max': r(180-lo),
                                    'declared': limb['bend_degrees_max']})
    grazes = [{'clip': c, 'leg': leg, 'time_s': r(g['time_s']), 'clearance_mm': r(g['clearance_mm'])}
              for c in sorted(clips) for leg, x in clips[c]['legs'].items() for g in x['grazes']]
    readings = {'unused_joints': sorted(set(body.names)-set(res['named'])), 'missed_contacts': missed,
                'rigid_limbs': rigid, 'grazes': grazes, 'bend_outside': outside,
                'below_rest_ground_mm': {c: r(clips[c]['below_rest_ground_mm']) for c in sorted(clips)}}
    measured = {'clips': {c: _clip_record(v) for c, v in clips.items()}, 'blends': blends,
                'fades': [{k: (r(v) if isinstance(v, float) else v) for k, v in f.items()} for f in fade_rows],
                'contacts': {k: int(len(v)) for k, v in legs.items()}, 'contact_gap_m': r(gap), 'sample_hz': gm.HZ}
    return gates, readings, measured, layers


def _clip_record(info):
    out = {k: (r(v) if isinstance(v, float) else v) for k, v in info.items() if k != 'legs'}
    out['legs'] = {leg: {'duty': r(x['duty']), 'stance_s': [[r(a), r(b)] for a, b in x['stance_s']],
                         'touchdowns_s': [r(t) for t in x['touchdowns_s']], 'speed_mps': r(x['speed_mps']),
                         'grazes': [{k: r(v) for k, v in g.items()} for g in x['grazes']],
                         'spread_mps': r(x['spread_mps']), 'skate_mps': r(x['skate_mps'])}
                   for leg, x in info['legs'].items()}
    return out


# --- the runtime file ---------------------------------------------------------------------------------------------

def runtime(spec, body, res, measured, layers, hashes):
    C, found = gm.C, res['found']
    rest = {n: body.rest_world[body.node[n]] for n in body.names}
    joints = []
    for n in body.names:
        t = body.rest_trs[body.node[n]]
        joints.append({'name': n, 'node': body.node[n], 'parent': body.parent[n], 'rest_translation': _v(t['translation']),
                       'rest_rotation': _v(t['rotation']), 'rest_scale': _v(t['scale']), 'rest_position': _v(rest[n][:3, 3])})
    clips = {}
    for c, m in measured['clips'].items():
        d = m['duration_s']
        clips[c] = {'duration_s': d, 'loop': m['loop'], 'locomotion': c in found['locomotion'],
                    'travel_speed_mps': m['travel_speed_mps'], 'root_travel_m': r(m['root_travel_mm']/1000),
                    'loop_seam_m': r(m['loop_seam_mm']/1000), 'events': body.clips[c].events,
                    'legs': {leg: {'duty': x['duty'], 'stance_phase': [[r(a/d), r(b/d)] for a, b in x['stance_s']],
                                   'touchdown_phase': [r(t/d) for t in x['touchdowns_s']], 'speed_mps': x['speed_mps'],
                                   'spread_mps': x['spread_mps']} for leg, x in m['legs'].items()}}
    graph = {'layers': [{**layer, 'mask_joints': res['masks'][k] or None} for k, layer in enumerate(layers)]}
    limbs = []
    for l in spec.get('limbs', []):
        pts = [rest[j][:3, 3] for j in l['joints']]
        row = {'name': l['name'], 'role': l['role'], 'solver': l['solver'], 'joints': l['joints'],
               'segment_lengths_m': [r(np.linalg.norm(b-a)) for a, b in zip(pts, pts[1:])]}
        if l['solver'] == 'two_bone':
            row.update(pole=_v(C @ np.asarray(l['pole'], float)), bend_degrees=l['bend_degrees'])
        if l['solver'] == 'chain':
            row['bend_degrees_max'] = l['bend_degrees_max']
        if l['role'] == 'leg':
            tip, ids = l['joints'][-1], res['legs'][l['name']]
            p = body.rest[ids] @ C.T                      # Blender rows -> glTF rows
            local = (np.linalg.inv(rest[tip]) @ np.c_[p, np.ones(len(p))].T).T[:, :3]
            row['contact'] = {'joint': tip, 'vertices': int(len(ids)),
                              'points': (np.unique(np.round(local, gm.PLACES), axis=0)+0.0).tolist()}
            row['clearance'] = clearance(body, rest[tip], tip, ids, float(p[:, 1].min()))
        limbs.append(row)
    aims = []
    for a in spec.get('aims', []):
        axis = C @ np.asarray(a['axis'], float)
        axis /= np.linalg.norm(axis)
        local = np.linalg.solve(rest[a['joints'][-1]][:3, :3], axis)
        aims.append({**{k: a[k] for k in ('name', 'joints', 'yaw_degrees', 'pitch_degrees', 'degrees_per_s')},
                     'axis_model': _v(axis), 'axis_local': _v(local/np.linalg.norm(local))})
    sockets = [{'name': s['name'], 'joint': s['joint'], 'offset_model': _v(C @ np.asarray(s['offset'], float)),
                'offset_local': _v(np.linalg.solve(rest[s['joint']][:3, :3], C @ np.asarray(s['offset'], float)))}
               for s in spec.get('sockets', [])]
    forward = C @ np.r_[res['forward'], 0]
    return {'format': 'mesh-jig game-ready character', 'version': 1, 'glb': 'character.glb',
            'axes': {'units': 'metres', 'up': [0, 1, 0], 'forward': _v(forward), 'handedness': 'right',
                     'from_contract': 'Blender (x, y, z) -> glTF (x, z, -y)'},
            'skeleton': {'roots': [n for n in body.names if body.parent[n] is None], 'joints': joints},
            'parameters': spec.get('parameters', {}), 'triggers': sorted(found['triggers']),
            'oneshots': sorted(found['oneshots']), 'clips': clips, 'graph': graph,
            'body': spec.get('body'), 'limbs': limbs, 'ground': spec.get('ground'), 'aims': aims, 'sockets': sockets,
            'blending': {'rotation': 'shortest-arc nlerp', 'translation': 'lerp', 'scale': 'lerp',
                         'sync': 'one normalised phase; cycle frequency lerps between the two points; each clip at '
                                 'phase + its phase_offset'},
            'measurement': {'sample_hz': gm.HZ, 'contact_gap_m': measured['contact_gap_m'],
                            'rest_ground_y': r(float(body.rest[:, 2].min()))},
            'capabilities': capabilities(spec, found, measured), 'hashes': hashes}


CLEARANCE_POINTS = 32


def farthest(points, count):
    """Deterministic farthest-point sampling: start from the first point in lexicographic order, then add, each
    time, the point farthest from those chosen (the lower index on a tie)."""
    order = np.lexsort(points.T[::-1])
    points = points[order]
    if len(points) <= count:
        return points
    chosen, distance = [0], np.linalg.norm(points-points[0], axis=1)
    while len(chosen) < count:
        k = int(np.argmax(distance))
        chosen.append(k)
        distance = np.minimum(distance, np.linalg.norm(points-points[k], axis=1))
    return points[sorted(chosen)]


def clearance(body, tip_rest, tip, contact_ids, plane_y):
    """The rest of a foot, for keeping it out of the ground: the vertices the tip joint carries most, less the
    contact, rounded and made unique, then at most CLEARANCE_POINTS by farthest-point sampling. Points in the tip
    joint's rest frame (glTF); heights above the contact plane (the contact's lowest rest point, model up)."""
    carried = np.flatnonzero(np.array([str(j) == tip for j in body.vjoint]))
    carried = np.setdiff1d(carried, contact_ids)
    if not len(carried):
        return {'points': [], 'heights': []}
    model = np.unique(np.round(body.rest[carried] @ gm.C.T, gm.PLACES), axis=0)
    model = farthest(model, CLEARANCE_POINTS)
    local = (np.linalg.inv(tip_rest) @ np.c_[model, np.ones(len(model))].T).T[:, :3]
    return {'points': (np.round(local, gm.PLACES)+0.0).tolist(), 'heights': [r(y-plane_y) for y in model[:, 1]]}


def capabilities(spec, found, measured):
    legs = [l for l in spec.get('limbs', []) if l['role'] == 'leg']
    blends = [b for b in measured['blends']]
    out = {'locomotion': '; '.join(f'{b["parameter"]} blends {", ".join(b["points"])}'
                                   f'{" on one phase" if b["sync"] else ""}' for b in blends) or 'none: no blend1d',
           'not_included': 'root motion, physics, navigation and movement: the game moves the character and the '
                           'runtime never writes its transform'}
    g = spec.get('ground')
    out['ground_adaptation'] = (f'{len(legs)} legs ({", ".join(l["name"] for l in legs)}) keep their contacts on the '
                                f'ground; {spec["body"]["joint"]} drops up to {g["drop_m_max"]} m and rises up to '
                                f'{g["rise_m_max"]} m; ' + ('tilts to the plane through the planted contacts' if g['tilt']
                                                            else 'no tilt')) if g else 'none'
    words = {'two_bone': 'two-bone IK bent toward its pole', 'chain': 'FABRIK chain, root pinned', 'none': 'no IK'}
    out['limbs'] = '; '.join(f'{l["name"]} ({l["role"]}): {words[l["solver"]]} on {", ".join(l["joints"])}'
                             for l in spec.get('limbs', [])) or 'none'
    out['aims'] = ', '.join(a['name'] for a in spec.get('aims', [])) or 'none'
    out['sockets'] = ', '.join(s['name'] for s in spec.get('sockets', [])) or 'none'
    out['triggers'] = ', '.join(sorted(found['triggers'])) or 'none'
    out['oneshots'] = ', '.join(sorted(found['oneshots'])) or 'none'
    return out


# --- the build ----------------------------------------------------------------------------------------------------

def measurement_digest():
    h = hashlib.sha256()
    for module in (game, gm, glb, skin):
        path = Path(module.__file__)
        h.update(path.name.encode())
        h.update(path.read_bytes())
    h.update(Path(__file__).read_bytes())
    h.update(np.__version__.encode())
    return h.hexdigest()


def dump(data):
    return (json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+'\n').encode('utf-8')


def build(rig, spec, out, *, diagnostic=False):
    """Measure, gate and write the package. Invalid input raises ValueError and writes nothing."""
    rig, out = Path(rig), Path(out)
    game.contract(spec)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('the output directory must be new or empty')
    digest = terrain.digest(rig)
    if digest != spec['rig_sha256']:
        raise ValueError(f'rig_sha256 does not match the GLB (it is {digest})')
    body = gm.Body(rig)
    res = game.resolve(spec, body)
    gates, readings, measured, layers = measure(spec, body, res)
    passed = all(g['passed'] for g in gates.values())
    issues = [f'{k}: {g["worst"]} {g["value"]:.6g}, needs at most {g["limit"]:g}' for k, g in gates.items() if not g['passed']]
    hashes = {'glb_sha256': digest, 'contract_sha256': terrain.spec_digest(spec), 'measurement_sha256': measurement_digest()}
    report = {'version': 1, 'passed': passed, 'issues': issues, 'gates': gates, 'readings': readings,
              'measured': measured, 'hashes': dict(hashes),
              'limits': ['clips are read at a fixed sample rate from posed vertices; a planted contact is one within '
                         'the contact gap of the clip\'s lowest vertex, so duty depends on that gap',
                         'no run-time check: the runtime, scenarios and trace checker are later slices',
                         'readings are not gates; appearance and naturalness need a person to look']}
    out.mkdir(parents=True, exist_ok=True)
    if passed:
        shutil.copyfile(rig, out/'character.glb')
        data = dump(runtime(spec, body, res, measured, layers, hashes))
        (out/'character.runtime.json').write_bytes(data)
        from . import game_godot
        game_godot.write(out)                     # godot/: the scene and a copy of the addon (slice 3)
        report['hashes'].update(package_glb_sha256=terrain.digest(out/'character.glb'),
                                runtime_sha256=hashlib.sha256(data).hexdigest())
    elif diagnostic:
        # A failed candidate can be inspected, but never occupies the delivery paths.
        preview = out/'diagnostic'
        preview.mkdir()
        shutil.copyfile(rig, preview/'character.glb')
        (preview/'character.runtime.json').write_bytes(dump(runtime(spec, body, res, measured, layers, hashes)))
        (preview/'diagnostic.json').write_bytes(dump({'accepted': False, 'issues': issues}))
    (out/'character.report.json').write_bytes(dump(report))
    return report
