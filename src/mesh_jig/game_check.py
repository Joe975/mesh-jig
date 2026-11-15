"""`game-check`: hold a runtime's trace to the gates, independently of any runtime.

It reads the package, the scenario and the trace, skins the real mesh from the trace's joint matrices and measures
what a player would see. It imports no runtime and trusts nothing a trace says about its own state: planted, aim
and target state are all read from the posed mesh and joints. Every limit is a key of the scenario's `criteria`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import game_measure as gm
from . import game_scenario as gs
from . import terrain

C = gm.C


def to_gltf(points):
    return np.asarray(points) @ C.T


class Skin:
    """Skins the package's mesh from joint world matrices, in glTF world axes."""

    def __init__(self, body, names):
        self.body, self.names = body, names
        self.index = {n: k for k, n in enumerate(names)}
        joints = set(body.node.values())
        self.carrier = {}
        for i in set(body.pair_nodes)-joints:     # a mesh node that is no joint follows its nearest joint ancestor
            up = body.parent_node.get(i)
            while up is not None and up not in joints:
                up = body.parent_node.get(up)
            self.carrier[i] = up

    def world(self, root, matrices):
        b = self.body
        world = {b.node[n]: matrices[k] for n, k in self.index.items()}
        for i, up in self.carrier.items():
            world[i] = (root if up is None else world[up] @ np.linalg.inv(b.rest_world[up])) @ b.rest_world[i]
        return world

    def vertices(self, world):
        return to_gltf(self.body.vertices(world))


def contact_ids(body, leg):
    """The package's contact for a leg as mesh vertices: those whose rest position is one of its contact points."""
    tip = body.rest_world[body.node[leg['joints'][-1]]]
    points = np.asarray(leg['contact']['points'], float) @ tip[:3, :3].T+tip[:3, 3]
    rest = to_gltf(body.rest)
    near = np.linalg.norm(rest[:, None, :]-points[None], axis=2).min(axis=1) < 1e-5
    if not near.any():
        raise ValueError(f'leg {leg["name"]}: no mesh vertex is at its contact points')
    return np.flatnonzero(near)


def stance_count(runtime, clip, legs, phase):
    c = runtime['clips'][clip]
    return sum(any(a <= phase <= b or a <= phase+1 <= b for a, b in c['legs'][l['name']]['stance_phase']) for l in legs)


def gate(values, limit, direction='max'):
    """values: [(reading, time)]."""
    if not values:
        return {'value': None, 'limit': limit, 'direction': direction, 'passed': True, 'worst_time_s': None}
    value, at = (max if direction == 'max' else min)(values, key=lambda v: (v[0], -v[1]))
    passed = value <= limit if direction == 'max' else value >= limit
    return {'value': gm.r(value), 'limit': limit, 'direction': direction, 'passed': bool(passed), 'worst_time_s': gm.r(at)}


def check(package, scenario, trace, scenario_sha256=None):
    """`scenario_sha256`: the SHA-256 of the scenario file's bytes (game_scenario.load); in memory, digest_json."""
    package = Path(package)
    runtime = json.loads((package/'character.runtime.json').read_text(encoding='utf-8'))
    gs.resolve(scenario, runtime)
    hashes = {'glb_sha256': terrain.digest(package/runtime['glb']),
              'runtime_sha256': terrain.digest(package/'character.runtime.json'),
              'scenario_sha256': scenario_sha256 or gs.digest_json(scenario)}
    if not isinstance(trace, dict) or trace.get('hashes') != hashes:
        raise ValueError('the trace was not made from this package and scenario (hashes differ)')
    body = gm.Body(package/runtime['glb'])
    names = [j['name'] for j in runtime['skeleton']['joints']]
    times = gs.frame_times(scenario)
    roots, mats = gs.read_trace(trace, names, times)
    crit, events = scenario['criteria'], scenario['events']
    gap = crit.get('contact_gap_mm_max', 0)/1000
    legs = [l for l in runtime['limbs'] if l['role'] == 'leg']
    ids = {l['name']: contact_ids(body, l) for l in legs}
    skin = Skin(body, names)
    ground = gs.Ground(scenario)
    patches = [e for e in events if e['type'] == 'terrain_patch']
    grace = crit.get('grace_s', 0.0)
    in_grace = np.array([any(p['time'] <= t < p['time']+grace for p in patches) for t in times])
    expect = scenario.get('expect')
    playback = 'playback_mm_max' in crit

    def expected_phase(t):
        c = runtime['clips'][expect['clip']]
        d = c['duration_s']
        tc = (t+expect['offset_s']) % d if c['loop'] else min(t+expect['offset_s'], d)
        return tc, tc/d

    pos = mats[:, :, :3, 3]                                   # joint world positions, frame by joint
    local = np.einsum('fij,fkj->fki', np.linalg.inv(roots)[:, :3, :3], pos-roots[:, None, :3, 3])
    rows, contact, gaps, touching, outside = [], {l['name']: [] for l in legs}, {}, {l['name']: [] for l in legs}, 0
    heights = ground.heights.copy()
    applied = 0
    for k, t in enumerate(times):
        while applied < len(patches) and patches[applied]['time'] <= t+1e-9:
            ground.patch(patches[applied]['rect'], patches[applied]['offset'])
            applied += 1
        heights = ground.heights
        world = skin.world(roots[k], mats[k])
        v = skin.vertices(world)
        points = np.concatenate([v, v[body.faces].mean(axis=1)])
        h = ground.height(points[:, 0], points[:, 2], heights)
        outside += int(np.isnan(h).sum())
        depth = float(np.nanmax(np.r_[h-points[:, 1], 0.0])) if np.isfinite(h).any() else 0.0
        row = {'time': float(t), 'penetration_m': max(0.0, depth), 'legs': {}}
        if depth > 0:
            deepest = int(np.nanargmax(h-points[:, 1]))
            vertex = deepest if deepest < len(v) else int(body.faces[deepest-len(v), 0])
            row['deepest'] = {'node': str(body.vnode[vertex]), 'joint': str(body.vjoint[vertex])}
        for l in legs:
            c = v[ids[l['name']]]
            clearance = c[:, 1]-ground.height(c[:, 0], c[:, 2], heights)
            contact[l['name']].append((c, clearance))
            low = float(np.nanmin(clearance)) if np.isfinite(clearance).any() else math.inf
            touching[l['name']].append(low <= gap)
            row['legs'][l['name']] = low
        if playback:
            tc, _ = expected_phase(t)
            model = body.world(body.full(expect['clip'], tc))
            want = to_gltf(body.vertices({i: roots[k] @ m for i, m in model.items()}))
            row['playback_m'] = float(np.linalg.norm(v-want, axis=1).max())
        rows.append(row)
    gates, readings = {}, {}
    # penetration and contact gap: outside a patch's grace time
    gates['penetration_mm'] = gate([(r['penetration_m']*1000, r['time']) for r, g in zip(rows, in_grace) if not g],
                                   crit['penetration_mm_max'])
    planted = {}
    if expect is not None and legs:
        for l in legs:
            windows = runtime['clips'][expect['clip']]['legs'][l['name']]['stance_phase']
            planted[l['name']] = [any(a <= ph <= b or a <= ph+1 <= b for a, b in windows)
                                  for ph in (expected_phase(t)[1] for t in times)]
        gates['contact_gap_mm'] = gate([(abs(r['legs'][n])*1000, r['time']) for n in planted for r, p, g in
                                        zip(rows, planted[n], in_grace) if p and not g], crit['contact_gap_mm_max'])
    if legs:
        slips, worst = [], []
        for l in legs:
            speeds = []
            for k in range(len(times)-1):
                (a, ga), (b, gb) = contact[l['name']][k], contact[l['name']][k+1]
                m = (ga <= gap) & (gb <= gap)
                if m.any() and times[k+1] > times[k]:
                    speeds.append((float(np.linalg.norm((b-a)[m][:, [0, 2]], axis=1).mean()/(times[k+1]-times[k])), times[k+1]))
            if speeds:
                rms = math.sqrt(sum(s*s for s, _ in speeds)/len(speeds))
                slips.append((rms, max(speeds)[1]))
                worst.append({'leg': l['name'], 'rms_mps': gm.r(rms), 'max_mps': gm.r(max(speeds)[0])})
        gates['slip_mps'] = gate(slips, crit['slip_mps_max'])
        readings['slip'] = worst
        clips = [expect['clip']] if expect else [c for c, x in runtime['clips'].items() if x['locomotion']]
        needed = min((stance_count(runtime, c, legs, p) for c in clips for p in np.linspace(0, 1, 241)[:-1]), default=0)
        count = np.array([sum(touching[n][k] for n in touching) for k in range(len(times))])
        ok = count >= needed
        span = float(times[-1]-times[0])
        fraction = float(np.sum(np.diff(times)*(ok[:-1] & ok[1:]))/span) if span > 0 else float(ok.all())
        gates['support_fraction'] = gate([(fraction, 0.0)], crit['support_fraction_min'], 'min')
        readings['support_legs_needed'] = needed
        readings['touching_fraction'] = {n: {'observed': gm.r(np.mean(touching[n])),
                                             'expected': gm.r(np.mean(planted[n])) if n in planted else None} for n in touching}
    # stretch: every limb segment's length against rest
    rest = {j['name']: np.asarray(j['rest_position'], float) for j in runtime['skeleton']['joints']}
    index = {n: k for k, n in enumerate(names)}
    stretch = []
    for limb in runtime['limbs']:
        for a, b in zip(limb['joints'], limb['joints'][1:]):
            length = np.linalg.norm(rest[b]-rest[a])
            d = np.abs(np.linalg.norm(pos[:, index[b]]-pos[:, index[a]], axis=1)-length)*1000
            stretch.append((float(d.max()), float(times[int(d.argmax())])))
    gates['stretch_mm'] = gate(stretch, crit['stretch_mm_max'])
    # continuity: the fastest joint, in the root's frame so the game's own movement is not counted
    dt = np.diff(times)
    speed = np.linalg.norm(np.diff(local, axis=0), axis=2)/np.where(dt > 0, dt, np.inf)[:, None]
    gates['joint_speed_mps'] = gate([(float(speed.max()), float(times[int(speed.max(axis=1).argmax())+1]))]
                                    if len(speed) else [], crit['joint_speed_mps_max'])
    # limits: knee angles, chain bends, aim yaw and pitch and turn rate
    violations, aims = [], {}
    for limb in runtime['limbs']:
        p = [pos[:, index[n]] for n in limb['joints']]
        for a, b, c in zip(p, p[1:], p[2:]):
            u, w = a-b, c-b
            inner = np.degrees(np.arccos(np.clip((u*w).sum(axis=1)/np.maximum(
                np.linalg.norm(u, axis=1)*np.linalg.norm(w, axis=1), 1e-12), -1, 1)))
            if limb['solver'] == 'two_bone':
                lo, hi = limb['bend_degrees']
                excess = np.maximum(lo-inner, inner-hi)
            elif limb['solver'] == 'chain':
                excess = (180-inner)-limb['bend_degrees_max']
            else:
                continue
            violations.append((float(max(0.0, excess.max())), float(times[int(excess.argmax())])))
    for aim in runtime['aims']:
        parent = runtime['skeleton']['joints'][index[aim['joints'][0]]]['parent']
        frames = mats[:, index[parent], :3, :3] if parent else roots[:, :3, :3]
        rest_parent = body.rest_world[body.node[parent]][:3, :3] if parent else np.eye(3)
        a0 = gm_unit(rest_parent.T @ np.asarray(aim['axis_model'], float))
        axes = np.einsum('fij,j->fi', mats[:, index[aim['joints'][-1]], :3, :3], np.asarray(aim['axis_local'], float))
        in_frame = np.einsum('fji,fj->fi', frames, axes)
        angles = np.array([gs.aim_angles(gm_unit(x), a0) for x in in_frame])
        (ylo, yhi), (plo, phi) = aim['yaw_degrees'], aim['pitch_degrees']
        excess = np.maximum.reduce([ylo-angles[:, 0], angles[:, 0]-yhi, plo-angles[:, 1], angles[:, 1]-phi])
        violations.append((float(max(0.0, excess.max())), float(times[int(excess.argmax())])))
        # turn rate: only while the game has given the aim a target (with none, the axis is the animation's)
        calls = [e for e in events if e['type'] == 'set_aim' and e['name'] == aim['name']]
        aimed = np.array([bool([e for e in calls if e['time'] <= t+1e-9]) and
                          [e for e in calls if e['time'] <= t+1e-9][-1]['point'] is not None for t in times[1:]])
        turn = np.degrees(np.arccos(np.clip((in_frame[1:]*in_frame[:-1]).sum(axis=1), -1, 1)))-aim['degrees_per_s']*dt
        turn = np.where(aimed, turn, -np.inf) if len(turn) else turn
        if len(turn) and aimed.any():
            violations.append((float(max(0.0, turn.max())), float(times[int(turn.argmax())+1])))
        aims[aim['name']] = (axes, frames, a0)
    if 'limit_violation_deg_max' in crit:
        gates['limits_deg'] = gate(violations, crit['limit_violation_deg_max'])
    # tracking: after settle_s, an aim at a target within its limits and a limb at a target within its reach
    settle = crit.get('settle_s', 0.0)
    aim_errors, limb_errors = [], []
    for e_index, e in enumerate(events):
        if e['type'] not in ('set_aim', 'set_limb_target') or e['point'] is None:
            continue
        until = next((x['time'] for x in events[e_index+1:] if x['type'] == e['type'] and x['name'] == e['name']), math.inf)
        target = np.asarray(e['point'], float)
        for k, t in enumerate(times):
            if not e['time']+settle <= t < until:
                continue
            if e['type'] == 'set_aim':
                aim = next(a for a in runtime['aims'] if a['name'] == e['name'])
                axes, frames, a0 = aims[e['name']]
                to = gm_unit(target-pos[k, index[aim['joints'][-1]]])
                yaw, pitch = gs.aim_angles(frames[k].T @ to, a0)
                if aim['yaw_degrees'][0] <= yaw <= aim['yaw_degrees'][1] and aim['pitch_degrees'][0] <= pitch <= aim['pitch_degrees'][1]:
                    aim_errors.append((math.degrees(math.acos(float(np.clip(gm_unit(axes[k]) @ to, -1, 1)))), float(t)))
            elif e['weight'] == 1:
                limb = next(l for l in runtime['limbs'] if l['name'] == e['name'])
                if reachable(limb, pos[k, index[limb['joints'][0]]], target):
                    limb_errors.append((float(np.linalg.norm(pos[k, index[limb['joints'][-1]]]-target))*1000, float(t)))
    if 'aim_error_deg_max' in crit:
        gates['aim_error_deg'] = gate(aim_errors, crit['aim_error_deg_max'])
    if 'limb_target_error_mm_max' in crit:
        gates['limb_target_error_mm'] = gate(limb_errors, crit['limb_target_error_mm_max'])
    # recovery: once a patch's grace time is over, penetration and contact gap stay within limits
    if patches:
        values = []
        for p_index, p in enumerate(patches):
            until = patches[p_index+1]['time'] if p_index+1 < len(patches) else math.inf
            for k, (r, t) in enumerate(zip(rows, times)):
                if p['time']+grace <= t < until:
                    worst = r['penetration_m']*1000
                    for n in planted:
                        if planted[n][k]:
                            worst = max(worst, abs(r['legs'][n])*1000)
                    values.append((worst, float(t)))
        gates['recovery_mm'] = gate(values, crit['recovery_mm_max'])
        readings['transient_penetration_mm'] = gm.r(max((r['penetration_m']*1000 for r, g in zip(rows, in_grace) if g), default=0.0))
    if playback:
        gates['playback_mm'] = gate([(r['playback_m']*1000, r['time']) for r in rows], crit['playback_mm_max'])
    readings['outside_terrain_samples'] = outside
    worst = max(rows, key=lambda r: r['penetration_m'])
    readings['deepest_penetration'] = {'time_s': gm.r(worst['time']), 'mm': gm.r(worst['penetration_m']*1000),
                                       **worst.get('deepest', {})}
    passed = all(g['passed'] for g in gates.values())
    return {'version': 1, 'passed': passed, 'gates': gates, 'readings': readings,
            'issues': [f'{k}: {g["value"]:.6g} at {g["worst_time_s"]} s, needs {g["direction"]} {g["limit"]:g}'
                       for k, g in gates.items() if not g['passed']],
            'runtime': trace['runtime'], 'hashes': hashes, 'frames': len(times)}


def gm_unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v/n if n > 1e-12 else v


def reachable(limb, root, target):
    """Whether a target is within the limb's reach from its first joint (for two_bone, within the reach its bend
    limits allow)."""
    d = float(np.linalg.norm(target-root))
    lengths = limb['segment_lengths_m']
    if limb['solver'] == 'two_bone':
        l1, l2 = lengths
        far = lambda angle: math.sqrt(l1*l1+l2*l2-2*l1*l2*math.cos(math.radians(angle)))
        return far(limb['bend_degrees'][0]) <= d <= far(limb['bend_degrees'][1])
    return d <= sum(lengths)
