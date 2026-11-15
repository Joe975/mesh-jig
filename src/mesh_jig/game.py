"""The game-ready character stage: the `character.json` v1 contract, `game-info` and `game-build`.

Coordinates in the contract are Blender metres (+Z up, +Y forward); the runtime file is in glTF axes. See
docs/GAME_READY.md. Unknown keys are refused at every level. Every block but `version`, `rig_sha256` and `graph` is
optional, and every limit a gate uses is a key of `criteria`, required when its block is present.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from . import game_measure as gm
from . import terrain

VERSION = 1
TOP = {'version', 'rig_sha256', 'forward', 'parameters', 'graph', 'body', 'limbs', 'ground', 'aims', 'sockets',
       'criteria'}
SOLVERS = {'two_bone': {'pole', 'bend_degrees'}, 'chain': {'bend_degrees_max'}, 'none': set()}
# gate -> (criterion or None for a count that must be zero, what it reads, in words)
GATES = {
    'loop_seam_mm': ('loop_seam_mm_max', 'each looping clip: the largest vertex distance between its first and last pose'),
    'root_travel_mm': ('root_travel_mm_max', 'each locomotion clip (a blend1d point): how far the skeleton root and '
                       'body joint move horizontally from start to end. Locomotion clips play in place'),
    'legs_planted': (None, 'each locomotion clip: legs whose contact never comes within the contact gap of the '
                     "rest ground for at least stance_s_min (must be none)"),
    'skate_mps': ('skate_mps_max', 'each leg in each locomotion clip: RMS of the planted contact velocity\'s '
                  "horizontal difference from the clip's travel speed backward. A wide spread is skating once the "
                  'game moves the root'),
    'points_in_order': (None, 'each blend1d: point values (measured ones filled in) that are missing, out of the '
                        "parameter's range or not strictly increasing (must be none)"),
    'gait_phase_difference': ('gait_phase_difference_max', 'each synced blend1d: the largest difference, as a share '
                              'of the cycle, between two neighbouring clips in when each leg touches down after the '
                              'reference leg'),
    'blend_slip_mps': ('blend_slip_mps_max', 'each synced blend1d: planted slip when two neighbouring clips are '
                       'blended half and half at the mean of their travel speeds'),
    'fade_joint_speed_mps': ('fade_joint_speed_mps_max', 'each state transition and one-shot fade: the largest joint '
                             'distance between the poses at its two ends over the fade time'),
}


def _keys(d, required, optional=(), where='contract'):
    if not isinstance(d, dict):
        raise ValueError(f'{where} must be an object')
    unknown, missing = set(d)-set(required)-set(optional), set(required)-set(d)
    if unknown:
        raise ValueError(f'{where}: unknown key(s) {sorted(unknown)}')
    if missing:
        raise ValueError(f'{where}: missing {sorted(missing)}')


def _name(v, where):
    if not isinstance(v, str) or not v:
        raise ValueError(f'{where} must be a nonempty string')
    return v


def _number(v, where, lo=-math.inf, hi=math.inf, positive=False):
    v = terrain.finite(v, where, positive=positive)
    if not lo <= v <= hi:
        raise ValueError(f'{where} must be from {lo:g} to {hi:g}')
    return v


def _vector(v, where, nonzero=False):
    a = np.asarray(v, float) if isinstance(v, list) and all(type(x) in (int, float) for x in v) else None
    if a is None or a.shape != (3,) or not np.isfinite(a).all() or (nonzero and np.linalg.norm(a) < 1e-9):
        raise ValueError(f'{where} must be a finite{" nonzero" if nonzero else ""} three-vector')
    return a


def _range(v, where, lo, hi):
    if not isinstance(v, list) or len(v) != 2:
        raise ValueError(f'{where} must be [low, high]')
    a, b = (_number(x, where, lo, hi) for x in v)
    if a >= b:
        raise ValueError(f'{where}: low must be below high')


def _names(v, where, minimum=1):
    if not isinstance(v, list) or len(v) < minimum or len(set(map(str, v))) != len(v):
        raise ValueError(f'{where} must list at least {minimum} distinct names')
    return [_name(x, where) for x in v]


def walk(node, where, found, parameters):
    """Check one graph node and its children; collect what the graph names in `found`."""
    if not isinstance(node, dict) or not isinstance(node.get('type'), str):
        raise ValueError(f'{where}: a node needs a type')
    kind = node['type']
    if kind == 'clip':
        _keys(node, {'type', 'clip'}, where=where)
        found['clips'].add(_name(node['clip'], where+'.clip'))
    elif kind == 'blend1d':
        _keys(node, {'type', 'parameter', 'points'}, {'sync'}, where)
        if node['parameter'] not in parameters:
            raise ValueError(f'{where}: parameter {node["parameter"]!r} is not declared in parameters')
        if not isinstance(node.get('sync', False), bool):
            raise ValueError(f'{where}.sync must be true or false')
        if not isinstance(node['points'], list) or len(node['points']) < 2:
            raise ValueError(f'{where}: a blend1d needs at least two points')
        for k, p in enumerate(node['points']):
            at = f'{where}.points[{k}]'
            _keys(p, {'value'}, {'clip', 'node'}, at)
            if ('clip' in p) == ('node' in p):
                raise ValueError(f'{at}: give exactly one of clip or node')
            if p['value'] != 'measured':
                _number(p['value'], at+'.value')
            if 'node' in p:
                if node.get('sync') or p['value'] == 'measured':
                    raise ValueError(f'{at}: a synced blend and a measured value need a clip, not a node')
                walk(p['node'], at+'.node', found, parameters)
            else:
                found['clips'].add(_name(p['clip'], at+'.clip'))
                found['locomotion'].add(p['clip'])
                found['measured'] |= p['value'] == 'measured'
        found['blends'].append((where, node))
    elif kind == 'states':
        _keys(node, {'type', 'start', 'states', 'transitions'}, where=where)
        if not isinstance(node['states'], dict) or not node['states'] or node['start'] not in node['states']:
            raise ValueError(f'{where}: states needs named states and a start among them')
        for name, child in node['states'].items():
            _name(name, where+'.states')
            if child is not None:
                walk(child, f'{where}.states.{name}', found, parameters)
        if not isinstance(node['transitions'], list):
            raise ValueError(f'{where}.transitions must be a list')
        for k, tr in enumerate(node['transitions']):
            at = f'{where}.transitions[{k}]'
            _keys(tr, {'from', 'to', 'fade_s'}, {'on', 'at_end'}, at)
            if tr['from'] not in node['states'] or tr['to'] not in node['states'] or tr['from'] == tr['to']:
                raise ValueError(f'{at}: from and to must be two different states')
            if ('on' in tr) == ('at_end' in tr):
                raise ValueError(f'{at}: give exactly one of on (a trigger name) or at_end')
            if 'on' in tr:
                found['triggers'].add(_name(tr['on'], at+'.on'))
            elif tr['at_end'] is not True or (node['states'][tr['from']] or {}).get('type') != 'clip':
                raise ValueError(f'{at}: at_end must be true and leave a state that plays one clip')
            _number(tr['fade_s'], at+'.fade_s', positive=True)
            found['fades'] = True
    elif kind == 'oneshots':
        _keys(node, {'type', 'clips'}, where=where)
        if not isinstance(node['clips'], dict) or not node['clips']:
            raise ValueError(f'{where}: oneshots needs clips')
        for name, f in node['clips'].items():
            _keys(f, {'fade_in_s', 'fade_out_s'}, where=f'{where}.clips.{name}')
            for key in f:
                _number(f[key], f'{where}.clips.{name}.{key}', positive=True)
            found['clips'].add(_name(name, where+'.clips'))
            found['oneshots'].add(name)
            found['fades'] = True
    else:
        raise ValueError(f'{where}: unknown node type {kind!r} (clip, blend1d, states, oneshots)')


def graph(spec):
    """What the graph names: clips, locomotion clips, blends, triggers, one-shots, masks."""
    g = spec['graph']
    _keys(g, {'layers'}, where='graph')
    if not isinstance(g['layers'], list) or not g['layers']:
        raise ValueError('graph needs at least one layer')
    found = {'clips': set(), 'locomotion': set(), 'blends': [], 'triggers': set(), 'oneshots': set(),
             'fades': False, 'measured': False, 'masks': set()}
    names = set()
    for k, layer in enumerate(g['layers']):
        at = f'graph.layers[{k}]'
        _keys(layer, {'name', 'node'}, {'mask'}, at)
        if _name(layer['name'], at+'.name') in names:
            raise ValueError(f'{at}: layer names must be unique')
        names.add(layer['name'])
        if 'mask' in layer:
            found['masks'] |= set(_names(layer['mask'], at+'.mask'))
        walk(layer['node'], f'{at}.node', found, spec.get('parameters', {}))
    return found


def needed_criteria(spec, found):
    legs = [l for l in spec.get('limbs', []) if l.get('role') == 'leg']
    need = {'loop_seam_mm_max'}
    if legs:
        need |= {'contact_gap_mm_max', 'stance_s_min'}
    if found['blends']:
        need.add('root_travel_mm_max')
        if legs:
            need.add('skate_mps_max')
            if any(n.get('sync') for _, n in found['blends']):
                need |= {'gait_phase_difference_max', 'blend_slip_mps_max'}
    if found['fades']:
        need.add('fade_joint_speed_mps_max')
    return need


def contract(spec):
    try:
        return _contract(spec)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError(f'invalid character contract: {error!r}') from error


def _contract(spec):
    _keys(spec, {'version', 'rig_sha256', 'graph'}, TOP-{'version', 'rig_sha256', 'graph'})
    if type(spec['version']) is not int or spec['version'] != VERSION:
        raise ValueError(f'character contract version must be {VERSION}')
    if not isinstance(spec['rig_sha256'], str) or not re.fullmatch('[0-9a-f]{64}', spec['rig_sha256']):
        raise ValueError('rig_sha256 must be the 64 lowercase hex digits of the GLB SHA-256')
    if 'forward' in spec:
        f = _vector(spec['forward'], 'forward', nonzero=True)
        if abs(f[2]) > 1e-9:
            raise ValueError('forward must be horizontal (z = 0)')
    params = spec.get('parameters', {})
    if not isinstance(params, dict):
        raise ValueError('parameters must be an object')
    for name, p in params.items():
        _keys(p, {'min', 'max', 'default'}, where=f'parameters.{name}')
        lo, hi = _number(p['min'], name+'.min'), _number(p['max'], name+'.max')
        if lo >= hi or not lo <= _number(p['default'], name+'.default') <= hi:
            raise ValueError(f'parameters.{name}: needs min < max and default between them')
    found = graph(spec)
    if 'body' in spec:
        _keys(spec['body'], {'joint'}, where='body')
        _name(spec['body']['joint'], 'body.joint')
    limbs = spec.get('limbs', [])
    if not isinstance(limbs, list):
        raise ValueError('limbs must be a list')
    seen = set()
    for k, limb in enumerate(limbs):
        at = f'limbs[{k}]'
        if not isinstance(limb, dict) or limb.get('solver') not in SOLVERS:
            raise ValueError(f'{at}: solver must be one of {sorted(SOLVERS)}')
        _keys(limb, {'name', 'role', 'joints', 'solver'} | SOLVERS[limb['solver']], {'contact'}, at)
        if _name(limb['name'], at+'.name') in seen:
            raise ValueError(f'{at}: limb names must be unique')
        seen.add(limb['name'])
        _name(limb['role'], at+'.role')
        joints = _names(limb['joints'], at+'.joints', 2)
        if limb['solver'] == 'two_bone':
            if len(joints) != 3:
                raise ValueError(f'{at}: a two_bone limb has exactly three joints')
            _vector(limb['pole'], at+'.pole', nonzero=True)
            _range(limb['bend_degrees'], at+'.bend_degrees', 0, 180)
        if limb['solver'] == 'chain':
            _number(limb['bend_degrees_max'], at+'.bend_degrees_max', 0, 180, positive=True)
        if (limb['role'] == 'leg') != ('contact' in limb):
            raise ValueError(f'{at}: a leg needs a contact, and only a leg has one')
        if 'contact' in limb:
            terrain.region(limb['contact'])
    legs = [l for l in limbs if l['role'] == 'leg']
    if found['measured'] and not legs:
        raise ValueError('a "measured" point value needs legs: travel speed is read from planted contacts')
    if 'ground' in spec:
        g = spec['ground']
        _keys(g, {'drop_m_max', 'rise_m_max', 'tilt', 'relock_m', 'blend_s', 'lookahead_s'}, where='ground')
        for key in ('drop_m_max', 'rise_m_max', 'blend_s', 'lookahead_s'):
            _number(g[key], 'ground.'+key, 0)
        _number(g['relock_m'], 'ground.relock_m', positive=True)
        if not isinstance(g['tilt'], bool):
            raise ValueError('ground.tilt must be true or false')
        if not legs or 'body' not in spec:
            raise ValueError('ground adaptation needs legs and a body joint')
        if g['tilt'] and len(legs) < 3:
            raise ValueError('ground.tilt needs three or more legs')
    for key, required in (('aims', {'name', 'joints', 'axis', 'yaw_degrees', 'pitch_degrees', 'degrees_per_s'}),
                          ('sockets', {'name', 'joint', 'offset'})):
        items, names = spec.get(key, []), set()
        if not isinstance(items, list):
            raise ValueError(f'{key} must be a list')
        for k, item in enumerate(items):
            at = f'{key}[{k}]'
            _keys(item, required, where=at)
            if _name(item['name'], at+'.name') in names:
                raise ValueError(f'{at}: names must be unique')
            names.add(item['name'])
            if key == 'aims':
                _names(item['joints'], at+'.joints')
                _vector(item['axis'], at+'.axis', nonzero=True)
                _range(item['yaw_degrees'], at+'.yaw_degrees', -180, 180)
                _range(item['pitch_degrees'], at+'.pitch_degrees', -90, 90)
                _number(item['degrees_per_s'], at+'.degrees_per_s', positive=True)
            else:
                _name(item['joint'], at+'.joint')
                _vector(item['offset'], at+'.offset')
    need, criteria = needed_criteria(spec, found), spec.get('criteria', {})
    if not isinstance(criteria, dict) or set(criteria) != need:
        raise ValueError('criteria must state exactly the limits this contract\'s gates use: '+', '.join(sorted(need)))
    for key, value in criteria.items():
        _number(value, 'criteria.'+key, positive=True)
    if 'gait_phase_difference_max' in criteria and criteria['gait_phase_difference_max'] > .5:
        raise ValueError('gait_phase_difference_max is a share of the cycle, at most 0.5')
    return found


def resolve(spec, body):
    """Every name against the GLB; refuse what does not resolve. Returns what the measurement needs."""
    found = contract(spec)
    joints = set(body.names)
    named = [spec['body']['joint']] if 'body' in spec else []
    for l in spec.get('limbs', []):
        named += l['joints']
    for a in spec.get('aims', []):
        named += a['joints']
    named += [s['joint'] for s in spec.get('sockets', [])] + sorted(found['masks'])
    missing = sorted(set(named)-joints)
    if missing:
        raise ValueError(f'joints the GLB does not have: {missing}; game-info lists its joints')
    for kind, items in (('limb', spec.get('limbs', [])), ('aim', spec.get('aims', []))):
        for item in items:
            for a, b in zip(item['joints'], item['joints'][1:]):
                if body.parent[b] != a:
                    raise ValueError(f'{kind} {item["name"]}: {b} is not a direct child of {a}')
    used = [j for l in spec.get('limbs', []) for j in l['joints']]
    if len(used) != len(set(used)):
        raise ValueError('limbs must not share joints')
    absent = sorted(found['clips']-set(body.clips))
    if absent:
        raise ValueError(f'the graph names clips the GLB does not have: {absent}')
    loops = {}
    for c in sorted(found['clips']):
        declared = body.clips[c].declared_loop
        loops[c] = declared if declared is not None else c in found['locomotion']
    for _, node in found['blends']:
        if node.get('sync'):
            for p in node['points']:
                if not loops[p['clip']]:
                    raise ValueError(f'clip {p["clip"]} is in a synced blend but the GLB says it plays once')
    legs, occupied = {}, set()
    for l in spec.get('limbs', []):
        if l['role'] != 'leg':
            continue
        lo, hi = np.asarray(l['contact']['box'], float)
        hit = (body.vnode == l['contact']['node']) & ((body.rest >= lo-1e-7) & (body.rest <= hi+1e-7)).all(axis=1)
        ids = np.flatnonzero(hit)
        if not len(ids):
            raise ValueError(f'limb {l["name"]}: its contact selects no vertices; game-info shows nodes and boxes')
        if occupied.intersection(ids.tolist()):
            raise ValueError(f'limb {l["name"]}: contacts overlap; each vertex belongs to one contact')
        occupied.update(ids.tolist())
        legs[l['name']] = ids
    roots = [n for n in body.names if body.parent[n] is None]
    if 'body' in spec and spec['body']['joint'] not in roots:
        roots.append(spec['body']['joint'])
    f = np.asarray(spec.get('forward', [0, 1, 0]), float)[:2]
    masks = [None if 'mask' not in layer else sorted(set().union(*(body.descendants(j) for j in layer['mask'])))
             for layer in spec['graph']['layers']]
    return {'found': found, 'loops': loops, 'legs': legs, 'roots': roots, 'forward': f/np.linalg.norm(f),
            'masks': masks, 'named': sorted(set(named))}


def _v(p, places=3):
    return '(' + ', '.join(f'{x:+.{places}f}' for x in p) + ')'


def info(rig, spec=None):
    """What an author needs to write character.json, as text."""
    body = gm.Body(rig)
    resolved = resolve(spec, body) if spec is not None else None
    gap = (spec['criteria'].get('contact_gap_mm_max', gm.TOUCH_MM) if spec else gm.TOUCH_MM)/1000
    lines = ['GAME-READY CHARACTER (mesh-jig game-info)',
             'Write character.json in Blender metres: +Z up, +Y forward, the model\'s own left is X<0. Joint names are '
             'skin joints; a contact names a mesh node and a box at rest. The package\'s runtime file is in glTF axes: '
             'Blender (x, y, z) becomes (x, z, -y).',
             f'rig_sha256: {terrain.digest(rig)}', '',
             f'JOINTS ({len(body.names)}): name, parent, rest position']
    rest = body.joints(body.rest_world)
    for n, p in zip(body.names, rest):
        lines.append(f'  {n}  parent {body.parent[n] or "-"}  at {_v(p)}')
    nodes = sorted(set(body.vnode.tolist()))
    lines += ['', f'MESH NODES ({len(nodes)}): name, vertices, rest box lo..hi, the joint that carries most of it']
    for n in nodes[:60]:
        m = body.vnode == n
        carried = [str(j) for j in body.vjoint[m]]
        lines.append(f'  {n}: {int(m.sum())} vertices, {_v(body.rest[m].min(axis=0))}..{_v(body.rest[m].max(axis=0))}, '
                     f'{max(sorted(set(carried)), key=carried.count)}')
    if len(nodes) > 60:
        lines.append(f'  ...and {len(nodes)-60} more')
    lines += ['', 'CLIPS: name, duration, loop (from the GLB\'s extras; a clip with none loops when it is a blend1d point)']
    for name, c in sorted(body.clips.items()):
        loop = {True: 'loops', False: 'plays once', None: 'loop not declared'}[c.declared_loop]
        lines.append(f'  {name}: {c.duration:g} s, {loop}' + (f', events {c.events}' if c.events else ''))
    lines += ['', f'GROUND CONTACT BY CLIP: vertex clusters within {gap*1000:g} mm of the ground (the rest pose\'s '
              'lowest vertex), grouped by the joint that carries them. A cluster planted part of each cycle is a foot: its '
              'node and box are a contact. Box in Blender metres at rest.']
    bends = {}
    for name, c in sorted(body.clips.items()):
        loop = c.declared_loop if c.declared_loop is not None else True
        s = gm.clip_sample(body, name, loop, {})
        bends[name] = gm.bend_ranges(body, s.joints, gm.triples(body))
        touch = gm.touching(body, s, gap)
        lines.append(f'  {name} (ground z {s.ground:+.3f}): {len(touch)} cluster{"s" if len(touch) != 1 else ""}')
        for t in touch[:24]:
            lines.append(f'    {t["joint"]} on {", ".join(t["nodes"][:3])}{"..." if len(t["nodes"]) > 3 else ""}: '
                         f'{t["vertices"]} vertices, box {t["box"]}, on the ground {t["share"]*100:.0f}% of the clip, '
                         f'touchdowns {t["touchdowns_s"]}')
        if len(touch) > 24:
            lines.append(f'    ...and {len(touch)-24} more')
    declared = {}
    for l in (spec or {}).get('limbs', []):
        for a, b, c in zip(l['joints'], l['joints'][1:], l['joints'][2:]):
            declared[(a, b, c)] = (f'{l["name"]}: bend_degrees {l["bend_degrees"]}' if l['solver'] == 'two_bone' else
                                   f'{l["name"]}: bend_degrees_max {l["bend_degrees_max"]} (inner angle at least '
                                   f'{180-l["bend_degrees_max"]:g})' if l['solver'] == 'chain' else f'{l["name"]}')
    lines += ['', 'BEND BY CLIP: the inner angle at each joint between its parent and a child, lowest..highest in each '
              'clip, degrees (180 is straight). A two_bone limb\'s bend_degrees bounds this angle at its middle joint; a '
              'chain\'s bend is 180 minus it, at each inner joint. Read the limits from here rather than guessing.']
    for t in gm.triples(body):
        ranges = ', '.join(f'{clip} {bends[clip][t][0]:.0f}..{bends[clip][t][1]:.0f}' for clip in sorted(bends))
        lines.append(f'  {t[0]} > {t[1]} > {t[2]}: {ranges}' + (f'  [{declared[t]}]' if t in declared else ''))
    lines += ['', 'GATES (each limit is a criteria key; the build fails on any; readings never fail it):']
    for gate, (criterion, words) in GATES.items():
        lines.append(f'  {gate} ({criterion or "must be zero"}): {words}')
    lines += ['  contact_gap_mm_max and stance_s_min are no gates: a contact vertex within the gap of the rest '
              'ground is planted, and a touch shorter than stance_s_min is a graze, not a stance.',
              'READINGS: joints no block names; vertices that touch the ground in a locomotion clip and belong to no '
              'contact (a missed leg); limbs whose chain never bends; clips whose own bend leaves a limb\'s declared '
              'range; grazes; how far a clip goes below the rest ground.']
    if resolved is not None:
        lines += ['', 'THIS CONTRACT resolves: clips ' + ', '.join(sorted(resolved['found']['clips'])),
                  '  locomotion clips: ' + (', '.join(sorted(resolved['found']['locomotion'])) or 'none'),
                  '  contacts: ' + (', '.join(f'{k} {len(v)} vertices' for k, v in resolved['legs'].items()) or 'none'),
                  '  criteria: ' + json.dumps(spec['criteria'], sort_keys=True)]
    return '\n'.join(lines)


def command(a):
    spec = json.loads(a.contract.read_text(encoding='utf-8')) if a.contract else None
    if a.cmd == 'game-info':
        print(info(a.rig, spec))
        return 0
    from . import game_package
    report = game_package.build(a.rig, spec, a.out)
    print(json.dumps({'passed': report['passed'], 'issues': report['issues'],
                      'gates': {k: {x: g[x] for x in ('value', 'limit', 'passed')} for k, g in report['gates'].items()}},
                     indent=2))
    return 0 if report['passed'] else 1


def play_command(a):
    """game-run, game-check and game-preview."""
    from . import game_scenario
    scenario, scenario_sha256 = game_scenario.load(a.scenario)
    if a.cmd == 'game-run':
        from . import game_runtime
        if a.out.exists():
            raise ValueError('the trace file must be new')
        if a.runtime == 'godot':
            from . import game_godot
            game_godot.run(a.package, a.scenario, a.out)
            print(f'{len(game_scenario.frame_times(scenario))} frames to {a.out.name} (Godot)')
            return 0
        trace = game_runtime.run(a.package, scenario, scenario_sha256)
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(trace, separators=(',', ':'), sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
        print(f'{len(trace["frames"])} frames to {a.out.name}')
        return 0
    trace = json.loads(a.trace.read_text(encoding='utf-8'))
    if a.cmd == 'game-preview':
        from . import game_look
        print(game_look.preview(a.package, scenario, trace, a.out, phases=a.phases, size=a.size))
        return 0
    from . import game_check
    report = game_check.check(a.package, scenario, trace, scenario_sha256)
    if a.report:
        a.report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'issues': report['issues'],
                      'gates': {k: {x: g[x] for x in ('value', 'limit', 'passed')} for k, g in report['gates'].items()}},
                     indent=2))
    return 0 if report['passed'] else 1


def add_commands(subs):
    for name, words in (('game-run', 'play a scenario with the reference runtime and write its trace'),
                        ('game-check', 'hold a trace to the gates, independently of the runtime that made it'),
                        ('game-preview', 'film strip and GIF of a trace on the scenario\'s heightfield')):
        p = subs.add_parser(name, help=words)
        p.add_argument('--package', type=Path, required=True)
        p.add_argument('--scenario', type=Path, required=True)
        if name != 'game-run':
            p.add_argument('--trace', type=Path, required=True)
        if name == 'game-check':
            p.add_argument('--report', type=Path)
        else:
            p.add_argument('--out', type=Path, required=True)
        if name == 'game-run':
            p.add_argument('--runtime', choices=('reference', 'godot'), default='reference',
                           help='godot: the Godot addon, headless (Godot from $MESH_JIG_GODOT or PATH)')
        if name == 'game-preview':
            p.add_argument('--phases', type=int, default=8)
            p.add_argument('--size', type=int, default=360)
        p.set_defaults(fn=play_command)
    p = subs.add_parser('game-info', help='what a game-ready character contract needs: joints, nodes, clips, ground contact')
    p.add_argument('--rig', type=Path, required=True)
    p.add_argument('--contract', type=Path)
    p.set_defaults(fn=command)
    p = subs.add_parser('game-build', help='measure an animated GLB against character.json and write a game package')
    p.add_argument('--rig', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.set_defaults(fn=command)
