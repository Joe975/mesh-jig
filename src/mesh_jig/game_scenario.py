"""Scenarios and traces for a game-ready package, shared by the reference runtime and the independent checker.

A scenario stands in for the game: a heightfield, a dt sequence, the root's world placement over time and a
timeline of calls. Everything is glTF axes (Y up, metres, right-handed). A trace is what a runtime did: per frame
the time, the root's world matrix and every joint's world matrix. Matrices are 4x4, written as 16 numbers in
glTF's column-major order (the translation is entries 12, 13, 14). See docs/GAME_READY.md.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from . import terrain

VERSION = 1
EVENTS = {'set_parameter': {'name', 'value'}, 'trigger': {'name'}, 'play': {'name'}, 'set_aim': {'name', 'point'},
          'set_limb_target': {'name', 'point', 'weight'}, 'set_ground_enabled': {'enabled'},
          'terrain_patch': {'rect', 'offset'}}
TRACE_FORMAT = 'mesh-jig game trace'
LAYOUT = ('4x4 world matrices as 16 numbers in column-major order (glTF), so entries 12, 13, 14 are the translation; '
          'glTF axes (Y up, right-handed), metres')


def aim_angles(v, rest_axis):
    """Yaw and pitch, degrees, of direction v relative to an aim's axis at rest, both in the frame of the aim's
    first joint's parent: yaw turns about that frame's +Y (right-handed) from the rest axis's horizontal direction;
    pitch is v's elevation above that frame's horizontal plane less the rest axis's."""
    v, a = np.asarray(v, float), np.asarray(rest_axis, float)
    vh, ah = np.array([v[0], 0, v[2]]), np.array([a[0], 0, a[2]])
    yaw = math.degrees(math.atan2(float(np.cross(ah, vh)[1]), float(ah @ vh)))
    pitch = math.degrees(math.asin(float(np.clip(v[1], -1, 1)))-math.asin(float(np.clip(a[1], -1, 1))))
    return yaw, pitch


def aim_direction(yaw, pitch, rest_axis):
    """The unit direction with these aim angles: the inverse of aim_angles."""
    a = np.asarray(rest_axis, float)
    ah = np.array([a[0], 0, a[2]])
    ah = ah/np.linalg.norm(ah) if np.linalg.norm(ah) > 1e-12 else np.array([0, 0, -1.0])
    y = math.radians(yaw)
    h = np.array([math.cos(y)*ah[0]+math.sin(y)*ah[2], 0, -math.sin(y)*ah[0]+math.cos(y)*ah[2]])
    e = math.asin(float(np.clip(a[1], -1, 1)))+math.radians(pitch)
    return h*math.cos(e)+np.array([0, 1.0, 0])*math.sin(e)


def _keys(d, required, optional=(), where='scenario'):
    if not isinstance(d, dict):
        raise ValueError(f'{where} must be an object')
    unknown, missing = set(d)-set(required)-set(optional), set(required)-set(d)
    if unknown:
        raise ValueError(f'{where}: unknown key(s) {sorted(unknown)}')
    if missing:
        raise ValueError(f'{where}: missing {sorted(missing)}')


def _point(v, where):
    a = np.asarray(v, float) if isinstance(v, list) and all(type(x) in (int, float) for x in v) else None
    if a is None or a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f'{where} must be a finite [x, y, z] in glTF axes, or null')
    return a


def contract(spec):
    """The scenario's shape, with no package: unknown keys refused, every number finite."""
    try:
        _contract(spec)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError(f'invalid scenario: {error!r}') from error


def _contract(spec):
    _keys(spec, {'version', 'terrain', 'dt', 'duration', 'root', 'events', 'criteria'}, {'expect'})
    if type(spec['version']) is not int or spec['version'] != VERSION:
        raise ValueError(f'scenario version must be {VERSION}')
    field = spec['terrain']
    _keys(field, {'x', 'z', 'heights'}, where='terrain')
    for key in ('x', 'z'):
        axis = np.asarray(field[key], float)
        if axis.ndim != 1 or not 2 <= len(axis) <= 257 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError(f'terrain {key} must be 2..257 finite increasing coordinates')
    h = np.asarray(field['heights'], float)
    if h.shape != (len(field['z']), len(field['x'])) or not np.isfinite(h).all():
        raise ValueError('terrain heights must be finite, a row for each z and a column for each x')
    dt = spec['dt']
    if not isinstance(dt, list) or not 1 <= len(dt) <= 64:
        raise ValueError('dt must list 1..64 frame steps (repeated in turn)')
    for d in dt:
        terrain.finite(d, 'dt')
        if not 1e-3 <= d <= .1:
            raise ValueError('each dt must be from 0.001 to 0.1 s')
    duration = terrain.finite(spec['duration'], 'duration', positive=True)
    if duration > 120:
        raise ValueError('duration must be at most 120 s')
    root = np.asarray(spec['root'], float)
    if root.ndim != 2 or root.shape[1] != 5 or not len(root) or not np.isfinite(root).all():
        raise ValueError('root needs [time, x, y, z, yaw_degrees] keys')
    if root[0, 0] != 0 or np.any(np.diff(root[:, 0]) <= 0) or root[-1, 0] < duration:
        raise ValueError('root keys must start at 0, increase, and reach the duration')
    if not isinstance(spec['events'], list):
        raise ValueError('events must be a list')
    previous = 0.0
    for k, e in enumerate(spec['events']):
        at = f'events[{k}]'
        if not isinstance(e, dict) or e.get('type') not in EVENTS:
            raise ValueError(f'{at}: type must be one of {sorted(EVENTS)}')
        _keys(e, {'time', 'type'} | EVENTS[e['type']], where=at)
        t = terrain.finite(e['time'], at+'.time')
        if not previous <= t <= duration:
            raise ValueError(f'{at}: event times must be in order, from 0 to the duration')
        previous = t
        kind = e['type']
        if kind in ('set_parameter', 'trigger', 'play', 'set_aim', 'set_limb_target'):
            if not isinstance(e['name'], str) or not e['name']:
                raise ValueError(f'{at}.name must be a nonempty string')
        if kind == 'set_parameter':
            terrain.finite(e['value'], at+'.value')
        if kind in ('set_aim', 'set_limb_target') and e['point'] is not None:
            _point(e['point'], at+'.point')
        if kind == 'set_limb_target' and not 0 <= terrain.finite(e['weight'], at+'.weight') <= 1:
            raise ValueError(f'{at}.weight must be from 0 to 1')
        if kind == 'set_ground_enabled' and not isinstance(e['enabled'], bool):
            raise ValueError(f'{at}.enabled must be true or false')
        if kind == 'terrain_patch':
            rect = np.asarray(e['rect'], float)
            if rect.shape != (2, 2) or not np.isfinite(rect).all() or np.any(rect[1] <= rect[0]):
                raise ValueError(f'{at}.rect must be [[x0, z0], [x1, z1]] with x0 < x1 and z0 < z1')
            terrain.finite(e['offset'], at+'.offset')
    if 'expect' in spec:
        _keys(spec['expect'], {'clip', 'offset_s'}, where='expect')
        if not isinstance(spec['expect']['clip'], str):
            raise ValueError('expect.clip must name a clip')
        terrain.finite(spec['expect']['offset_s'], 'expect.offset_s')
    if not isinstance(spec['criteria'], dict):
        raise ValueError('criteria must be an object')


def flat(spec):
    h = np.asarray(spec['terrain']['heights'], float)
    return bool(np.ptp(h) == 0 and not any(e['type'] == 'terrain_patch' for e in spec['events']))


def ground_always_off(spec):
    """The ground is on at the start; this is true when an event at time 0 turns it off and none turns it on."""
    toggles = [e for e in spec['events'] if e['type'] == 'set_ground_enabled']
    return bool(toggles) and toggles[0]['time'] == 0 and not toggles[0]['enabled'] and not any(e['enabled'] for e in toggles)


def needed_criteria(spec, runtime):
    """Exactly the limits this scenario's gates use, from what the scenario and the package have."""
    legs = any(l['role'] == 'leg' for l in runtime['limbs'])
    kinds = {e['type'] for e in spec['events']}
    need = {'penetration_mm_max', 'stretch_mm_max', 'joint_speed_mps_max'}
    if legs:
        need |= {'contact_gap_mm_max', 'slip_mps_max', 'support_fraction_min'}
    if 'expect' in spec and ground_always_off(spec) and flat(spec):
        need.add('playback_mm_max')
    if any(l['solver'] != 'none' for l in runtime['limbs']) or runtime['aims']:
        need.add('limit_violation_deg_max')
    if any(e['type'] == 'set_aim' and e['point'] is not None for e in spec['events']):
        need |= {'aim_error_deg_max', 'settle_s'}
    if any(e['type'] == 'set_limb_target' and e['point'] is not None for e in spec['events']):
        need |= {'limb_target_error_mm_max', 'settle_s'}
    if 'terrain_patch' in kinds:
        need |= {'recovery_mm_max', 'grace_s'}
    return need


def resolve(spec, runtime):
    """Every name in the scenario against the package's runtime file; the criteria exactly as needed."""
    contract(spec)
    names = {'set_parameter': set(runtime['parameters']), 'trigger': set(runtime['triggers']),
             'play': set(runtime['oneshots']), 'set_aim': {a['name'] for a in runtime['aims']},
             'set_limb_target': {l['name'] for l in runtime['limbs']}}
    for k, e in enumerate(spec['events']):
        if e['type'] in names and e['name'] not in names[e['type']]:
            raise ValueError(f'events[{k}]: the package has no {e["type"].replace("set_", "")} named {e["name"]!r}')
    if 'expect' in spec and spec['expect']['clip'] not in runtime['clips']:
        raise ValueError(f'expect.clip {spec["expect"]["clip"]!r} is not one of the package\'s clips')
    need = needed_criteria(spec, runtime)
    if set(spec['criteria']) != need:
        raise ValueError('scenario criteria must state exactly the limits its gates use: '+', '.join(sorted(need)))
    for key, value in spec['criteria'].items():
        terrain.finite(value, 'criteria.'+key, positive=True)
    if 'support_fraction_min' in need and spec['criteria']['support_fraction_min'] > 1:
        raise ValueError('support_fraction_min is a fraction, at most 1')


class Ground:
    """The scenario's heightfield, each grid cell split along its (x0, z0)-(x1, z1) diagonal, with terrain patches
    applied from their time on. `snapshot()` is the ground(x, z) a runtime is given for one frame."""

    def __init__(self, spec):
        self.x = np.asarray(spec['terrain']['x'], float)
        self.z = np.asarray(spec['terrain']['z'], float)
        self.heights = np.asarray(spec['terrain']['heights'], float).copy()

    def patch(self, rect, offset):
        """Raise (or lower) the grid points inside the rectangle by offset."""
        (x0, z0), (x1, z1) = rect
        inside = ((self.z[:, None] >= z0) & (self.z[:, None] <= z1)) & ((self.x[None] >= x0) & (self.x[None] <= x1))
        self.heights = self.heights+offset*inside

    def height(self, x, z, heights=None):
        """Height at each (x, z); NaN outside the grid."""
        field = {'x': self.x, 'y': self.z, 'heights': self.heights if heights is None else heights}
        return terrain.height(field, np.stack([np.asarray(x, float), np.asarray(z, float)], axis=-1))

    def snapshot(self):
        heights = self.heights.copy()

        xs, zs = self.x, self.z

        def ground(x, z):
            """height, normal (unit, glTF), walkable. Outside the grid: the nearest edge height, not walkable.
            The same triangles as Ground.height, read for one point."""
            xc, zc = min(max(float(x), xs[0]), xs[-1]), min(max(float(z), zs[0]), zs[-1])
            i = min(max(int(np.searchsorted(xs, xc, side='right'))-1, 0), len(xs)-2)
            j = min(max(int(np.searchsorted(zs, zc, side='right'))-1, 0), len(zs)-2)
            sx, sz = xs[i+1]-xs[i], zs[j+1]-zs[j]
            u, v = (xc-xs[i])/sx, (zc-zs[j])/sz
            a, b, c, d = heights[j, i], heights[j, i+1], heights[j+1, i+1], heights[j+1, i]
            if u >= v:
                h, du, dv = a*(1-u)+b*(u-v)+c*v, b-a, c-b
            else:
                h, du, dv = a*(1-v)+c*u+d*(v-u), c-d, d-a
            n = np.array([-du/sx, 1.0, -dv/sz])
            return float(h), n/np.linalg.norm(n), bool(xc == x and zc == z)
        return ground


def frame_times(spec):
    """Frame times: 0, then the dt sequence in turn, a step cut short to land exactly on an event time or the end."""
    marks = sorted({float(e['time']) for e in spec['events']} | {float(spec['duration'])})
    times, t, k = [0.0], 0.0, 0
    end = float(spec['duration'])
    while t < end-1e-12:
        step = float(spec['dt'][k % len(spec['dt'])])
        k += 1
        nxt = t+step
        mark = next(m for m in marks if m > t+1e-12)
        if mark <= nxt+1e-9:
            nxt = mark
        times.append(nxt)
        t = nxt
    return np.array(times)


def root_at(spec, t):
    """The root's world position and yaw (degrees about +Y, right-handed) at time t, linear between keys."""
    keys = np.asarray(spec['root'], float)
    values = [float(np.interp(t, keys[:, 0], keys[:, k])) for k in range(1, 5)]
    return np.array(values[:3]), values[3]


def root_matrix(position, yaw_degrees):
    a = math.radians(yaw_degrees)
    m = np.eye(4)
    m[:3, :3] = [[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]
    m[:3, 3] = position
    return m


def column_major(m):
    return [round(float(x), 7)+0.0 for x in np.asarray(m).T.reshape(16)]


def matrix(values):
    return np.asarray(values, float).reshape(4, 4).T


def digest_json(data):
    """A scenario held in memory has no file: its hash is that of json.dumps(data, sort_keys=True) as UTF-8, which
    is the file hash of a scenario written exactly so."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load(path):
    """A scenario file: (the scenario, the SHA-256 of the file's bytes). Any runtime can reproduce that hash."""
    data = open(path, 'rb').read()
    return json.loads(data.decode('utf-8')), hashlib.sha256(data).hexdigest()


def read_trace(data, joint_names, times):
    """Check a trace's shape against the package and scenario; return (root matrices, joint matrices by frame)."""
    _keys(data, {'format', 'version', 'runtime', 'hashes', 'joints', 'matrix_layout', 'frames'}, where='trace')
    if data['format'] != TRACE_FORMAT or data['version'] != 1:
        raise ValueError(f'trace format must be {TRACE_FORMAT!r} version 1')
    if data['joints'] != list(joint_names):
        raise ValueError('trace joints must be the runtime file\'s skeleton joints, in its order')
    frames = data['frames']
    if not isinstance(frames, list) or len(frames) != len(times):
        raise ValueError(f'trace has {len(frames) if isinstance(frames, list) else 0} frames; the scenario has {len(times)}')
    roots, joints = [], []
    for k, (f, t) in enumerate(zip(frames, times)):
        _keys(f, {'time', 'root', 'joints'}, where=f'trace frame {k}')
        if abs(float(f['time'])-t) > 1e-6:
            raise ValueError(f'trace frame {k} is at {f["time"]} s; the scenario frame is at {t:.9g} s')
        m = np.asarray(f['joints'], float)
        if m.shape != (len(joint_names), 16) or np.asarray(f['root'], float).shape != (16,) or not np.isfinite(m).all():
            raise ValueError(f'trace frame {k}: a root matrix and one 16-number matrix per joint are needed')
        roots.append(matrix(f['root']))
        joints.append(m.reshape(-1, 4, 4).transpose(0, 2, 1))
    return np.array(roots), np.array(joints)
