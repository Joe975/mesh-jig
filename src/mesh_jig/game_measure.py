"""What a game-ready package measures, read from the posed vertices of a skinned, animated GLB with numpy.

Generic for any number of legs. Every time is seconds into a clip; every position here is Blender metres (+Z up,
+Y forward) until `game` writes the runtime file in glTF axes. Deterministic: same file, same numbers.

- A clip is sampled at HZ. A loop is sampled over [0, duration) and wraps; a clip that plays once over [0, duration].
- A leg is planted in a sample when its lowest contact vertex is within the contact gap of the ground (the
  rest pose's lowest vertex, as rig-check's), for at least the shortest stance; a shorter touch is a graze. The
  planted velocity of a sample is the mean horizontal velocity
  of the contact vertices that are within the gap at both ends of the step to the next sample.
- A blend is done on joint local transforms: translations and scales lerp, rotations shortest-arc nlerp, then skin.
"""
from __future__ import annotations

import math

import numpy as np

from . import glb, skin

HZ = 120                      # samples a second of clip
C = np.array([[1., 0, 0], [0, 0, 1], [0, -1, 0]])     # Blender column vector -> glTF: (x, y, z) -> (x, z, -y)
BEND_DEG = 1.0                # reading only: a limb whose every inner angle moves less than this never bends
TOUCH_MM = 5.0                # game-info only, with no contract gap: a vertex this near the rest ground touches it
CLUSTER_SHARE = 0.02          # touching vertices further apart than this share of the model's size are two clusters
PLACES = 6


def r(x):
    return None if x is None else round(float(x), PLACES)+0.0


def to_gltf(v):
    return C @ np.asarray(v, float)


def to_blender(points):
    return np.asarray(points, float) @ C     # rows (x, y, z) glTF -> (x, -z, y)


class Clip:
    def __init__(self, doc, blob, anim):
        self.name, self.channels = anim.get('name'), []
        for ch in anim['channels']:
            target = ch['target']
            if 'node' not in target or target['path'] not in ('translation', 'rotation', 'scale'):
                continue
            s = anim['samplers'][ch['sampler']]
            interpolation = s.get('interpolation', 'LINEAR')
            if interpolation not in ('LINEAR', 'STEP'):
                raise ValueError(f'clip {self.name}: unsupported interpolation {interpolation}')
            times, values = glb.accessor(doc, blob, s['input']).ravel(), glb.accessor(doc, blob, s['output'])
            if not len(times) or np.any(np.diff(times) <= 0) or not np.isfinite(values).all():
                raise ValueError(f'clip {self.name}: animation times must be increasing and values finite')
            self.channels.append((target['node'], target['path'], times, values, interpolation))
        if not self.channels:
            raise ValueError(f'clip {self.name}: no joint channels')
        self.duration = float(max(c[2][-1] for c in self.channels))
        if self.duration <= 0:
            raise ValueError(f'clip {self.name}: zero duration')
        extras = anim.get('extras') if isinstance(anim.get('extras'), dict) else {}
        self.declared_loop = extras['loop'] if isinstance(extras.get('loop'), bool) else None
        self.events = extras['events'] if isinstance(extras.get('events'), list) else []
        self.nodes = sorted({c[0] for c in self.channels})

    def local(self, t):
        """node -> {path: value} at clip time t, as glTF samples it (skin.sample's rule)."""
        out = {}
        for node, path, times, values, interpolation in self.channels:
            k = int(np.clip(np.searchsorted(times, t, side='right')-1, 0, len(times)-1))
            v = values[k]
            if interpolation == 'LINEAR' and k < len(times)-1:
                u = float(np.clip((t-times[k])/(times[k+1]-times[k]), 0, 1))
                v = skin.slerp(values[k], values[k+1], u) if path == 'rotation' else values[k]*(1-u)+values[k+1]*u
            out.setdefault(node, {})[path] = np.asarray(v, float)
        return out


class Body:
    """The GLB read once: skeleton, clips, every triangle vertex with the joints that carry it."""

    def __init__(self, path):
        doc, blob = glb.read_glb(path)
        self.doc = doc
        nodes = doc.get('nodes', [])
        joints = sorted({j for s in doc.get('skins', []) for j in s['joints']})
        if not joints:
            raise ValueError('the GLB has no skin joints')
        names = [nodes[j].get('name') for j in joints]
        if any(not n for n in names) or len(set(names)) != len(names):
            raise ValueError('every skin joint needs a unique name')
        self.parent_node = {c: i for i, n in enumerate(nodes) for c in n.get('children', [])}
        self.names, self.node = names, dict(zip(names, joints))
        self.parent = {n: nodes[self.parent_node[j]]['name'] if self.parent_node.get(j) in joints else None
                       for n, j in self.node.items()}
        self.clips = {}
        for a in doc.get('animations', []):
            if a.get('name'):
                if a['name'] in self.clips:
                    raise ValueError(f'two clips are named {a["name"]}')
                self.clips[a['name']] = Clip(doc, blob, a)
        animated = sorted(set(joints) | {n for c in self.clips.values() for n in c.nodes})
        for i in animated:
            if 'matrix' in nodes[i]:
                raise ValueError(f'node {nodes[i].get("name", i)}: a joint or animated node with a matrix transform; '
                                 'export translation, rotation and scale')
        self.rest_trs = {i: {'translation': np.array(nodes[i].get('translation', [0, 0, 0]), float),
                             'rotation': np.array(nodes[i].get('rotation', [0, 0, 0, 1]), float),
                             'scale': np.array(nodes[i].get('scale', [1, 1, 1]), float)} for i in animated}
        self.rest_world = skin.worlds(doc)
        pair_nodes, pair_inv, skin_base = [], [], {}
        pos, idx, wts, vnode, vjoint, faces, count = [], [], [], [], [], [], 0
        for ni, _ in glb.mesh_nodes(doc):
            node = nodes[ni]
            for prim in doc['meshes'][node['mesh']]['primitives']:
                if prim.get('mode', 4) != 4:
                    continue
                p = glb.accessor(doc, blob, prim['attributes']['POSITION'])
                if 'skin' in node:
                    si = node['skin']
                    sk = doc['skins'][si]
                    if si not in skin_base:
                        inv = (glb.accessor(doc, blob, sk['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
                               if 'inverseBindMatrices' in sk else np.repeat(np.eye(4)[None], len(sk['joints']), axis=0))
                        skin_base[si] = len(pair_nodes)
                        pair_nodes += list(sk['joints'])
                        pair_inv += list(inv)
                    j = glb.accessor(doc, blob, prim['attributes']['JOINTS_0']).astype(int)
                    w = glb.accessor(doc, blob, prim['attributes']['WEIGHTS_0'])
                    idx.append(j+skin_base[si])
                    wts.append(w)
                    strongest = np.asarray(sk['joints'])[j[np.arange(len(j)), w.argmax(axis=1)]]
                    vjoint += [nodes[k]['name'] for k in strongest]
                else:
                    pair_nodes.append(ni)
                    pair_inv.append(np.eye(4))
                    idx.append(np.full((len(p), 4), len(pair_nodes)-1))
                    wts.append(np.tile([1., 0, 0, 0], (len(p), 1)))
                    up = ni
                    while up is not None and up not in joints:
                        up = self.parent_node.get(up)
                    vjoint += [nodes[up]['name'] if up is not None else None]*len(p)
                faces.append((glb.accessor(doc, blob, prim['indices']).astype(int).reshape(-1, 3) if 'indices' in prim
                              else np.arange(len(p)).reshape(-1, 3))+count)
                count += len(p)
                pos.append(p)
                vnode += [node.get('name', f'node{ni}')]*len(p)
        if not pos:
            raise ValueError('the GLB has no triangle meshes')
        self.pair_nodes, self.pair_inv = pair_nodes, np.asarray(pair_inv)
        self.idx, self.w = np.concatenate(idx), np.concatenate(wts)
        self.ph = np.c_[np.concatenate(pos), np.ones(sum(len(p) for p in pos))]
        self.vnode, self.vjoint = np.array(vnode, dtype=object), np.array(vjoint, dtype=object)
        self.faces = np.concatenate(faces)
        self.rest = self.vertices(self.rest_world)
        if not np.isfinite(self.rest).all():
            raise ValueError('the rest pose has nonfinite vertices')

    def vertices(self, world):
        m = np.stack([world[n] for n in self.pair_nodes]) @ self.pair_inv
        return to_blender(np.einsum('vk,vkij,vj->vi', self.w, m[self.idx], self.ph)[:, :3])

    def joints(self, world, names=None):
        return to_blender(np.array([world[self.node[n]][:3, 3] for n in (names or self.names)]))

    def full(self, clip, t):
        """Every animated node's local transform in `clip` at time t (no wrapping)."""
        out = {i: dict(v) for i, v in self.rest_trs.items()}
        for node, parts in self.clips[clip].local(t).items():
            out[node].update(parts)
        return out

    def world(self, trs):
        return skin.worlds(self.doc, {i: glb.node_matrix({k: list(v) for k, v in p.items()}) for i, p in trs.items()})

    def descendants(self, name):
        out, stack = set(), [name]
        while stack:
            n = stack.pop()
            out.add(n)
            stack += [c for c, p in self.parent.items() if p == n]
        return out


def blend(a, b, w):
    """Joint local transforms half way (or any share w of b): lerp translation and scale, shortest-arc nlerp rotation."""
    out = {}
    for i in a:
        qa, qb = a[i]['rotation'], b[i]['rotation']
        if qa @ qb < 0:
            qb = -qb
        q = (1-w)*qa+w*qb
        out[i] = {'translation': (1-w)*a[i]['translation']+w*b[i]['translation'], 'rotation': q/np.linalg.norm(q),
                  'scale': (1-w)*a[i]['scale']+w*b[i]['scale']}
    return out


def sample_times(duration, loop):
    n = max(2, math.ceil(duration*HZ-1e-9))
    return np.linspace(0, duration, n+1)[:-1] if loop else np.linspace(0, duration, n+1)


def windows(low, times, duration, loop, gap, shortest=0.0):
    """Planted windows [[on, off], ...] from the lowest contact height per sample, each end interpolated where it
    crosses the gap; in a loop an `off` past the duration wraps. A window shorter than `shortest` seconds is a
    graze, not a stance. Returns the windows, their touchdown times, the grazes ({time_s, clearance_mm}: the
    graze's start and its lowest height above the ground) and which samples are in a kept window."""
    on = low <= gap
    n = len(times)
    dt = float(times[1]-times[0])           # samples are k*dt, so an index past either end is a time in the next cycle
    if on.all():
        return [[0.0, float(duration)]], [], [], on.copy()
    if not on.any():
        return [], [], [], on.copy()
    starts = [i for i in range(n) if on[i] and (not on[i-1] if loop else i == 0 or not on[i-1])]
    keep, pairs, downs, grazes = np.zeros(n, bool), [], [], []
    for first in starts:
        last = first
        while (loop or last+1 < n) and on[(last+1) % n]:
            last += 1
        a, b = low[(first-1) % n], low[first]
        t0 = 0.0 if not loop and first == 0 else (first-1+float(np.clip((a-gap)/(a-b), 0, 1)))*dt
        a, b = low[last % n], low[(last+1) % n]
        t1 = float(duration) if not loop and last == n-1 else (last+float(np.clip((gap-a)/(b-a), 0, 1)))*dt
        if t0 < 0:
            t0, t1 = t0+duration, t1+duration
        cut = not loop and (first == 0 or last == n-1)     # a stance the ends of a once-played clip cut short
        if t1-t0 < shortest and not cut:
            grazes.append({'time_s': t0, 'clearance_mm': float(min(low[k % n] for k in range(first, last+1)))*1000})
            continue
        pairs.append([t0, t1])
        if loop or first > 0:
            downs.append(t0)
        for k in range(first, last+1):
            keep[k % n] = True
    return sorted(pairs), sorted(downs), sorted(grazes, key=lambda g: g['time_s']), keep


def legs_read(contacts, times, duration, loop, ground, gap, shortest=0.0):
    """leg -> planted windows, duty, touchdowns, grazes and the planted velocities (one per planted sample step)."""
    out, n = {}, len(times)
    dt = times[1]-times[0]
    for leg, c in contacts.items():
        h = c[..., 2]-ground
        low = h.min(axis=1)
        stance, downs, grazes, keep = windows(low, times, duration, loop, gap, shortest)
        vel = []
        for i in range(n if loop else n-1):
            j = (i+1) % n
            if keep[i]:
                m = (h[i] <= gap) & (h[j] <= gap)
                if m.any():
                    vel.append((c[j][m, :2]-c[i][m, :2]).mean(axis=0)/dt)
        out[leg] = {'stance': stance, 'touchdowns': downs, 'grazes': grazes, 'duty': float(keep.mean()),
                    'velocities': np.array(vel).reshape(-1, 2), 'lowest': low}
    return out


def speeds(read, forward):
    """Each leg's planted backward speed (the median of its planted samples) and its spread (standard deviation),
    the clip's travel speed (the median over every leg's planted samples together) and each leg's skate: RMS over
    all its planted samples of the velocity's horizontal difference from -travel*forward. The median follows the
    flat-foot stance; skate still reads a heel-toe roll that slides."""
    per, pooled = {}, []
    for leg, x in read.items():
        v = x['velocities']
        back = -(v @ forward)
        pooled += back.tolist()
        per[leg] = {'speed_mps': float(np.median(back)) if len(v) else None,
                    'spread_mps': float(back.std()) if len(v) else None}
    travel = float(np.median(pooled)) if pooled else None
    for leg, x in read.items():
        v = x['velocities']
        per[leg]['skate_mps'] = float(np.sqrt((np.linalg.norm(v+travel*forward, axis=1)**2).mean())) if len(v) else None
    return travel, per


class Sampled:
    """One clip (or a blend) sampled: per sample the vertex heights, the contact vertices and the joints. The ground
    is the rest pose's lowest vertex, as rig-check's; `lowest` is the lowest any vertex goes in these samples."""

    def __init__(self, body, pose_at, times, duration, loop, contacts):
        self.times, self.duration, self.loop = times, duration, loop
        ids = np.concatenate([np.asarray(v, int) for v in contacts.values()]) if contacts else np.zeros(0, int)
        z, cpos, jpos = [], [], []
        for t in times:
            world = body.world(pose_at(float(t)))
            v = body.vertices(world)
            z.append(v[:, 2].astype(np.float32))
            cpos.append(v[ids])
            jpos.append(body.joints(world))
        self.z, self.joints = np.array(z), np.array(jpos)
        self.ground = float(body.rest[:, 2].min())
        self.lowest = float(min(float(zz.min()) for zz in z))
        cpos, k, self.contacts = np.array(cpos), 0, {}
        for leg, v in contacts.items():
            self.contacts[leg] = cpos[:, k:k+len(v)]
            k += len(v)


def clip_sample(body, clip, loop, contacts):
    c = body.clips[clip]
    times = sample_times(c.duration, loop)
    return Sampled(body, lambda t: body.full(clip, t), times, c.duration, loop, contacts)


def clip_read(body, clip, loop, contacts, gap, forward, roots, shortest=0.0):
    """Everything one clip measures. `roots`: the joints whose horizontal travel over the clip is its root travel."""
    c = body.clips[clip]
    s = clip_sample(body, clip, loop, contacts)
    first, last = body.world(body.full(clip, 0.0)), body.world(body.full(clip, c.duration))
    seam = float(np.linalg.norm(body.vertices(last)-body.vertices(first), axis=1).max())
    travel = body.joints(last, roots)-body.joints(first, roots)
    out = {'duration_s': c.duration, 'loop': loop, 'loop_seam_mm': seam*1000, 'ground_z': s.ground,
           'below_rest_ground_mm': max(0.0, s.ground-s.lowest)*1000,
           'root_travel_mm': float(np.linalg.norm(travel[:, :2], axis=1).max())*1000, 'legs': {}}
    read = legs_read(s.contacts, s.times, c.duration, loop, s.ground, gap, shortest) if contacts else {}
    out['travel_speed_mps'], per = speeds(read, forward) if read else (None, {})
    for leg, x in read.items():
        out['legs'][leg] = {'duty': x['duty'], 'stance_s': x['stance'], 'touchdowns_s': x['touchdowns'],
                            'grazes': x['grazes'], **per[leg]}
    return out, s, read


def phases(reads, clips, legs):
    """For one synced blend's clips: which clips step (a leg touches down; only those are on the blend's cycle),
    the reference leg (the first in contract order that touches down in every clip that steps), each stepping
    clip's phase offset (the reference touchdown over the duration) and each leg's touchdown phases after it."""
    stepping = [c for c in clips if any(reads[c][1][l]['touchdowns'] for l in legs)]
    ref = next((l for l in legs if all(reads[c][1][l]['touchdowns'] for c in stepping)), None)
    out = {}
    for c in clips:
        info, read = reads[c]
        if c not in stepping or ref is None:
            out[c] = {'phase_offset': 0.0, 'relative': {}}
            continue
        d, t0 = info['duration_s'], read[ref]['touchdowns'][0]
        out[c] = {'phase_offset': t0/d, 'relative': {l: sorted(((t-t0)/d) % 1 for t in read[l]['touchdowns']) for l in legs}}
    return ref, stepping, out


def phase_difference(a, b):
    """The largest circular difference between two clips' relative touchdown phases, leg by leg; 0.5 (the most two
    phases can differ) when a leg steps a different number of times in each."""
    worst = 0.0
    for leg in a:
        x, y = a[leg], b.get(leg, [])
        if len(x) != len(y):
            return 0.5
        for p, q in zip(x, y):
            worst = max(worst, abs((p-q+.5) % 1-.5))
    return worst


def sync_times(durations, offsets, cycles, w, t):
    """The synced blend1d clock for two neighbouring points, t seconds in, weight w on the second. The points on
    the cycle share one phase whose frequency lerps between theirs; a point off the cycle (no leg touches down in
    its clip) runs on its own clock and adds nothing to the frequency. Returns each point's clip time."""
    on = [k for k in (0, 1) if cycles[k]]
    if len(on) == 2:
        freq = (1-w)/durations[0]+w/durations[1]
    elif on:
        freq = 1/durations[on[0]]
    phi = t*freq if on else 0.0
    return [((phi+offsets[k]) % 1)*durations[k] if cycles[k] else t % durations[k] for k in (0, 1)]


def blend_read(body, a, b, offsets, cycles, contacts, gap, forward, travel, shortest=0.0):
    """Two clips of a synced blend, half and half, on the sync clock (sync_times); the planted slip against the
    mean of the two travel speeds, sampled over one blended cycle."""
    d = [body.clips[a].duration, body.clips[b].duration]
    on = [k for k in (0, 1) if cycles[(a, b)[k]]]
    period = 1/(.5/d[0]+.5/d[1]) if len(on) == 2 else d[on[0]] if on else max(d)
    times = sample_times(period, True)

    def pose(t):
        ta, tb = sync_times(d, [offsets[a], offsets[b]], [cycles[a], cycles[b]], .5, t)
        return blend(body.full(a, ta), body.full(b, tb), .5)
    s = Sampled(body, pose, times, period, True, contacts)
    read = legs_read(s.contacts, times, period, True, s.ground, gap, shortest)
    speed = (travel[a]+travel[b])/2
    slip = {}
    for leg, x in read.items():
        v = x['velocities']
        slip[leg] = float(np.sqrt((np.linalg.norm(v+speed*forward, axis=1)**2).mean())) if len(v) else None
    known = [v for v in slip.values() if v is not None]
    return {'clips': [a, b], 'period_s': period, 'speed_mps': speed, 'slip_mps': slip,
            'worst_slip_mps': max(known) if known else None}


def angles(joints, chain_idx):
    """(samples, inner joints): the angle at each inner joint of a chain, degrees, 180 straight."""
    out = []
    for a, b, c in zip(chain_idx, chain_idx[1:], chain_idx[2:]):
        u, v = joints[:, a]-joints[:, b], joints[:, c]-joints[:, b]
        cos = (u*v).sum(axis=1)/np.maximum(np.linalg.norm(u, axis=1)*np.linalg.norm(v, axis=1), 1e-12)
        out.append(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return np.array(out).T if out else np.zeros((len(joints), 0))


def triples(body):
    """(parent, joint, child) for every joint with a parent and a child: where a bend can be read."""
    return [(body.parent[j], j, c) for j in body.names if body.parent[j] for c in body.names if body.parent[c] == j]


def bend_ranges(body, joints, chains):
    """{(a, b, c): (lowest, highest)} inner angle at b, degrees (180 straight), over the samples `joints`."""
    index = {n: k for k, n in enumerate(body.names)}
    out = {}
    for chain in chains:
        a = angles(joints, [index[j] for j in chain])
        out[tuple(chain)] = (float(a[:, 0].min()), float(a[:, 0].max()))
    return out


def clusters(body, ids):
    """Split vertex ids into groups: by the joint that carries each most, then into clusters joined through a shared
    triangle, a welded position or neighbouring cells at rest (cells of CLUSTER_SHARE of the model's size)."""
    if not len(ids):
        return []
    cell = CLUSTER_SHARE*(float(np.ptp(body.rest, axis=0).max()) or 1.0)
    carried = np.array([str(j) for j in body.vjoint[ids]])
    parent = {int(v): int(v) for v in ids}

    def root(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def join(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    member = np.zeros(len(body.rest), bool)
    member[ids] = True
    same = {int(v): c for v, c in zip(ids, carried)}
    for f in body.faces[member[body.faces].sum(axis=1) >= 2]:
        inside = [int(v) for v in f if member[v]]
        for a, b in zip(inside, inside[1:]):
            if same[a] == same[b]:
                join(a, b)
    first = {}
    for v, c, k in zip(ids, carried, np.floor(body.rest[ids]/cell).astype(int)):
        first.setdefault((c, *k), int(v))
        join(first[(c, *k)], int(v))
    for (c, *k), v in sorted(first.items()):
        for d in np.ndindex(3, 3, 3):
            m = (c, k[0]+d[0]-1, k[1]+d[1]-1, k[2]+d[2]-1)
            if m in first:
                join(v, first[m])
    groups = {}
    for v in ids:
        groups.setdefault(root(int(v)), []).append(int(v))
    out = [{'joint': None if same[g[0]] == 'None' else same[g[0]], 'ids': np.array(sorted(g))} for g in groups.values()]
    return sorted(out, key=lambda c: (str(c['joint']), body.rest[c['ids']].min(axis=0).round(4).tolist()))


def describe(body, cluster):
    g = cluster['ids']
    return {'joint': cluster['joint'], 'nodes': sorted(set(body.vnode[g].tolist())), 'vertices': int(len(g)),
            'box': [[r(x) for x in body.rest[g].min(axis=0)], [r(x) for x in body.rest[g].max(axis=0)]]}


def touching(body, sampled, gap, exclude=None):
    """Clusters of vertices within the gap of the rest ground in any sample, leaving out `exclude`, each with the
    share of samples it touches and its planted windows."""
    near = (sampled.z-sampled.ground) <= gap
    hit = near.any(axis=0)
    if exclude is not None and len(exclude):
        hit[np.asarray(exclude, int)] = False
    out = []
    for c in clusters(body, np.flatnonzero(hit)):
        low = sampled.z[:, c['ids']].min(axis=1).astype(float)-sampled.ground
        stance, downs, _, _ = windows(low, sampled.times, sampled.duration, sampled.loop, gap)
        out.append({**describe(body, c), 'share': r(near[:, c['ids']].any(axis=1).mean()),
                    'stance_s': [[r(a), r(b)] for a, b in stance], 'touchdowns_s': [r(t) for t in downs]})
    return out
