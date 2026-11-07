"""Frozen-GLB rig authoring: declarative scripts, manual skins and sampled clips.

Append-only binary export avoids a Blender import/re-export changing the delivered
geometry. This is an initial stage, not the video-reference scoring pipeline.
"""
import copy
import hashlib
import json
import math
import re
import struct
from pathlib import Path

import numpy as np

from . import glb, render, script_check, skin

class ContractError(ValueError):
    """A rig that built and is out of contract. `report` is the whole check, numbers included: a builder told only
    which gate failed had to loosen the contract in a scratch copy to see by how much."""

    def __init__(self, report):
        super().__init__('rig contract: '+'; '.join(report['issues']))
        self.report = report


AXES = {'X': np.array([1., 0, 0]), 'Y': np.array([0., 0, -1]), 'Z': np.array([0., 1, 0])}
IMPORTS = {'math', 'numpy'}


def vector(value):
    a = np.asarray(value, float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError('expected a finite three-component vector')
    return a


def to_gltf(value):
    x, y, z = vector(value)
    return np.array([x, z, -y])


def to_blender(points):
    return np.stack((points[:, 0], -points[:, 2], points[:, 1]), axis=1)


def components(points, faces):
    """Same 1e-6 m position welding as review.component_count; seams are not pieces."""
    _, ids = np.unique(np.round(points, 6), axis=0, return_inverse=True)
    parent = list(range(int(ids.max())+1))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b, c in ids[faces]:
        parent[root(b)] = parent[root(a)]
        parent[root(c)] = parent[root(a)]
    roots = np.array([root(i) for i in ids])
    return np.unique(roots, return_inverse=True)[1]


def model_data(doc, blob):
    model, data = {'nodes': []}, []
    for ni, world in glb.mesh_nodes(doc):
        node = doc['nodes'][ni]
        points, faces, prims, offset = [], [], [], 0
        for p in doc['meshes'][node['mesh']]['primitives']:
            if p.get('mode', 4) != 4:
                raise ValueError('rig source must contain triangle primitives only')
            pos = glb.accessor(doc, blob, p['attributes']['POSITION'])
            pts = to_blender(pos @ world[:3, :3].T + world[:3, 3])
            idx = (glb.accessor(doc, blob, p['indices']).astype(int).ravel() if 'indices' in p else np.arange(len(pos)))
            faces.append(idx.reshape(-1, 3)+offset)
            points.append(pts)
            prims.append((p, pts))
            offset += len(pos)
        pts = np.concatenate(points)
        if not len(pts) or not np.isfinite(pts).all():
            raise ValueError('rig source has empty or nonfinite positions')
        piece = components(pts, np.concatenate(faces))
        groups = []
        for k in range(int(piece.max())+1):
            selected = pts[piece == k]
            groups.append({'id': k, 'lo': selected.min(axis=0).tolist(), 'hi': selected.max(axis=0).tolist(),
                           'centroid': selected.mean(axis=0).tolist(), 'vertices': len(selected)})
        name = node.get('name', f'node{ni}')
        model['nodes'].append({'name': name, 'lo': pts.min(axis=0).tolist(), 'hi': pts.max(axis=0).tolist(),
                               'centroid': pts.mean(axis=0).tolist(), 'pieces': groups})
        data.append((ni, name, prims, piece, world))
    if not data:
        raise ValueError('rig source contains no mesh nodes')
    all_pts = np.concatenate([pts for _, _, prims, _, _ in data for _, pts in prims])
    model['lo'], model['hi'] = all_pts.min(axis=0).tolist(), all_pts.max(axis=0).tolist()
    return model, data


def contract(spec):
    fps = spec.get('fps', 30)
    if isinstance(fps, bool) or not isinstance(fps, int) or not 1 <= fps <= 120:
        raise ValueError('rig fps must be an integer from 1 to 120')
    influences = spec.get('max_influences', 4)
    if isinstance(influences, bool) or not isinstance(influences, int) or not 1 <= influences <= 4:
        raise ValueError('rig max_influences must be from 1 to 4')
    clips = spec.get('clips', {})
    if not clips:
        raise ValueError('rig contract needs clips')
    for name, clip in clips.items():
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', name):
            raise ValueError('clip names must be safe file names')
        seconds = clip.get('seconds', 0)
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= 60:
            raise ValueError(f'{name}: seconds must be finite, positive and at most 60')
        if abs(seconds*fps-round(seconds*fps)) > 1e-6:
            raise ValueError(f'{name}: duration must be a whole number of frames')


def skeleton_check(bones, spec):
    names = [b['name'] for b in bones]
    if not names or len(set(names)) != len(names) or len(names) > spec.get('bones', {}).get('max', 64):
        raise ValueError('skeleton must have unique names and fit its bone budget')
    required = spec.get('bones', {}).get('required', [])
    if set(required)-set(names):
        raise ValueError(f'missing required bones: {set(required)-set(names)}')
    done, roots = set(), 0
    for bone in bones:
        head, tail = vector(bone['head']), vector(bone['tail'])
        if np.linalg.norm(tail-head) < 1e-8:
            raise ValueError('bone has zero length')
        parent = bone.get('parent')
        if parent is None:
            roots += 1
        elif parent not in done:
            raise ValueError('bones must be parent-first with no cycles')
        done.add(bone['name'])
    if roots != 1:
        raise ValueError('skeleton needs exactly one root')


def normalize_weights(rows, names, count, limit):
    if isinstance(rows, str):
        rows = {rows: np.ones(count)}
    if not isinstance(rows, dict) or not rows or set(rows)-set(names):
        raise ValueError('weights must name existing bones')
    w = np.zeros((count, len(names)))
    for name, value in rows.items():
        a = np.asarray(value, float)
        if a.shape != (count,) or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError('weights must be finite nonnegative arrays, one entry per vertex')
        w[:, names.index(name)] = a
    order = np.argsort(-w, axis=1, kind='stable')[:, :min(limit, len(names))]
    kept = np.take_along_axis(w, order, axis=1)
    total = kept.sum(axis=1)
    if (total <= 0).any():
        raise ValueError('unweighted vertices: provide explicit weights, no nearest-bone fallback')
    kept /= total[:, None]
    return np.pad(order, ((0, 0), (0, 4-order.shape[1]))).astype(np.uint16), np.pad(kept, ((0, 0), (0, 4-kept.shape[1]))).astype(np.float32)


def worn_on(script_worn, node_names):
    """The script's optional top-level WORN, checked: {node worn: node it lies on}. Each vertex of a worn node takes
    the weights of the nearest vertex of the node under it, so two layers move as one whatever tests the script's
    `weights` makes. Tests on position alone cannot promise that: a vest a centimetre off a chest got weights up to
    0.7 apart from the skin under it at the edge of an arm's region, and slid 35 mm."""
    if script_worn is None:
        return {}
    if not isinstance(script_worn, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in script_worn.items()):
        raise ValueError('WORN must be a dictionary {node worn: node it lies on}')
    for worn, under in script_worn.items():
        if worn not in node_names or under not in node_names:
            raise ValueError(f'WORN names a node the model does not have: {worn!r} on {under!r}')
        if under in script_worn or worn == under:
            raise ValueError(f'WORN: {under!r} is worn itself; name the node at the bottom')
    return dict(script_worn)


def nearest(points, others, chunk=512):
    """For each point, the index of the nearest of `others`."""
    out = np.empty(len(points), dtype=np.int64)
    for k in range(0, len(points), chunk):
        part = points[k:k+chunk]
        out[k:k+chunk] = ((part[:, None, :]-others[None, :, :])**2).sum(axis=2).argmin(axis=1)
    return out


def quaternion(rotations):
    for axis, degrees in rotations:
        if axis not in AXES or not math.isfinite(degrees):
            raise ValueError('rotation needs model axis X/Y/Z and finite degrees')
    # Compose Hamilton quaternions directly, avoiding trace singularities at 180 degrees.
    q = np.array([0., 0, 0, 1])
    for axis, degrees in rotations:
        angle = math.radians(degrees)/2
        v, w = AXES[axis]*math.sin(angle), math.cos(angle)
        q = np.r_[w*q[:3]+q[3]*v+np.cross(v, q[:3]), w*q[3]-v @ q[:3]]
    return q / np.linalg.norm(q)


def append_accessor(doc, blob, values, kind, component=5126):
    while len(blob) % 4:
        blob.append(0)
    dtype = glb.COMPONENT[component]
    array = np.asarray(values, dtype=dtype)
    view = {'buffer': 0, 'byteOffset': len(blob), 'byteLength': array.nbytes}
    blob.extend(array.tobytes())
    doc['bufferViews'].append(view)
    acc = {'bufferView': len(doc['bufferViews'])-1, 'componentType': component, 'count': len(array), 'type': kind}
    if kind == 'SCALAR':
        acc.update(min=[float(array.min())], max=[float(array.max())])
    doc['accessors'].append(acc)
    return len(doc['accessors'])-1


def write_glb(path, doc, blob):
    while len(blob) % 4:
        blob.append(0)
    doc['buffers'][0]['byteLength'] = len(blob)
    js = json.dumps(doc, separators=(',', ':')).encode()
    js += b' ' * (-len(js) % 4)
    data = struct.pack('<III', glb.GLB_MAGIC, 2, 28+len(js)+len(blob))
    data += struct.pack('<II', len(js), glb.CHUNK_JSON)+js+struct.pack('<II', len(blob), glb.CHUNK_BIN)+blob
    Path(path).write_bytes(data)


def build(source, script, spec, out, failed=None):
    """Build, check, and deliver to `out` only what holds its contract. A candidate that fails it is deleted, or
    with `failed` (a path) left there beside its report, for whoever has to see what it did."""
    source, script, out = Path(source), Path(script), Path(out)
    if source.resolve() == out.resolve():
        raise ValueError('rig output must not overwrite the frozen source')
    contract(spec)
    refused = script_check.check_script(script.read_text(encoding='utf-8'), required=('skeleton', 'weights', 'pose'), allowed_imports=IMPORTS)
    if refused:
        raise ValueError('rig script refused: '+'; '.join(refused))
    doc, binary = glb.read_glb(source)
    if doc.get('skins') or doc.get('animations') or len(doc['buffers']) != 1 or doc.get('extensionsRequired'):
        raise ValueError('source must be a static, uncompressed single-buffer GLB')
    if any('sparse' in a for a in doc['accessors']):
        raise ValueError('sparse accessors are not supported by the rig reader')
    model, data = model_data(doc, binary)
    env = {'__name__': 'rig_script'}
    exec(compile(script.read_text(encoding='utf-8'), str(script), 'exec'), env)
    bones = env['skeleton'](copy.deepcopy(model))
    skeleton_check(bones, spec)
    names = [b['name'] for b in bones]
    first = len(doc['nodes'])
    indices = dict(zip(names, range(first, first+len(names))))
    heads = {b['name']: to_gltf(b['head']) for b in bones}
    for b in bones:
        parent = b.get('parent')
        doc['nodes'].append({'name': b['name'], 'translation': (heads[b['name']]-(heads[parent] if parent else 0)).tolist()})
        if parent:
            doc['nodes'][indices[parent]].setdefault('children', []).append(indices[b['name']])
        else:
            doc['scenes'][doc.get('scene', 0)]['nodes'].append(indices[b['name']])
    doc['skins'] = []
    blob = bytearray(binary)
    worn = worn_on(env.get('WORN'), [name for _, name, _, _, _ in data])
    skinned = {}                                   # node -> [(points, joints, weights)] a primitive
    for ni, name, prims, pieces, own in sorted(data, key=lambda d: d[1] in worn):     # what is worn, after what it lies on
        start, done = 0, []
        under = worn.get(name)
        if under is not None:
            below = [np.concatenate(column) for column in zip(*skinned[under])]
        for _, points in prims:
            points.setflags(write=False)
            if under is not None:
                k = nearest(points, below[0])
                joints, weights = below[1][k], below[2][k]
            else:
                part = pieces[start:start+len(points)].copy()
                part.setflags(write=False)
                rows = env['weights'](name, points, part)
                joints, weights = normalize_weights(rows, names, len(points), spec.get('max_influences', 4))
            done.append((points, joints, weights))
            start += len(points)
        skinned[name] = done
    for ni, name, prims, pieces, own in data:
        mesh = copy.deepcopy(doc['meshes'][doc['nodes'][ni]['mesh']])
        for prim, (_, joints, weights) in zip(mesh['primitives'], skinned[name]):
            prim['attributes']['JOINTS_0'] = append_accessor(doc, blob, joints, 'VEC4', 5123)
            prim['attributes']['WEIGHTS_0'] = append_accessor(doc, blob, weights, 'VEC4')
        doc['nodes'][ni]['mesh'] = len(doc['meshes'])
        doc['meshes'].append(mesh)
        inverse = []
        for name in names:
            joint = np.eye(4)
            joint[:3, 3] = heads[name]
            inverse.append((np.linalg.inv(joint) @ own).T.reshape(16))
        doc['nodes'][ni]['skin'] = len(doc['skins'])
        doc['skins'].append({'joints': list(indices.values()), 'skeleton': indices[bones[0]['name']],
                             'inverseBindMatrices': append_accessor(doc, blob, inverse, 'MAT4')})
    doc['animations'] = []
    for name, c in spec['clips'].items():
        frames = round(c['seconds']*spec.get('fps', 30))
        times = np.linspace(0, c['seconds'], frames+1)
        time_accessor = append_accessor(doc, blob, times, 'SCALAR')
        animation = {'name': name, 'channels': [], 'samplers': [], 'extras': {'loop': c.get('loop', False), 'events': c.get('events', [])}}
        poses = [env['pose'](name, float(t)) for t in times]
        if any(not isinstance(p, dict) or set(p)-set(names) for p in poses):
            raise ValueError('pose must return a dictionary of known bones')
        for bone in bones:
            rotations, translations = [], []
            parent = bone.get('parent')
            origin = heads[bone['name']]-(heads[parent] if parent else 0)
            for pose in poses:
                value = pose.get(bone['name'], {})
                value = {'rotate': value} if isinstance(value, list) else value
                rotations.append(quaternion(value.get('rotate', [])))
                translations.append(origin+to_gltf(value.get('move', (0, 0, 0))))
            for path, values, kind in (('rotation', rotations, 'VEC4'), ('translation', translations, 'VEC3')):
                output = append_accessor(doc, blob, values, kind)
                animation['channels'].append({'sampler': len(animation['samplers']), 'target': {'node': indices[bone['name']], 'path': path}})
                animation['samplers'].append({'input': time_accessor, 'output': output, 'interpolation': 'LINEAR'})
        doc['animations'].append(animation)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Verify before replacing any prior delivery.
    temporary = out.with_suffix('.candidate.glb')
    write_glb(temporary, doc, blob)
    report = check(source, temporary, spec)
    report.update(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                  contract=copy.deepcopy(spec), contract_sha256=hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest())
    if report['issues']:
        if failed is None:
            temporary.unlink()
        else:
            temporary.replace(failed)
            Path(failed).with_suffix('.rig.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        raise ContractError(report)
    temporary.replace(out)
    report['rig_sha256'] = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix('.rig.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def check(source, rigged, spec):
    contract(spec)
    doc, blob = glb.read_glb(rigged)
    source_doc, source_blob = glb.read_glb(source)
    issues, max_weight_error, max_influences = [], 0., 0
    for ni, _ in glb.mesh_nodes(doc):
        node = doc['nodes'][ni]
        if 'skin' not in node:
            raise ValueError('mesh node has no skin')
        for prim in doc['meshes'][node['mesh']]['primitives']:
            if not {'JOINTS_0', 'WEIGHTS_0'} <= prim['attributes'].keys():
                raise ValueError('mesh primitive is missing joints or weights')
    if any('sparse' in a for a in doc['accessors']):
        raise ValueError('sparse accessors are not supported by the rig reader')
    for key in ('materials', 'textures', 'images', 'samplers'):
        if doc.get(key) != source_doc.get(key):
            issues.append(f'frozen source {key} changed')
    source_prims = [p for ni, _ in glb.mesh_nodes(source_doc) for p in source_doc['meshes'][source_doc['nodes'][ni]['mesh']]['primitives']]
    output_prims = [p for ni, _ in glb.mesh_nodes(doc) for p in doc['meshes'][doc['nodes'][ni]['mesh']]['primitives']]
    if len(source_prims) != len(output_prims):
        issues.append('frozen source primitive count changed')
    else:
        for a, b in zip(source_prims, output_prims):
            for key, ai in a['attributes'].items():
                if key not in b['attributes'] or not np.array_equal(glb.accessor(source_doc, source_blob, ai), glb.accessor(doc, blob, b['attributes'][key])):
                    issues.append(f'frozen source attribute {key} changed')
    rest, posed = render.load(rigged), skin.posed(rigged)
    original = render.load(source)
    rest_error = float(np.max(np.abs(rest.tris-original.tris))) if rest.tris.shape == original.tris.shape else float('inf')
    color_error = float(np.max(np.abs(rest.colors-original.colors))) if rest.colors.shape == original.colors.shape else float('inf')
    bind_error = float(np.max(np.abs(posed.tris-rest.tris)))
    if rest_error > 1e-7 or color_error > 1e-7:
        issues.append('frozen source geometry or colours changed')
    if bind_error > 1e-5:
        issues.append('inverse binds do not recover the rest pose')
    joint_ids = set(j for s in doc.get('skins', []) for j in s['joints'])
    names = [doc['nodes'][j].get('name') for j in sorted(joint_ids)]
    if not joint_ids or len(set(names)) != len(names):
        issues.append('joint names must be present and unique')
    parents = {c: i for i, n in enumerate(doc['nodes']) for c in n.get('children', [])}
    if sum(parents.get(j) not in joint_ids for j in joint_ids) != 1:
        issues.append('skeleton must have one joint root')
    if len(names) > spec.get('bones', {}).get('max', 64) or set(spec.get('bones', {}).get('required', []))-set(names):
        issues.append('bone contract failed')
    for ni, _ in glb.mesh_nodes(doc):
        node = doc['nodes'][ni]
        if 'skin' not in node:
            issues.append('mesh node has no skin')
            continue
        for prim in doc['meshes'][node['mesh']]['primitives']:
            a = prim['attributes']
            w = glb.accessor(doc, blob, a['WEIGHTS_0'])
            j = glb.accessor(doc, blob, a['JOINTS_0'])
            if not np.isfinite(w).all() or (w < 0).any() or (j < 0).any() or (j >= len(doc['skins'][node['skin']]['joints'])).any():
                issues.append('invalid weights or joint indices')
            max_weight_error = max(max_weight_error, float(np.max(np.abs(w.sum(axis=1)-1))))
            max_influences = max(max_influences, int((w > 0).sum(axis=1).max()))
    if max_weight_error > 1e-5 or max_influences > spec.get('max_influences', 4):
        issues.append('weight normalization/influence contract failed')
    if spec.get('rigid'):
        _, objects = model_data(doc, blob)
        for ni, name, prims, pieces, _ in objects:
            joints = np.concatenate([glb.accessor(doc, blob, p['attributes']['JOINTS_0']) for p, _ in prims])
            weights = np.concatenate([glb.accessor(doc, blob, p['attributes']['WEIGHTS_0']) for p, _ in prims])
            dominant = joints[np.arange(len(joints)), weights.argmax(axis=1)]
            if not np.allclose(weights.max(axis=1), 1) or any(len(set(dominant[pieces == k])) != 1 for k in set(pieces)):
                issues.append(f'{name}: rigid connected pieces must follow one bone at full weight')
    by_name = {a.get('name'): a for a in doc.get('animations', [])}
    clips = {}
    base_edges = np.linalg.norm(rest.tris-np.roll(rest.tris, 1, axis=1), axis=2)
    areas = np.linalg.norm(np.cross(rest.tris[:, 1]-rest.tris[:, 0], rest.tris[:, 2]-rest.tris[:, 0]), axis=1)/2
    for name, c in spec['clips'].items():
        if name not in by_name:
            issues.append(f'missing clip {name}')
            continue
        a = by_name[name]
        for ch in a['channels']:
            sampler = a['samplers'][ch['sampler']]
            times = glb.accessor(doc, blob, sampler['input']).ravel()
            values = glb.accessor(doc, blob, sampler['output'])
            if not np.isfinite(values).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or abs(times[0]) > 1e-6 or abs(times[-1]-c['seconds']) > 1e-5:
                issues.append(f'{name}: invalid duration or channel values')
            if ch['target']['node'] not in joint_ids or ch['target']['path'] not in ('rotation', 'translation', 'scale'):
                issues.append(f'{name}: animation must target joints only')
            if ch['target']['path'] == 'scale' and not np.allclose(values, 1):
                issues.append(f'{name}: animated scale is forbidden')
            if ch['target']['path'] == 'rotation' and not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5):
                issues.append(f'{name}: rotation quaternion is not normalized')
        samples = [skin.posed(rigged, name, t).tris for t in np.linspace(0, c['seconds'], round(c['seconds']*spec.get('fps', 30))+1)]
        loop = float(np.linalg.norm(samples[-1]-samples[0], axis=2).max()*1000)
        displacement = max(float(np.linalg.norm(p-rest.tris, axis=2).max()*1000) for p in samples)
        ratios = [np.divide(np.linalg.norm(p-np.roll(p, 1, axis=1), axis=2), base_edges, out=np.ones_like(base_edges), where=base_edges > 1e-8) for p in samples]
        stretch = max(float(r.max()) for r in ratios)
        stretched_area = max(float(areas[np.any(r > 1.5, axis=1)].sum()/max(areas.sum(), 1e-12)) for r in ratios)
        minimum = min(float(p[:, :, 1].min()) for p in samples)
        if c.get('loop') and loop > spec.get('loop_mm_max', 2):
            issues.append(f"{name}: loop seam {loop:.3f} mm ({spec.get('loop_mm_max', 2):g} allowed): the pose at "
                          f"t = {c['seconds']:g} s is not the pose at t = 0")
        if displacement < c.get('min_displacement_mm', 0):
            issues.append(f"{name}: clip does not move enough ({displacement:.1f} mm from rest at most, "
                          f"{c['min_displacement_mm']:g} asked)")
        ground = float(rest.tris[:, :, 1].min())
        if 'ground_drop_mm_max' in c and minimum < ground-c['ground_drop_mm_max']/1000:
            issues.append(f"{name}: pose sinks below its rest ground ({(ground-minimum)*1000:.1f} mm, "
                          f"{c['ground_drop_mm_max']:g} allowed)")
        if stretch > c.get('max_edge_ratio', float('inf')):
            issues.append(f"{name}: excessive edge stretching ({stretch:.2f}x, {c['max_edge_ratio']:g}x allowed)")
        if stretched_area > c.get('stretch_area_max', float('inf')):
            issues.append(f"{name}: too much surface area stretches past 1.5 ({stretched_area*100:.2f}%, "
                          f"{c['stretch_area_max']*100:g}% allowed)")
        if spec.get('rigid') and max(float(np.abs(r-1).max()) for r in ratios) > 1e-5:
            issues.append(f'{name}: rigid pieces deform')
        clips[name] = {'seconds': c['seconds'], 'loop': c.get('loop', False), 'loop_mm': loop,
                       'max_displacement_mm': displacement, 'max_edge_ratio': stretch,
                       'stretch_area_fraction': stretched_area, 'min_y_m': minimum}
    return {'issues': sorted(set(issues)), 'rest_position_error_m': rest_error, 'rest_colour_error': color_error,
            'bind_error_m': bind_error, 'weight_sum_error': max_weight_error, 'max_influences': max_influences,
            'bones': names, 'clips': clips, 'triangles': len(rest.tris)}


def preview(rigged, clip, times, out, spec):
    from PIL import Image, ImageDraw
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    shot = spec.get('shot', {'name': 'hero', 'cam_pos': [-2.4, 1.8, -3.4], 'cam_target': [0, .65, 0], 'fov': 36})
    frames = []
    for k, t in enumerate(times):
        frame = render.render(skin.posed(rigged, clip, t), shot, width=480, height=480)
        ImageDraw.Draw(frame).text((12, 12), f'{clip}  {t:.3f}s', fill=(20, 20, 20))
        frame.save(out / f'{clip}-{k:03}.png')
        frames.append(frame)
    strip = Image.new('RGB', (480*len(frames), 480))
    for k, frame in enumerate(frames):
        strip.paste(frame, (480*k, 0))
    strip.save(out / f'{clip}-strip.png')
    return out / f'{clip}-strip.png'
