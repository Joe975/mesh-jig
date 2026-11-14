"""Optional, explicitly bound face rigs on frozen static GLBs.

No landmark guesses or topology invention. Build/check/preview share the existing
rig stage; the runtime controller lives in face_motion.pose_at.
"""
import copy
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from . import face_motion, glb, rig, skin

STATES = ('look_at', 'talk', 'smile', 'anger', 'sadness', 'confusion', 'yell')
CONTROLS = ('neck', 'eye_l', 'eye_r', 'jaw', 'mouth_l', 'mouth_r', 'brow_l', 'brow_r')


def contract(spec):
    if not isinstance(spec, dict):
        raise ValueError('face contract must be a JSON object')
    spec = copy.deepcopy(spec)
    allowed = {'version', 'fps', 'face_scale_m', 'look_target', 'jaw_open_degrees', 'controls', 'mouth_topology', 'required', 'gates', 'preview', 'visible_eyes'}
    if set(spec)-allowed or spec.get('version') != 1:
        raise ValueError('face contract needs version 1 and known fields only')
    scale = spec.get('face_scale_m', 0)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('face_scale_m must be finite and positive')
    rig.vector(spec.get('look_target'))
    jaw = spec.get('jaw_open_degrees', 28)
    if isinstance(jaw, bool) or not isinstance(jaw, (int, float)) or not np.isfinite(jaw) or not 0 < jaw <= 45:
        raise ValueError('jaw_open_degrees must be in (0, 45]')
    if spec.get('mouth_topology', 'unknown') not in ('unknown', 'closed_surface', 'open_mouth'):
        raise ValueError('mouth_topology must be unknown, closed_surface or open_mouth')
    controls = spec.get('controls', {})
    eyes = spec.get('visible_eyes', ['left', 'right'])
    if not isinstance(eyes, list) or not eyes or any(e not in ('left', 'right') for e in eyes) or len(set(eyes)) != len(eyes):
        raise ValueError('visible_eyes must list unique left/right eyes')
    if not isinstance(controls, dict) or 'neck' not in controls or set(controls)-set(CONTROLS):
        raise ValueError('face controls need neck and only known semantic controls')
    # Stable parent-first order, independent of JSON ordering.
    spec['controls'] = {key: controls[key] for key in CONTROLS if key in controls}
    for name, control in spec['controls'].items():
        if not isinstance(control, dict) or set(control)-{'pivot', 'region', 'ramp', 'rigid_nodes'}:
            raise ValueError(f'{name}: unknown control field')
        rig.vector(control.get('pivot'))
        region = control.get('region', {})
        if not isinstance(region, dict):
            raise ValueError(f'{name}: region must be an object')
        nodes = region.get('nodes')
        if not isinstance(nodes, list) or not nodes or not all(isinstance(n, str) and n for n in nodes) or len(set(nodes)) != len(nodes):
            raise ValueError(f'{name}: region needs unique exact node names')
        if 'rigid_nodes' in control:
            rigid = control['rigid_nodes']
            if name != 'neck' or not isinstance(rigid, list) or not all(isinstance(n, str) for n in rigid) or set(rigid)-set(nodes):
                raise ValueError('neck rigid_nodes must be a list drawn from its region nodes')
        if set(region)-{'nodes', 'box', 'center', 'radius', 'inner'}:
            raise ValueError(f'{name}: unknown region field')
        if 'box' in region:
            a = np.asarray(region['box'], float)
            if a.shape != (2, 3) or not np.isfinite(a).all() or (a[1] <= a[0]).any() or set(region)&{'center', 'radius', 'inner'}:
                raise ValueError(f'{name}: invalid box region')
        elif set(region)&{'center', 'radius', 'inner'}:
            rig.vector(region.get('center'))
            if (rig.vector(region.get('radius')) <= 0).any() or not 0 <= region.get('inner', .5) < 1:
                raise ValueError(f'{name}: invalid ellipsoid region')
        if 'ramp' in control:
            a = rig.vector(control['ramp'])
            if name != 'neck' or a[0] not in (0, 1, 2) or a[1] >= a[2]:
                raise ValueError('only neck supports ramp [axis_index, start, end]')
    required = spec.get('required', [])
    if not isinstance(required, list) or any(s not in STATES for s in required):
        raise ValueError('required must list face state names')
    gates = spec.get('gates', {})
    if not isinstance(gates, dict) or set(gates)-{'max_edge_ratio', 'stretch_area_max'}:
        raise ValueError('unknown face deformation gate')
    for name, value in gates.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError('face gates must be finite and nonnegative')
    preview = spec.get('preview', {})
    if not isinstance(preview, dict) or set(preview)-{'aim', 'span'}:
        raise ValueError('unknown face preview field')
    if preview:
        rig.vector(preview.get('aim'))
        if not isinstance(preview.get('span'), (int, float)) or not np.isfinite(preview['span']) or preview['span'] <= 0:
            raise ValueError('preview span must be positive')
    rig.contract(rig_contract(spec))
    return spec


def rig_contract(spec):
    return {'fps': spec.get('fps', 24), 'max_influences': 4,
            'bones': {'required': ['face_'+c for c in spec['controls']], 'max': 9},
            'clips': {s: {'seconds': 2, 'loop': True, **spec.get('gates', {})} for s in STATES}}


def inspect(source, spec):
    spec = contract(spec)
    doc, blob = glb.read_glb(source)
    if doc.get('skins') or doc.get('animations') or doc.get('extensionsRequired') or len(doc.get('buffers', [])) != 1:
        raise ValueError('face stage needs a static uncompressed single-buffer GLB; use the original modelling export')
    if any('sparse' in a for a in doc.get('accessors', [])) or any(p.get('targets') for m in doc['meshes'] for p in m['primitives']):
        raise ValueError('face stage does not support sparse accessors or existing morph targets')
    reserved = {'face_root'} | {'face_'+c for c in CONTROLS}
    if any(n.get('name') in reserved for n in doc['nodes']):
        raise ValueError('source uses reserved face joint names')
    model, data = rig.model_data(doc, blob)
    names = [n['name'] for n in model['nodes']]
    if len(set(names)) != len(names):
        raise ValueError('face binding requires unique mesh node names')
    counts = {key: 0 for key in spec['controls']}
    maximum = dict.fromkeys(counts, 0.)
    for key, control in spec['controls'].items():
        missing = set(control['region']['nodes'])-set(names)
        if missing:
            raise ValueError(f'{key}: missing nodes {sorted(missing)}')
    # Count actual exported influences after four-weight pruning, not selected boxes alone.
    bones = [b['name'] for b in face_motion.make_skeleton(spec)]
    for _, name, prims, _, _ in data:
        for _, points in prims:
            j, w = rig.normalize_weights(face_motion.make_weights(spec, name, points), bones, len(points), 4)
            for key in counts:
                weights = np.where(j == bones.index('face_'+key), w, 0).sum(axis=1)
                counts[key] += int((weights > 1e-6).sum())
                maximum[key] = max(maximum[key], float(weights.max()))
    # Neck may have no direct weights when all its geometry follows child joints.
    if any(count == 0 for key, count in counts.items() if key != 'neck'):
        raise ValueError('face controls bind no vertices: '+', '.join(k for k, n in counts.items() if not n and k != 'neck'))
    if not any(counts.values()):
        raise ValueError('face binding selects no vertices')
    have = set(counts)
    needs = {'look_at': {'neck'} | {'eye_'+e[0] for e in spec.get('visible_eyes', ['left','right'])}, 'talk': {'neck', 'jaw'}, 'smile': {'mouth_l', 'mouth_r'},
             'anger': {'brow_l', 'brow_r'}, 'sadness': {'mouth_l', 'mouth_r', 'brow_l', 'brow_r'},
             'confusion': {'neck', 'brow_l', 'brow_r'}, 'yell': {'jaw', 'brow_l', 'brow_r'}}
    capabilities = {}
    for state, needed in needs.items():
        missing = sorted(needed-have)
        reasons = ['missing '+s for s in missing]
        if state in ('talk', 'yell') and spec.get('mouth_topology', 'unknown') != 'open_mouth':
            reasons.append('no authored open mouth declared; jaw motion cannot create a mouth cavity')
        capabilities[state] = {'status': 'bound' if not reasons else 'partial' if needed & have else 'unavailable', 'limitations': reasons}
    issues = [f'{s}: required capability is {capabilities[s]["status"]}' for s in spec.get('required', []) if capabilities[s]['status'] != 'bound']
    return {'version': 1, 'issues': issues, 'controls': {k: {'weighted_vertices': counts[k], 'max_weight': maximum[k]} for k in counts},
            'capabilities': capabilities, 'mouth_topology_declared': spec.get('mouth_topology', 'unknown'),
            'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'notes': ['Bound means channels and vertex influences exist, not anatomical or visual approval.',
                      'Open-mouth topology is an author declaration, not automatically certified.',
                      'No audio lip sync, new topology, body animation merge or collision certification.']}


def script_text(spec):
    return (Path(face_motion.__file__).read_text(encoding='utf-8')+'\nCONFIG = '+repr(spec)+
            '\ndef skeleton(model):\n    return make_skeleton(CONFIG)\n'
            '\ndef weights(name, position, piece):\n    return make_weights(CONFIG, name, position)\n'
            '\ndef pose(clip, t):\n    return pose_at(CONFIG, clip, t)\n')


def check(source, rigged, spec):
    spec = contract(spec)
    report = inspect(source, spec)
    result = rig.check(source, rigged, rig_contract(spec))
    report['rig'] = result
    report['issues'] += result['issues']
    doc, blob = glb.read_glb(rigged)
    clip_names = [a.get('name') for a in doc.get('animations', [])]
    if len(clip_names) != len(STATES) or set(clip_names) != set(STATES):
        report['issues'].append('face export must contain exactly the seven default clips')
    source_doc, source_blob = glb.read_glb(source)
    _, source_data = rig.model_data(source_doc, source_blob)
    exported = list(glb.mesh_nodes(doc))
    bones = face_motion.make_skeleton(spec)
    ordered_names = [b['name'] for b in bones]
    # A file with normalized but wrong weights must not pass the face contract.
    if len(source_data) == len(exported):
        for (_, name, prims, _, _), (ni, _) in zip(source_data, exported):
            node = doc['nodes'][ni]
            actual_names = [doc['nodes'][j].get('name') for j in doc['skins'][node['skin']]['joints']]
            if actual_names != ordered_names:
                report['issues'].append('face skin joint order differs from contract')
                continue
            actual_prims = doc['meshes'][node['mesh']]['primitives']
            if len(prims) != len(actual_prims):
                report['issues'].append('face primitive layout differs from source')
                continue
            for (_, points), prim in zip(prims, actual_prims):
                j, w = rig.normalize_weights(face_motion.make_weights(spec, name, points), ordered_names, len(points), 4)
                a = prim['attributes']
                if not np.array_equal(j, glb.accessor(doc, blob, a['JOINTS_0'])) or not np.allclose(w, glb.accessor(doc, blob, a['WEIGHTS_0']), atol=1e-7):
                    report['issues'].append(f'{name}: facial weights differ from contract')
    # Verify the runtime/export contract at endpoints and interior samples.
    expected_names = set(ordered_names)
    nodes = {n.get('name'): i for i, n in enumerate(doc['nodes']) if n.get('name') in expected_names}
    if set(nodes) != expected_names or sum(n.get('name') in expected_names for n in doc['nodes']) != len(expected_names):
        report['issues'].append('missing face runtime joints')
    else:
        parents = {child: i for i, n in enumerate(doc['nodes']) for child in n.get('children', [])}
        for bone in bones:
            parent = bone['parent']
            expected_parent = nodes[parent] if parent else None
            if parents.get(nodes[bone['name']]) != expected_parent:
                report['issues'].append(f'{bone["name"]}: face hierarchy differs from contract')
            pivot = rig.to_gltf(bone['head'])
            if parent:
                pivot -= rig.to_gltf(next(b['head'] for b in bones if b['name'] == parent))
            expected_local = glb.node_matrix({'translation': pivot.tolist()})
            if not np.allclose(glb.node_matrix(doc['nodes'][nodes[bone['name']]]), expected_local, atol=1e-7):
                report['issues'].append(f'{bone["name"]}: rest pivot differs from contract')
        for name, ni in nodes.items():
            if name not in ('face_root', 'face_neck') and parents.get(ni) != nodes['face_neck']:
                report['issues'].append(f'{name}: face control is not a neck child')
        for state in STATES:
            if state not in result['clips']:
                continue
            for t in (0., .5, 1., 1.5, 2.):
                expected = face_motion.pose_at(spec, state, t)
                actual = skin.sample(doc, blob, state, t)
                for name, delta in expected.items():
                    ni = nodes[name]
                    parent = parents.get(ni)
                    local = np.linalg.inv(actual[parent]) @ actual[ni] if parent is not None else actual[ni]
                    base = glb.node_matrix(doc['nodes'][ni])
                    wanted = glb.node_matrix({'translation': (base[:3, 3]+rig.to_gltf(delta.get('move', [0, 0, 0]))).tolist(),
                                             'rotation': rig.quaternion(delta.get('rotate', [])).tolist()})
                    # At sampled keys these should agree exactly; lower FPS interpolation has bounded approximation.
                    if not np.allclose(local, wanted, atol=.03):
                        report['issues'].append(f'{state}: exported controller differs at {t}s')
    report['rig_sha256'] = hashlib.sha256(Path(rigged).read_bytes()).hexdigest()
    report['contract'] = spec
    report['issues'] = sorted(set(report['issues']))
    return report


def build(source, spec, out):
    source, out = Path(source), Path(out)
    spec = contract(spec)
    if source.resolve() == out.resolve():
        raise ValueError('face output must not overwrite source')
    report = inspect(source, spec)
    if report['issues']:
        raise rig.ContractError(report)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='face-', dir=out.parent) as directory:
        script, candidate = Path(directory)/'rig.py', Path(directory)/'face.glb'
        script.write_text(script_text(spec), encoding='utf-8')
        rig.build(source, script, rig_contract(spec), candidate)
        report = check(source, candidate, spec)
        if report['issues']:
            raise rig.ContractError(report)
        candidate.replace(out)
        out.with_suffix('.face-rig.py').write_text(script.read_text(encoding='utf-8'), encoding='utf-8')
    out.with_suffix('.face.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


def preview(rigged, spec, out):
    from PIL import Image, ImageDraw
    from . import rig_look
    spec = contract(spec)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    doc, blob = glb.read_glb(rigged)
    model, _ = rig.model_data(doc, blob)
    shot = rig_look.shot('front', model['lo'], model['hi'], **spec.get('preview', {}))
    sheet = Image.new('RGB', (320*4, 350*2), (245, 245, 245))
    for k, state in enumerate(('rest',)+STATES):
        image = rig_look.draw(skin.posed(rigged, None if state == 'rest' else state, .5), shot, 320)
        sheet.paste(image, ((k % 4)*320, (k//4)*350))
        ImageDraw.Draw(sheet).text(((k % 4)*320+8, (k//4)*350+325), state, fill=(20, 20, 20))
    sheet.save(out/'expressions.png')
    for state in ('talk', 'look_at'):
        frames = [rig_look.draw(skin.posed(rigged, state, float(t)), shot, 320) for t in np.arange(0, 2, .125)]
        frames[0].save(out/f'{state}.gif', save_all=True, append_images=frames[1:], duration=125, loop=0)
    return out/'expressions.png'


def command(a):
    spec = json.loads(a.contract.read_text(encoding='utf-8'))
    if a.cmd == 'face-preview':
        print(preview(a.rig, spec, a.out))
        return 0
    try:
        report = inspect(a.source, spec) if a.cmd == 'face-info' else build(a.source, spec, a.out) if a.cmd == 'face-build' else check(a.source, a.rig, spec)
    except rig.ContractError as error:
        report = error.report
    print(json.dumps(report, indent=2, allow_nan=False))
    return 1 if report['issues'] else 0


def add_commands(subs):
    for name in ('face-info', 'face-build', 'face-check', 'face-preview'):
        parser = subs.add_parser(name, help='optional explicitly bound facial rig and default expressions')
        parser.add_argument('--contract', type=Path, required=True)
        if name != 'face-preview':
            parser.add_argument('--source', type=Path, required=True)
        if name in ('face-build', 'face-preview'):
            parser.add_argument('--out', type=Path, required=True)
        if name in ('face-check', 'face-preview'):
            parser.add_argument('--rig', type=Path, required=True)
        parser.set_defaults(fn=command)
