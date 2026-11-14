"""Face controls tested on known geometry, including actual exported motion."""
import json

import numpy as np
import pytest

from glbkit import box, write_glb
from mesh_jig import cli, face, face_motion, glb, rig, skin


def spec():
    controls = {'neck': {'pivot': [0, 0, 0], 'region': {'nodes': ['face']}}}
    for name, center in [('eye_l', [-.3, 0, .7]), ('eye_r', [.3, 0, .7]), ('jaw', [0, 0, .1]),
                         ('mouth_l', [-.3, 0, .2]), ('mouth_r', [.3, 0, .2]),
                         ('brow_l', [-.3, 0, 1.]), ('brow_r', [.3, 0, 1.])]:
        controls[name] = {'pivot': center, 'region': {'nodes': ['face'], 'center': center, 'radius': [.2, .3, .13], 'inner': .5}}
    return {'version': 1, 'fps': 8, 'face_scale_m': .6, 'look_target': [1, 3, 1],
            'mouth_topology': 'open_mouth', 'controls': controls}


@pytest.fixture
def source(tmp_path):
    points, faces = [], []
    for control in spec()['controls'].values():
        center = np.array(control['pivot'])
        p, f = box((.05, .05, .05), rig.to_gltf(center))
        faces.extend(np.asarray(f)+len(points))
        points.extend(p)
    path = tmp_path/'source.glb'
    write_glb(path, [('face', (np.asarray(points, np.float32), np.asarray(faces, np.uint16)), (0, 0, 0))])
    return path


def test_defaults_export_preserves_geometry_and_drives_controls(source, tmp_path):
    out = tmp_path/'face.glb'
    report = face.build(source, spec(), out)
    assert report['issues'] == []
    assert set(report['rig']['clips']) == set(face.STATES)
    assert all(c['status'] == 'bound' for c in report['capabilities'].values())
    np.testing.assert_allclose(skin.posed(out).tris, skin.posed(source).tris, atol=1e-7)
    assert report['rig']['rest_colour_error'] == 0
    assert report['rig']['clips']['yell']['max_displacement_mm'] > report['rig']['clips']['talk']['max_displacement_mm']
    for state in face.STATES:
        assert np.isfinite(skin.posed(out, state, .137).tris).all()
    np.testing.assert_allclose(skin.posed(out, 'talk', 0).tris, skin.posed(out, 'talk', 2).tris, atol=1e-7)
    # Corrupt an expression channel: face-check must notice even though ordinary rig gates pass.
    doc, blob = glb.read_glb(out)
    animation = next(a for a in doc['animations'] if a['name'] == 'smile')
    for channel in animation['channels']:
        if doc['nodes'][channel['target']['node']]['name'] == 'face_mouth_l' and channel['target']['path'] == 'translation':
            accessor = animation['samplers'][channel['sampler']]['output']
            binary = bytearray(blob)
            a = doc['accessors'][accessor]
            start = doc['bufferViews'][a['bufferView']]['byteOffset']
            binary[start:start+a['count']*12] = np.zeros((a['count'], 3), '<f4').tobytes()
            rig.write_glb(out, doc, binary)
    assert any('exported controller differs' in s for s in face.check(source, out, spec())['issues'])


def test_target_tracks_point_and_converges_eyes():
    c = face.contract(spec())
    target = np.array([.4, 3., .9])
    p = face_motion.pose_at(c, 'look_at', target=target)
    neck = p['face_neck']['rotate']
    rot = face_motion.rotation(neck[1][1], neck[0][1])
    for side in ('l', 'r'):
        e = p['face_eye_'+side]['rotate']
        origin = rot @ np.array(c['controls']['eye_'+side]['pivot'])
        gaze = rot @ face_motion.rotation(e[1][1], e[0][1]) @ [0, 1, 0]
        wanted = target-origin
        np.testing.assert_allclose(gaze, wanted/np.linalg.norm(wanted), atol=1e-7)
    assert p != face_motion.pose_at(c, 'look_at', target=[-.4, 3, .9])
    assert face_motion.pose_at(c, 'look_at', target=[0, 0, 0])
    assert face_motion.pose_at(c, 'look_at', target=[0, -5, 0])


def test_missing_capabilities_are_honest_and_required_failure_preserves_delivery(source, tmp_path):
    c = spec()
    del c['controls']['eye_l']
    c['mouth_topology'] = 'closed_surface'
    report = face.inspect(source, c)
    assert report['capabilities']['look_at']['status'] == 'partial'
    assert report['capabilities']['yell']['status'] == 'partial'
    c['required'] = ['yell']
    out = tmp_path/'face.glb'
    out.write_bytes(b'old delivery')
    with pytest.raises(rig.ContractError):
        face.build(source, c, out)
    assert out.read_bytes() == b'old delivery'
    c = spec()
    c['gates'] = {'max_edge_ratio': .9}
    with pytest.raises(rig.ContractError):
        face.build(source, c, out)
    assert out.read_bytes() == b'old delivery'


def test_overlapping_weights_respect_neck_blend():
    c = spec()
    c['controls']['neck']['ramp'] = [2, 0, 2]
    for name in ('jaw', 'eye_l'):
        c['controls'][name]['region'] = {'nodes': ['face']}
    values = face_motion.make_weights(c, 'face', np.array([[0, 0, 1.]]))
    assert sum(v[0] for v in values.values()) == pytest.approx(1)
    assert values['face_root'][0] == pytest.approx(.5)
    c['controls']['neck']['rigid_nodes'] = ['face']
    values = face_motion.make_weights(c, 'face', np.array([[0, 0, 1.]]))
    assert sum(v[0] for v in values.values()) == pytest.approx(1)
    assert values['face_root'][0] == 0


@pytest.mark.parametrize('change', [
    {'version': 2}, {'face_scale_m': float('nan')}, {'look_target': [0, 1]}, {'fps': 0},
    {'jaw_open_degrees': 90}, {'required': ['blink']}, {'surprise': 1}, {'mouth_topology': 'magic'},
    {'gates': {'stretch_area_max': -1}}, {'preview': {'aim': [0,0,0], 'span': 0}},
    {'gates': []}, {'preview': []}, {'controls': {'neck': None}},
    {'controls': {'neck': {'pivot': [0,0,0], 'region': []}}}
])
def test_invalid_contract(change):
    with pytest.raises(ValueError):
        face.contract({**spec(), **change})


def test_bad_binding_and_animated_source_refused(source, tmp_path):
    c = spec()
    c['controls']['eye_l']['region']['nodes'] = ['absent']
    with pytest.raises(ValueError, match='missing nodes'):
        face.inspect(source, c)
    c = spec()
    c['controls']['eye_l']['region']['center'] = [50, 50, 50]
    with pytest.raises(ValueError, match='bind no vertices'):
        face.inspect(source, c)
    out = tmp_path/'face.glb'
    face.build(source, spec(), out)
    with pytest.raises(ValueError, match='static'):
        face.inspect(out, spec())
    with pytest.raises(ValueError, match='overwrite'):
        face.build(source, spec(), source)


def test_checker_refuses_changed_bindings(source, tmp_path):
    out = tmp_path/'face.glb'
    face.build(source, spec(), out)
    doc, blob = glb.read_glb(out)
    binary = bytearray(blob)
    ni, _ = next(iter(glb.mesh_nodes(doc)))
    primitive = doc['meshes'][doc['nodes'][ni]['mesh']]['primitives'][0]
    accessor = doc['accessors'][primitive['attributes']['JOINTS_0']]
    start = doc['bufferViews'][accessor['bufferView']]['byteOffset']
    binary[start:start+accessor['count']*8] = np.zeros((accessor['count'], 4), '<u2').tobytes()
    rig.write_glb(out, doc, binary)
    assert any('facial weights differ' in s for s in face.check(source, out, spec())['issues'])


def test_reserved_names_refused(source):
    doc, blob = glb.read_glb(source)
    doc['nodes'].append({'name': 'face_neck'})
    rig.write_glb(source, doc, bytearray(blob))
    with pytest.raises(ValueError, match='reserved'):
        face.inspect(source, spec())


def test_cli_and_preview(source, tmp_path):
    config = tmp_path/'face.json'
    config.write_text(json.dumps(spec()))
    out = tmp_path/'animated.glb'
    args = ['--source', str(source), '--contract', str(config)]
    assert cli.main(['face-info', *args]) == 0
    assert cli.main(['face-build', *args, '--out', str(out)]) == 0
    assert cli.main(['face-check', *args, '--rig', str(out)]) == 0
    assert cli.main(['face-preview', '--rig', str(out), '--contract', str(config), '--out', str(tmp_path/'preview')]) == 0
    assert (tmp_path/'preview'/'expressions.png').is_file()


def test_runtime_intensity_and_expressions():
    c = face.contract(spec())
    for state in face.STATES:
        p = face_motion.pose_at(c, state, .123, intensity=0)
        for delta in p.values():
            assert all(degrees == 0 for _, degrees in delta.get('rotate', []))
            assert all(v == 0 for v in delta.get('move', []))
    smile = face_motion.pose_at(c, 'smile')
    sad = face_motion.pose_at(c, 'sadness')
    assert smile['face_mouth_l']['move'][2] > 0 > sad['face_mouth_l']['move'][2]
    anger = face_motion.pose_at(c, 'anger')
    # In Blender axes, +Y rotation lowers the inner end of the left brow.
    assert anger['face_brow_l']['rotate'][0][1] > 0 > anger['face_brow_r']['rotate'][0][1]
    assert sad['face_brow_l']['rotate'][0][1] < 0 < sad['face_brow_r']['rotate'][0][1]
    with pytest.raises(ValueError):
        face_motion.pose_at(c, 'talk', float('nan'))
    with pytest.raises(ValueError):
        face_motion.pose_at(c, 'look_at', target=[float('inf'),0,0])
