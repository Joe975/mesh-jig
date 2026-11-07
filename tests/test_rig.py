"""Frozen mesh, known joint motions and refusal cases. No Blender or model calls."""
import json

import numpy as np
import pytest

from glbkit import box, write_glb
from mesh_jig import cli, glb, rig, skin, script_check

SCRIPT = '''import math
import numpy as np
def skeleton(model):
    return [dict(name="root", head=(0,0,0), tail=(0,0,.1), parent=None),
            dict(name="hinge", head=(1,0,0), tail=(1,0,.1), parent="root")]
def weights(name, position, piece):
    return "hinge"
def pose(clip, t):
    return {"hinge": {"rotate": [("Z", 180*t)]}}
'''
SPEC = {'fps': 4, 'max_influences': 4, 'bones': {'required': ['root', 'hinge'], 'max': 2},
        'clips': {'turn': {'seconds': 1, 'loop': False}}, 'rigid': True}


@pytest.fixture
def built(tmp_path):
    # Nonidentity source mesh transform must be present in inverse binds.
    source = write_glb(tmp_path/'source.glb', [('part', box((.4,.6,.8)), (2,.3,.2))])
    script, out = tmp_path/'rig.py', tmp_path/'rig.glb'
    script.write_text(SCRIPT)
    report = rig.build(source, script, SPEC, out)
    return source, script, out, report


def test_frozen_accessors_bind_and_known_world_axis_rotation(built):
    source, script, out, report = built
    assert report['issues'] == []
    assert report['rest_position_error_m'] == 0 and report['rest_colour_error'] == 0
    assert report['bind_error_m'] < 1e-6
    original = skin.posed(out).tris
    pivot = np.array([1,0,0])
    # Blender Z rotation is about glTF +Y. At 90 degrees x points toward -Z.
    rotation = np.array([[0,0,1], [0,1,0], [-1,0,0]])
    np.testing.assert_allclose(skin.posed(out, 'turn', .5).tris, (original-pivot) @ rotation.T+pivot, atol=1e-6)
    before, binary = glb.read_glb(source)
    after, augmented = glb.read_glb(out)
    assert augmented[:len(binary)] == binary
    for i, acc in enumerate(before['accessors']):
        assert after['accessors'][i] == acc
    assert out.with_suffix('.rig.json').exists()


def test_slerp_off_key_and_shortest_path():
    a, b = [0,0,0,1], [0,0,1,0]
    np.testing.assert_allclose(skin.slerp(a, b, .25), [0,0,np.sin(np.pi/8),np.cos(np.pi/8)])
    np.testing.assert_allclose(skin.slerp(a, -np.array(b), .25), skin.slerp(a, b, .25)*np.array([1,1,-1,1]))
    np.testing.assert_allclose(skin.slerp(a, -np.array(a), .3), a)


def test_blended_weights_and_translation(tmp_path):
    source = write_glb(tmp_path/'source.glb', [('part', box((.4,.6,.8)), (0,0,0))])
    script = tmp_path/'rig.py'
    script.write_text(SCRIPT.replace('return "hinge"', 'return {"root": np.full(len(position), .25), "hinge": np.full(len(position), .75)}')
                      .replace('{"rotate": [("Z", 180*t)]}', '{"move": (0, t, 0)}'))
    out = tmp_path/'rig.glb'
    rig.build(source, script, {**SPEC, 'rigid': False}, out)
    np.testing.assert_allclose(skin.posed(out, 'turn', 1).tris, skin.posed(out).tris+np.array([0,0,-.75]), atol=1e-7)


def test_welded_seam_is_one_piece():
    pts = np.array([[0,0,0], [1,0,0], [0,1,0], [1,0,0], [1,1,0], [0,1,0], [4,0,0], [5,0,0], [4,1,0]])
    pieces = rig.components(pts, np.arange(9).reshape(-1,3))
    assert len(set(pieces)) == 2 and pieces[0] == pieces[5] != pieces[6]


@pytest.mark.parametrize('rows', [{'root': [0,0]}, {'wrong': [1,1]}, {'root': [-1,1]}, {'root': [float('nan'),1]}])
def test_invalid_weights_are_refused(rows):
    with pytest.raises(ValueError):
        rig.normalize_weights(rows, ['root'], 2, 4)


def test_weight_pruning_is_stable_and_normalized():
    j, w = rig.normalize_weights({str(k): np.ones(2) for k in range(6)}, list(map(str, range(6))), 2, 3)
    assert j.tolist() == [[0,1,2,0], [0,1,2,0]]
    np.testing.assert_allclose(w.sum(axis=1), 1)


def test_loop_failure_does_not_replace_delivery(built):
    source, script, out, _ = built
    before = out.read_bytes()
    with pytest.raises(ValueError, match='loop seam'):
        rig.build(source, script, {**SPEC, 'clips': {'turn': {'seconds': 1, 'loop': True}}}, out)
    assert out.read_bytes() == before


def test_rig_checker_cannot_weaken_model_script_checker():
    code = 'import bpy\ndef skeleton(m): pass\ndef weights(n,p,c): pass\ndef pose(c,t): pass'
    assert script_check.check_script(code, required=('skeleton','weights','pose'), allowed_imports=rig.IMPORTS)
    assert script_check.check_script(code)[-1].startswith('line 0: no top-level `def build')
    assert script_check.check_script('import bpy\ndef build(): pass') == []


def test_cli_build_check_preview(built, tmp_path):
    source, script, out, _ = built
    spec = tmp_path/'contract.json'
    spec.write_text(json.dumps(SPEC))
    assert cli.main(['rig-check', '--source', str(source), '--rig', str(out), '--contract', str(spec)]) == 0
    assert cli.main(['rig-preview', '--rig', str(out), '--clip', 'turn', '--times', '0', '.5', '1', '--out', str(tmp_path/'poses'), '--contract', str(spec)]) == 0
    assert (tmp_path/'poses'/'turn-strip.png').exists()


def test_step_and_single_sample(built):
    _, _, out, _ = built
    doc, blob = glb.read_glb(out)
    for s in doc['animations'][0]['samplers']:
        s['interpolation'] = 'STEP'
    w = skin.sample(doc, blob, 'turn', .24)
    joint = next(i for i, node in enumerate(doc['nodes']) if node.get('name') == 'hinge')
    np.testing.assert_allclose(w[joint], skin.worlds(doc)[joint], atol=1e-6)
    for s in doc['animations'][0]['samplers']:
        doc['accessors'][s['input']]['count'] = 1
        doc['accessors'][s['output']]['count'] = 1
    skin.sample(doc, blob, 'turn', 100)


def test_parent_chain_rotations_use_rest_aligned_model_axes(built):
    source, script, out, _ = built
    script.write_text(SCRIPT.replace('return {"hinge":', 'return {"root": {"rotate": [("Z", 180*t)]}, "hinge":'))
    rig.build(source, script, SPEC, out)
    original = skin.posed(out).tris
    pivot = np.array([1,0,0])
    rotate = np.array([[0,0,1], [0,1,0], [-1,0,0]])
    expected = ((original-pivot) @ rotate.T+pivot) @ rotate.T
    np.testing.assert_allclose(skin.posed(out, 'turn', .5).tris, expected, atol=1e-6)


def test_rigid_gate_refuses_a_split_welded_piece_even_when_no_clip_moves(built):
    source, script, out, _ = built
    script.write_text(SCRIPT.replace('return "hinge"', 'return {"root": (position[:,0]<2).astype(float), "hinge": (position[:,0]>=2).astype(float)}')
                      .replace('180*t', '0'))
    with pytest.raises(ValueError, match='rigid connected pieces'):
        rig.build(source, script, SPEC, out)


def test_motion_gate_catches_static_clip(built):
    source, script, out, _ = built
    script.write_text(SCRIPT.replace('180*t', '0'))
    with pytest.raises(ValueError, match='does not move enough'):
        rig.build(source, script, {**SPEC, 'clips': {'turn': {'seconds': 1, 'min_displacement_mm': 10}}}, out)


def test_blended_deformation_area_gate(built):
    source, script, out, _ = built
    script.write_text(SCRIPT.replace('return "hinge"', 'return {"root": (position[:,0]<2).astype(float), "hinge": (position[:,0]>=2).astype(float)}')
                      .replace('{"rotate": [("Z", 180*t)]}', '{"move": (0, 2*t, 0)}'))
    with pytest.raises(ValueError, match='surface area stretches'):
        rig.build(source, script, {**SPEC, 'rigid': False, 'clips': {'turn': {'seconds': 1, 'stretch_area_max': .01}}}, out)


def test_missing_clip_and_corrupted_original_colour_are_detected(built, tmp_path):
    source, _, out, _ = built
    doc, blob = glb.read_glb(out)
    doc['animations'][0]['name'] = 'wrong'
    prim = doc['meshes'][doc['nodes'][0]['mesh']]['primitives'][0]
    a = prim['attributes']['COLOR_0']
    colours = glb.accessor(doc, blob, a)
    binary = bytearray(blob)
    colours[:,0] = 0
    prim['attributes']['COLOR_0'] = rig.append_accessor(doc, binary, colours, 'VEC4')
    changed = tmp_path/'bad.glb'
    rig.write_glb(changed, doc, binary)
    issues = rig.check(source, changed, SPEC)['issues']
    assert 'missing clip turn' in issues
    assert 'frozen source attribute COLOR_0 changed' in issues


def test_rebuild_bytes_are_deterministic(built, tmp_path):
    source, script, out, _ = built
    other = tmp_path/'again.glb'
    rig.build(source, script, SPEC, other)
    assert out.read_bytes() == other.read_bytes()
