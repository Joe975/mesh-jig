"""The game-ready character stage on bodies built in-test with known answers: a biped, a four-legged body, a chain
limb with no legs and a body with no limbs; each defect fails its own gate, reading or refusal."""
import copy
import json
import math
import shlex
from pathlib import Path

import numpy as np
import pytest

from glbkit import box, write_glb
from mesh_jig import cli, game, game_measure, game_package, rig, terrain

ROOT = Path(__file__).resolve().parents[1]
HIP_Z, KNEE_Z, ANKLE_Z, DROP = .9, .5, .06, .1


def part(name, size, centre):
    """A box mesh node from Blender size and centre (glTF stores Y up)."""
    sx, sy, sz = size
    cx, cy, cz = centre
    return (name, box((sx, sz, sy)), (cx, cz, -cy))


# A rig script for a body on any number of legs: each leg a thigh, shin and foot solved for the foot's place in its
# own sagittal plane, the body held DROP below rest. Stance moves the foot back at stride/(duty*seconds); swing is a
# Hermite curve whose end speeds match stance, lifted by lift*sin(pi*v), so a planted foot never jumps speed.
LEGGED = '''import math
HIP_Z, KNEE_Z, ANKLE_Z, DROP = {hip}, {knee}, {ankle}, {drop}
L1, L2 = HIP_Z-KNEE_Z, KNEE_Z-ANKLE_Z
LEGS = {legs}
GAITS = {gaits}
HIP_JOINTS = {hipj}
TOE_JOINTS = {toej}
def skeleton(model):
    out = [dict(name='root', head=(0, 0, 0), tail=(0, 0, .1), parent=None),
           dict(name='hips', head=(0, 0, HIP_Z), tail=(0, 0, HIP_Z+.1), parent='root'),
           dict(name='spine', head=(0, 0, HIP_Z+.1), tail=(0, 0, HIP_Z+.4), parent='hips'),
           dict(name='head', head=(0, 0, HIP_Z+.45), tail=(0, 0, HIP_Z+.6), parent='spine')]
    for leg, x, y in LEGS:
        top = 'hips'
        if HIP_JOINTS:        # a short joint above each thigh, for four-joint chain legs
            out.append(dict(name='hip.'+leg, head=(x, y, HIP_Z+.08), tail=(x, y, HIP_Z), parent='hips'))
            top = 'hip.'+leg
        out += [dict(name='thigh.'+leg, head=(x, y, HIP_Z), tail=(x, y, KNEE_Z), parent=top),
                dict(name='shin.'+leg, head=(x, y, KNEE_Z), tail=(x, y, ANKLE_Z), parent='thigh.'+leg),
                dict(name='foot.'+leg, head=(x, y, ANKLE_Z), tail=(x, y+.1, ANKLE_Z), parent='shin.'+leg)]
        if TOE_JOINTS:        # a joint at the front of the foot, so a turn of the foot moves a joint
            out.append(dict(name='toe.'+leg, head=(x, y+.16, ANKLE_Z-.03), tail=(x, y+.2, ANKLE_Z-.03), parent='foot.'+leg))
    return out
def weights(name, position, piece):
    return 'foot.'+name[4:] if name.startswith('toe.') else {{'body': 'hips', 'head': 'head'}}.get(name, name)
def foot(u, g):
    d, s = g['duty'], g['stride']
    if u < d:
        q = u/d
        if g.get('skate'):
            return s/2-s*q*q, 0.0
        if g.get('roll'):           # the last fifth of stance at a quarter of the speed: the median is the rest
            return s/2-s/.85*(min(q, .8)+max(0.0, q-.8)/4), 0.0
        sink = g.get('sink', 0)*math.sin(math.pi*(q-.2)/.2)**2 if .2 <= q <= .4 else 0.0
        return s/2-s*q, -sink
    v = (u-d)/(1-d)
    m = -s*(1-d)/d
    y = -s/2*(2*v**3-3*v**2+1)+m*(v**3-2*v**2+v)+s/2*(-2*v**3+3*v**2)+m*(v**3-v**2)
    return y, g['lift']*(math.sin(2*math.pi*v)**2 if g.get('graze') else math.sin(math.pi*v))
def ik(dy, dz):
    r = math.hypot(dy, dz)
    phi = math.atan2(dy, -dz)
    a = math.acos(max(-1, min(1, (L1*L1+r*r-L2*L2)/(2*L1*r))))
    k = math.acos(max(-1, min(1, (L1*L1+L2*L2-r*r)/(2*L1*L2))))
    thigh, shin = math.degrees(phi+a), -math.degrees(math.pi-k)
    return thigh, shin, -(thigh+shin)
def pose(clip, t):
    g = GAITS[clip]
    h = -DROP+g.get('bob', 0)*math.sin(2*math.pi*t/g['seconds'])
    out = {{'hips': {{'move': (0, 0, h)}}}}
    out['root'] = {{'move': (0, g.get('travel', 0)*t, g.get('jump', 0)*math.sin(math.pi*t/g['seconds']))}}
    if g.get('slash'):
        out['spine'] = [('X', g['slash']*math.sin(math.pi*t/g['seconds']))]
    if g.get('salute'):
        out['spine'] = [('X', g['salute'])]
    for leg, x, y in LEGS:
        fy, fz = foot((t/g['seconds']-g['phase'][leg]) % 1, g) if 'duty' in g else (0.0, 0.0)
        a, b, c = ik(fy, fz-(HIP_Z+h-ANKLE_Z))
        out['thigh.'+leg], out['shin.'+leg], out['foot.'+leg] = [('X', a)], [('X', b)], [('X', c)]
    return out
'''

BIPED_LEGS = [('L', -.15, 0.), ('R', .15, 0.)]
WALK = {'seconds': 1.0, 'duty': .6, 'stride': .36, 'lift': .1, 'phase': {'L': .1, 'R': .6}}
RUN = {'seconds': .6, 'duty': .4, 'stride': .36, 'lift': .12, 'phase': {'L': .25/.6, 'R': .25/.6+.5}}
BIPED_GAITS = {'idle': {'seconds': 2.0, 'bob': .01}, 'walk': WALK, 'run': RUN,
               'jump': {'seconds': .8, 'jump': .3}, 'slash': {'seconds': .6, 'slash': 30},
               'salute': {'seconds': .8, 'salute': 35}}
QUAD_LEGS = [('LF', -.2, .3), ('RF', .2, .3), ('LH', -.2, -.3), ('RH', .2, -.3)]
QUAD_WALK = {'seconds': 1.2, 'duty': .75, 'stride': .3, 'lift': .08, 'phase': {'LH': 0, 'LF': .25, 'RH': .5, 'RF': .75}}
QUAD_RUN = {'seconds': .8, 'duty': .5, 'stride': .3, 'lift': .1, 'phase': {'LH': .1, 'LF': .35, 'RH': .6, 'RF': .85}}
QUAD_PACE = {**QUAD_RUN, 'phase': {'LH': .1, 'RH': .35, 'LF': .6, 'RF': .85}}


def script(legs, gaits, hipj=False, toej=False):
    return LEGGED.format(hip=HIP_Z, knee=KNEE_Z, ankle=ANKLE_Z, drop=DROP, legs=repr(legs), gaits=repr(gaits),
                         hipj=repr(hipj), toej=repr(toej))


def legged(tmp, legs, gaits, body_size, name='animated.glb', toes=False, hipj=False, toej=False):
    parts = [part('body', body_size, (0, 0, HIP_Z+.15)), part('head', (.2, .2, .2), (0, 0, HIP_Z+.55))]
    parts += [part('foot.'+leg, (.12, .24, .06), (x, y+.04, .03)) for leg, x, y in legs]
    if toes:      # 6 cm ahead of the sole, from 15 to 45 mm up: no part of the contact
        parts += [part('toe.'+leg, (.1, .06, .03), (x, y+.19, .03)) for leg, x, y in legs]
    source = write_glb(tmp/('source-'+name), parts)
    path = tmp/('rig-'+name+'.py')
    path.write_text(script(legs, gaits, hipj, toej))
    clips = {c: {'seconds': g['seconds'], 'loop': c not in ('jump', 'slash', 'salute')} for c, g in gaits.items()}
    out = tmp/name
    rig.build(source, path, {'fps': 60, 'rigid': True, 'max_influences': 1, 'loop_mm_max': 1e6, 'clips': clips}, out)
    return out


def leg(name, x, y):
    return {'name': 'leg.'+name, 'role': 'leg', 'joints': ['thigh.'+name, 'shin.'+name, 'foot.'+name],
            'solver': 'two_bone', 'pole': [0, 1, 0], 'bend_degrees': [3, 175],
            'contact': {'node': 'foot.'+name, 'box': [[x-.07, y-.09, -.001], [x+.07, y+.17, .001]]}}


def biped_contract(path):
    return {
        'version': 1, 'rig_sha256': terrain.digest(path) if path else '0'*64, 'forward': [0, 1, 0],
        'parameters': {'speed': {'min': 0, 'max': 2, 'default': 0}},
        'graph': {'layers': [
            {'name': 'base', 'node': {'type': 'blend1d', 'parameter': 'speed', 'sync': True, 'points': [
                {'clip': 'idle', 'value': 0}, {'clip': 'walk', 'value': 'measured'}, {'clip': 'run', 'value': 'measured'}]}},
            {'name': 'jump', 'node': {'type': 'states', 'start': 'off', 'states': {'off': None, 'jump': {'type': 'clip', 'clip': 'jump'}},
                                      'transitions': [{'from': 'off', 'to': 'jump', 'on': 'jump', 'fade_s': .1},
                                                      {'from': 'jump', 'to': 'off', 'at_end': True, 'fade_s': .15}]}},
            {'name': 'upper', 'mask': ['spine'], 'node': {'type': 'oneshots', 'clips': {
                'slash': {'fade_in_s': .08, 'fade_out_s': .2}, 'salute': {'fade_in_s': .2, 'fade_out_s': .2}}}}]},
        'body': {'joint': 'hips'},
        'limbs': [leg(n, x, y) for n, x, y in BIPED_LEGS],
        'ground': {'drop_m_max': .25, 'rise_m_max': .15, 'tilt': False, 'relock_m': .12, 'blend_s': .08,
                   'lookahead_s': .15},
        'aims': [{'name': 'head', 'joints': ['spine', 'head'], 'axis': [0, 1, 0], 'yaw_degrees': [-70, 70],
                  'pitch_degrees': [-40, 40], 'degrees_per_s': 240}],
        'sockets': [{'name': 'crest', 'joint': 'head', 'offset': [0, .1, 0]}],
        'criteria': {'loop_seam_mm_max': 2, 'contact_gap_mm_max': 2, 'root_travel_mm_max': 5, 'skate_mps_max': .05,
                     'gait_phase_difference_max': .05, 'blend_slip_mps_max': .1, 'fade_joint_speed_mps_max': 5,
                     'stance_s_min': .05}}


def quad_contract(path, legs=QUAD_LEGS):
    return {'version': 1, 'rig_sha256': terrain.digest(path),
            'parameters': {'speed': {'min': 0, 'max': 2, 'default': 0}},
            'graph': {'layers': [{'name': 'base', 'node': {'type': 'blend1d', 'parameter': 'speed', 'sync': True, 'points': [
                {'clip': 'walk', 'value': 'measured'}, {'clip': 'run', 'value': 'measured'}]}}]},
            'body': {'joint': 'hips'}, 'limbs': [leg(n, x, y) for n, x, y in legs],
            'ground': {'drop_m_max': .2, 'rise_m_max': .1, 'tilt': True, 'relock_m': .1, 'blend_s': .1,
                       'lookahead_s': .15},
            'criteria': {'loop_seam_mm_max': 2, 'contact_gap_mm_max': 2, 'root_travel_mm_max': 5, 'skate_mps_max': .05,
                         'gait_phase_difference_max': .05, 'blend_slip_mps_max': .1, 'stance_s_min': .05}}


@pytest.fixture(scope='module')
def biped(tmp_path_factory):
    return legged(tmp_path_factory.mktemp('biped'), BIPED_LEGS, BIPED_GAITS, (.4, .25, .3))


@pytest.fixture(scope='module')
def biped_build(biped, tmp_path_factory):
    out = tmp_path_factory.mktemp('biped-build')/'package'
    return game_package.build(biped, biped_contract(biped), out), out


@pytest.fixture(scope='module')
def quad(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('quad')
    return (legged(tmp, QUAD_LEGS, {'walk': QUAD_WALK, 'run': QUAD_RUN}, (.5, .8, .25)),
            legged(tmp, QUAD_LEGS, {'walk': QUAD_WALK, 'run': QUAD_PACE}, (.5, .8, .25), 'pace.glb'))


def build(path, spec, tmp, name='package'):
    return game_package.build(path, spec, tmp/name)


def test_biped_measures_known_speed_duty_and_phase(biped_build):
    report, _ = biped_build
    assert report['passed'], report['issues']
    clips = report['measured']['clips']
    assert clips['walk']['travel_speed_mps'] == pytest.approx(.6, rel=.01)
    assert clips['run']['travel_speed_mps'] == pytest.approx(1.5, rel=.01)
    assert abs(clips['idle']['travel_speed_mps']) < .005
    for side in ('leg.L', 'leg.R'):
        # the swing lifts by sin, so a 2 mm gap adds about half a percent of stance either side of it
        assert clips['walk']['legs'][side]['duty'] == pytest.approx(.6, abs=.015)
        assert clips['run']['legs'][side]['duty'] == pytest.approx(.4, abs=.015)
        assert clips['walk']['legs'][side]['spread_mps'] < .01
        assert clips['idle']['legs'][side]['duty'] == 1
    assert clips['walk']['legs']['leg.L']['touchdowns_s'][0] == pytest.approx(.1, abs=.006)
    assert clips['walk']['legs']['leg.R']['touchdowns_s'][0] == pytest.approx(.6, abs=.006)
    blend = report['measured']['blends'][0]
    assert blend['reference_leg'] == 'leg.L'
    assert blend['phase_offsets']['walk'] == pytest.approx(.1, abs=.006)
    assert blend['phase_offsets']['run'] == pytest.approx(.25/.6, abs=.01)
    assert blend['phase_offsets']['idle'] == 0
    walk_run = blend['pairs'][1]
    assert walk_run['gait_phase_difference'] < .01
    # half and half: frequencies lerp (1.333 Hz); both planted the foot moves back at 0.75 m a cycle, 1.0 m/s,
    # against the mean travel speed of 1.05 m/s
    assert walk_run['blend']['speed_mps'] == pytest.approx(1.05, rel=.01)
    assert report['gates']['blend_slip_mps']['items']['layer[0]: walk-run'] == pytest.approx(.05, abs=.01)
    assert report['gates']['blend_slip_mps']['items']['layer[0]: idle-walk'] < .02
    assert report['gates']['loop_seam_mm']['value'] < .01 and report['gates']['root_travel_mm']['value'] < .01
    assert report['readings']['missed_contacts'] == [] and report['readings']['rigid_limbs'] == []
    assert report['readings']['unused_joints'] == ['root']


def test_fades_measure_pose_distance_over_fade_time(biped_build):
    report, _ = biped_build
    fades = {f['name']: f for f in report['measured']['fades']}
    assert set(fades) == {'jump: off -> jump', 'jump: jump -> off', 'upper: slash fades in', 'upper: slash fades out',
                          'upper: salute fades in', 'upper: salute fades out'}
    # salute holds the spine 35 degrees forward: the head joint, 0.35 m up it, is 0.21 m from where it was
    assert fades['upper: salute fades in']['distance_m'] == pytest.approx(2*.35*math.sin(math.radians(17.5)), abs=.002)
    # the jump starts and ends at the idle pose; the trigger can fire anywhere in idle's 1 cm bob
    assert fades['jump: off -> jump']['distance_m'] == pytest.approx(.01, abs=.002)
    assert fades['jump: jump -> off']['distance_m'] < .011
    # slash bends the spine 30 degrees at its middle, none at its ends
    assert fades['upper: slash fades in']['distance_m'] < .002
    assert report['gates']['fade_joint_speed_mps']['passed']


def test_a_snap_fade_fails_its_gate(biped, tmp_path):
    spec = biped_contract(biped)
    spec['graph']['layers'][1]['node']['transitions'][0]['fade_s'] = .001     # 1 cm in a millisecond: 10 m/s
    report = build(biped, spec, tmp_path)
    gate = report['gates']['fade_joint_speed_mps']
    assert not gate['passed'] and gate['worst'] == 'jump: off -> jump' and gate['value'] == pytest.approx(10, rel=.2)


def test_package_is_a_byte_copy_with_axes_converted(biped, biped_build, tmp_path):
    report, out = biped_build
    assert (out/'character.glb').read_bytes() == biped.read_bytes()
    data = json.loads((out/'character.runtime.json').read_text())
    assert report['hashes']['runtime_sha256'] == terrain.hashlib.sha256((out/'character.runtime.json').read_bytes()).hexdigest()
    assert data['axes']['forward'] == [0, 0, -1]
    assert data['measurement']['rest_ground_y'] == pytest.approx(0, abs=1e-6)      # the soles' rest height, glTF y
    joints = {j['name']: j for j in data['skeleton']['joints']}
    assert joints['foot.L']['rest_position'] == pytest.approx([-.15, ANKLE_Z, 0], abs=1e-6)
    assert joints['foot.L']['parent'] == 'shin.L'
    legs = {l['name']: l for l in data['limbs']}
    assert legs['leg.L']['pole'] == [0, 0, -1]
    assert legs['leg.L']['segment_lengths_m'] == pytest.approx([HIP_Z-KNEE_Z, KNEE_Z-ANKLE_Z])
    # the sole's corners: Blender (x +-0.06, y -0.08 or 0.16, z 0) about the ankle at (x, 0, 0.06)
    assert sorted(map(tuple, legs['leg.L']['contact']['points'])) == pytest.approx(
        sorted((x, -ANKLE_Z, -y) for x in (-.06, .06) for y in (-.08, .16)), abs=1e-6)
    assert data['sockets'][0]['offset_model'] == pytest.approx([0, 0, -.1])
    assert data['aims'][0]['axis_local'] == pytest.approx([0, 0, -1])
    walk = data['graph']['layers'][0]['node']['points'][1]
    assert walk['measured'] and walk['value'] == data['clips']['walk']['travel_speed_mps']
    assert data['graph']['layers'][0]['node']['sync_clock'] == 'frequency_lerp'
    assert [p['cycle'] for p in data['graph']['layers'][0]['node']['points']] == [False, True, True]
    assert data['graph']['layers'][2]['mask_joints'] == ['head', 'spine']
    assert data['clips']['walk']['legs']['leg.L']['stance_phase'][0][0] == pytest.approx(.1, abs=.006)
    assert data['triggers'] == ['jump'] and data['oneshots'] == ['salute', 'slash']
    assert set(data['capabilities']) >= {'locomotion', 'ground_adaptation', 'limbs', 'aims', 'not_included'}


def test_two_builds_are_byte_identical(biped, biped_build, tmp_path):
    _, first = biped_build
    build(biped, biped_contract(biped), tmp_path)
    for name in ('character.glb', 'character.runtime.json', 'character.report.json'):
        assert (tmp_path/'package'/name).read_bytes() == (first/name).read_bytes()


def test_four_legs_and_the_missed_leg(quad, tmp_path):
    walk_run, _ = quad
    report = build(walk_run, quad_contract(walk_run), tmp_path)
    assert report['passed'], report['issues']
    clips = report['measured']['clips']
    assert clips['walk']['travel_speed_mps'] == pytest.approx(.3/(.75*1.2), rel=.01)
    assert clips['run']['travel_speed_mps'] == pytest.approx(.3/(.5*.8), rel=.01)
    assert all(x['duty'] == pytest.approx(.75, abs=.015) for x in clips['walk']['legs'].values())
    order = report['measured']['blends'][0]['pairs'][0]['order']
    assert order['walk'] == order['run'] == ['leg.LF', 'leg.RH', 'leg.RF', 'leg.LH']
    missed = build(walk_run, quad_contract(walk_run, QUAD_LEGS[:3]), tmp_path, 'missed')
    joints = {m['joint'] for m in missed['readings']['missed_contacts']}
    assert joints == {'foot.RH'}
    assert {m['clip'] for m in missed['readings']['missed_contacts']} == {'walk', 'run'}


def test_incompatible_leg_order_fails_under_sync(quad, tmp_path):
    _, pace = quad
    report = build(pace, quad_contract(pace), tmp_path)
    gate = report['gates']['gait_phase_difference']
    assert not gate['passed'] and gate['value'] == pytest.approx(.5, abs=.01)
    assert not report['passed'] and not (tmp_path/'package'/'character.runtime.json').exists()
    assert not (tmp_path/'package'/'character.glb').exists() and (tmp_path/'package'/'character.report.json').is_file()


@pytest.mark.parametrize('change,gate', [
    (lambda g: g['walk'].update(skate=True), 'skate_mps'),
    (lambda g: g['walk'].update(travel=.6), 'root_travel_mm')])
def test_locomotion_defects_fail_their_gate(tmp_path, change, gate):
    gaits = copy.deepcopy(BIPED_GAITS)
    change(gaits)
    path = legged(tmp_path, BIPED_LEGS, gaits, (.4, .25, .3))
    report = build(path, biped_contract(path), tmp_path)
    assert not report['gates'][gate]['passed'] and report['gates'][gate]['worst'].startswith('walk')
    assert not report['passed']
    if gate == 'skate_mps':     # in stance the foot moves back from 0 to twice the travel speed
        assert report['measured']['clips']['walk']['legs']['leg.L']['spread_mps'] > .2


@pytest.mark.parametrize('change', ['sink', 'graze', 'roll'])
def test_ground_stance_and_speed_rules(tmp_path, change):
    gaits = copy.deepcopy(BIPED_GAITS)
    gaits['walk'][change] = {'sink': .003, 'graze': True, 'roll': True}[change]
    path = legged(tmp_path, BIPED_LEGS, gaits, (.4, .25, .3))
    report = build(path, biped_contract(path), tmp_path)
    walk = report['measured']['clips']['walk']
    if change == 'sink':
        # the foot dips 3 mm below the rest ground for a fifth of stance; the rest of stance is still planted
        assert report['passed'], report['issues']
        assert report['readings']['below_rest_ground_mm']['walk'] == pytest.approx(3, abs=.3)
        assert report['readings']['below_rest_ground_mm']['run'] < .3     # LINEAR keys at 60 fps sag a little
        assert all(x['duty'] == pytest.approx(.6, abs=.015) for x in walk['legs'].values())
    if change == 'graze':
        # mid-swing the foot comes down to the ground for about 18 ms: a graze, not a second touchdown
        assert report['passed'], report['issues']
        grazes = [g for g in report['readings']['grazes'] if g['clip'] == 'walk']
        assert [g['leg'] for g in grazes] == ['leg.L', 'leg.R']
        assert grazes[0]['time_s'] == pytest.approx(.9, abs=.02) and grazes[0]['clearance_mm'] < 1
        assert all(len(x['touchdowns_s']) == 1 for x in walk['legs'].values())
    if change == 'roll':
        # the median follows the 80% of stance at full speed; skate still reads the slow fifth
        assert walk['travel_speed_mps'] == pytest.approx(.36/(.85*.6), rel=.01)
        assert not report['gates']['skate_mps']['passed']


def test_contract_refusals(biped, tmp_path):
    def refused(change, match):
        spec = biped_contract(biped)
        change(spec)
        with pytest.raises(ValueError, match=match):
            build(biped, spec, tmp_path)
        assert not (tmp_path/'package').exists()
    refused(lambda s: s['limbs'][0].update(joints=['hips', 'shin.L', 'foot.L']), 'not a direct child')
    refused(lambda s: s['limbs'][1].update(contact=copy.deepcopy(s['limbs'][0]['contact'])), 'overlap')
    refused(lambda s: s['ground'].update(height=1), 'unknown key')
    refused(lambda s: s.update(colour='red'), 'unknown key')
    refused(lambda s: s.update(rig_sha256='0'*64), 'rig_sha256 does not match')
    refused(lambda s: s['graph']['layers'][1]['node']['states'].update(jump={'type': 'clip', 'clip': 'leap'}), 'leap')
    refused(lambda s: s['criteria'].pop('blend_slip_mps_max'), 'criteria must state exactly')
    refused(lambda s: s['limbs'][0].pop('contact'), 'needs a contact')
    refused(lambda s: s['limbs'][0]['contact'].update(node='head'), 'selects no vertices')


@pytest.mark.parametrize('change', [
    lambda s: s.update(version=2), lambda s: s.update(version=True), lambda s: s.pop('graph'),
    lambda s: s['graph']['layers'][0]['node'].update(parameter='lean'),
    lambda s: s['graph']['layers'][0]['node'].update(points=s['graph']['layers'][0]['node']['points'][:1]),
    lambda s: s['graph']['layers'][0]['node']['points'][0].update(node={'type': 'clip', 'clip': 'idle'}),
    lambda s: s['graph']['layers'][1]['node']['transitions'][0].update(at_end=True),
    lambda s: s['graph']['layers'][1]['node']['transitions'][0].update(fade_s=0),
    lambda s: s['graph']['layers'][1]['node'].update(start='land'),
    lambda s: s['graph']['layers'][2].update(mask=[]),
    lambda s: s['graph']['layers'][0]['node'].update(type='blend2d'),
    lambda s: s['limbs'][0].update(joints=['thigh.L', 'shin.L']),
    lambda s: s['limbs'][0].update(solver='chain'),
    lambda s: s['limbs'][0].update(pole=[0, 0, 0]),
    lambda s: s['limbs'][0].update(bend_degrees=[90, 10]),
    lambda s: s['ground'].update(tilt=True),
    lambda s: s.pop('body'),
    lambda s: s['aims'][0].update(pitch_degrees=[-100, 40]),
    lambda s: s.update(forward=[0, 1, 1]),
    lambda s: s['parameters']['speed'].update(default=3),
    lambda s: s['criteria'].update(skate_mps_max=-1),
    lambda s: s['criteria'].update(gait_phase_difference_max=.7),
    lambda s: s['criteria'].update(penetration_mm_max=1),
])
def test_contract_shape_is_strict(change):
    spec = biped_contract(None)
    change(spec)
    with pytest.raises(ValueError):
        game.contract(spec)


def test_one_chain_limb_and_no_legs(tmp_path):
    joints = ['a1', 'a2', 'a3', 'a4', 'a5']
    zs = [.9, .7, .5, .3, .1]
    bones = ',\n'.join(f"dict(name='{j}', head=(0, 0, {z}), tail=(0, 0, {z-.2:.1f}), parent='{p}')"
                       for j, z, p in zip(joints, zs, ['body']+joints))
    text = f'''import math
def skeleton(model):
    return [dict(name='root', head=(0, 0, 0), tail=(0, 0, .1), parent=None),
            dict(name='body', head=(0, 0, 1.0), tail=(0, 0, 1.2), parent='root'),
{bones}]
def weights(name, position, piece):
    return {{'body': 'body'}}.get(name, 'a'+name[-1])
def pose(clip, t):
    k = 4 if clip == 'wave' else 1
    return {{j: [('X', 15*math.sin(2*math.pi*t))] for j in ['a1', 'a2', 'a3', 'a4'][:k]}}
'''
    parts = [part('body', (.3, .3, .2), (0, 0, 1.0))] + [part(f'seg{k+1}', (.08, .08, .19), (0, 0, z-.1)) for k, z in enumerate(zs[:4])]
    source = write_glb(tmp_path/'source.glb', parts)
    (tmp_path/'rig.py').write_text(text)
    path = tmp_path/'arm.glb'
    rig.build(source, tmp_path/'rig.py', {'fps': 30, 'rigid': True, 'max_influences': 1,
                                           'clips': {'wave': {'seconds': 1, 'loop': True}, 'stiff': {'seconds': 1, 'loop': True}}}, path)
    spec = {'version': 1, 'rig_sha256': terrain.digest(path),
            'graph': {'layers': [{'name': 'base', 'node': {'type': 'clip', 'clip': 'wave'}}]},
            'limbs': [{'name': 'arm', 'role': 'reach', 'joints': joints, 'solver': 'chain', 'bend_degrees_max': 45}],
            'criteria': {'loop_seam_mm_max': 2}}
    report = build(path, spec, tmp_path)
    assert report['passed'] and report['readings']['rigid_limbs'] == []
    data = json.loads((tmp_path/'package'/'character.runtime.json').read_text())
    assert data['limbs'][0]['segment_lengths_m'] == pytest.approx([.2]*4)
    assert data['limbs'][0]['bend_degrees_max'] == 45 and 'contact' not in data['limbs'][0]
    assert set(report['gates']) == {'loop_seam_mm'} and data['clips']['wave']['legs'] == {}
    spec['graph']['layers'][0]['node']['clip'] = 'stiff'
    stiff = build(path, spec, tmp_path, 'stiff')
    assert [x['limb'] for x in stiff['readings']['rigid_limbs']] == ['arm']


def test_no_limbs_graph_only(tmp_path):
    text = '''import math
def skeleton(model):
    return [dict(name='root', head=(0, 0, 0), tail=(0, 0, .1), parent=None),
            dict(name='turret', head=(0, 0, .3), tail=(0, 0, .5), parent='root')]
def weights(name, position, piece):
    return 'root' if name == 'base' else 'turret'
def pose(clip, t):
    return {'turret': [('Z', 180*t)]}
'''
    source = write_glb(tmp_path/'source.glb', [part('base', (.6, .6, .2), (0, 0, .1)), part('gun', (.1, .5, .1), (0, .2, .35))])
    (tmp_path/'rig.py').write_text(text)
    path = tmp_path/'turret.glb'
    rig.build(source, tmp_path/'rig.py', {'fps': 30, 'rigid': True, 'max_influences': 1,
                                           'clips': {'spin': {'seconds': 2, 'loop': True}}}, path)
    spec = {'version': 1, 'rig_sha256': terrain.digest(path),
            'graph': {'layers': [{'name': 'base', 'node': {'type': 'clip', 'clip': 'spin'}}]},
            'aims': [{'name': 'gun', 'joints': ['turret'], 'axis': [0, 1, 0], 'yaw_degrees': [-180, 180],
                      'pitch_degrees': [-10, 10], 'degrees_per_s': 90}],
            'criteria': {'loop_seam_mm_max': 2}}
    report = build(path, spec, tmp_path)
    assert report['passed'] and report['measured']['clips']['spin']['loop_seam_mm'] < .01
    data = json.loads((tmp_path/'package'/'character.runtime.json').read_text())
    assert data['limbs'] == [] and data['ground'] is None and data['aims'][0]['axis_model'] == [0, 0, -1]
    assert report['readings']['unused_joints'] == ['root']


def test_cli_exit_codes_info_and_empty_output(biped, biped_build, tmp_path, capsys):
    contract = tmp_path/'character.json'
    contract.write_text(json.dumps(biped_contract(biped)))
    assert cli.main(['game-info', '--rig', str(biped)]) == 0
    text = capsys.readouterr().out
    assert 'foot.L' in text and 'GROUND CONTACT BY CLIP' in text and 'skate_mps' in text
    # walk's two feet are found on the ground on their own, each about 60% of the clip
    assert text.count('foot.L on foot.L') >= 3
    assert cli.main(['game-info', '--rig', str(biped), '--contract', str(contract)]) == 0
    assert 'leg.L 4 vertices' in capsys.readouterr().out
    assert cli.main(['game-build', '--rig', str(biped), '--contract', str(contract), '--out', str(tmp_path/'p')]) == 0
    assert cli.main(['game-build', '--rig', str(biped), '--contract', str(contract), '--out', str(tmp_path/'p')]) == 2
    spec = biped_contract(biped)
    spec['criteria']['skate_mps_max'] = 1e-9
    contract.write_text(json.dumps(spec))
    assert cli.main(['game-build', '--rig', str(biped), '--contract', str(contract), '--out', str(tmp_path/'q')]) == 1


def test_example_contract_and_documented_commands():
    from test_readme import commands
    game.contract(json.loads((ROOT/'examples/game-ready/character.example.json').read_text()))
    found = commands((ROOT/'docs/GAME_READY.md').read_text())
    assert found
    for line in found:
        cli.parser().parse_args(shlex.split(line)[1:])


def test_bend_ranges_are_printed_and_a_clip_outside_the_declared_range_is_read(biped, biped_build, tmp_path, capsys):
    report, _ = biped_build
    assert report['readings']['bend_outside'] == []
    assert cli.main(['game-info', '--rig', str(biped)]) == 0
    line = next(l for l in capsys.readouterr().out.splitlines() if l.strip().startswith('thigh.L > shin.L > foot.L'))
    walk = [float(x) for x in line.split('walk ')[1].split(',')[0].split('..')]
    assert 60 < walk[0] < walk[1] < 180          # the walk's knee bends and straightens
    spec = biped_contract(biped)
    spec['limbs'][0]['bend_degrees'] = [walk[0]+10, 179]
    tight = build(biped, spec, tmp_path)
    found = [x for x in tight['readings']['bend_outside'] if x['limb'] == 'leg.L' and x['clip'] == 'walk']
    assert found and found[0]['inner_degrees'][0] == pytest.approx(walk[0], abs=.5) and found[0]['joint'] == 'shin.L'


@pytest.mark.parametrize('change,expect', [('skate', 'skate'), ('sink', 'below'), ('graze', 'graze'), (None, None)])
def test_rig_readings_say_what_a_game_would_see(tmp_path, change, expect):
    from mesh_jig import rig_read
    gaits = copy.deepcopy(BIPED_GAITS)
    if change:
        gaits['walk'][change] = {'sink': .003, 'graze': True, 'skate': True}[change]
    path = legged(tmp_path, BIPED_LEGS, gaits, (.4, .25, .3))
    spec = {'fps': 60, 'clips': {c: {'seconds': g['seconds'], 'loop': c not in ('jump', 'slash', 'salute')}
                                 for c, g in gaits.items()}}
    found = rig_read.ground_contacts(path, spec)
    walk = found['walk']
    feet = {c['joint']: c for c in walk['contacts']}
    assert set(feet) == {'foot.L', 'foot.R'} and walk['travel_speed_mps'] == pytest.approx(.6, abs=.005)
    assert feet['foot.L']['touchdown_phases'] == [pytest.approx(.1, abs=.015)]     # a 5 mm gap meets it early
    assert (feet['foot.L']['skate_mps'] > .2) == (expect == 'skate')
    assert (walk['below_rest_ground_mm'] > 2.5) == (expect == 'below')
    assert bool(feet['foot.L']['grazes']) == (expect == 'graze')
    assert found['jump']['contacts'][0]['grazes'] == []          # a once-played clip's ends are no grazes
    text = '\n'.join(rig_read.contact_text(found))
    assert 'GROUND CONTACT AS A GAME WOULD USE IT' in text
    assert ('! walk goes' in text) == (expect == 'below')


def test_the_brief_says_what_makes_clips_packageable(tmp_path):
    from mesh_jig import rig_brief
    source = write_glb(tmp_path/'s.glb', [part('body', (.4, .3, .3), (0, 0, .5))])
    example = json.loads((ROOT/'examples/game-ready/locomotion.rig.json').read_text())
    rig.contract(example)
    assert rig_brief.PACKAGEABLE in rig_brief.text(source, example)
    once = {'clips': {'wave': {'seconds': 1}}}
    assert rig_brief.PACKAGEABLE not in rig_brief.text(source, once)
