"""The reference runtime on packages built in-test: ground adaptation over a slope, a step and a raise (and the
same scenarios failing with the ground off), playback, variable dt, four legs with tilt, a chain limb's targets,
an aim's limits and turn rate, determinism and the scenario format."""
import copy
import json
import math

import numpy as np
import pytest

import gamekit as K
from glbkit import write_glb
from mesh_jig import cli, game_check, game_package, game_runtime, game_scenario, rig, terrain
from test_game import (BIPED_GAITS, BIPED_LEGS, QUAD_LEGS, QUAD_RUN, QUAD_WALK, biped_contract, legged, part,
                       quad_contract)

STEP_Z = sorted(set(np.linspace(-3.5, 1, 46).round(4).tolist()+[-.6, -.6001]))
TERRAINS = {
    'slope': dict(height=lambda x, z: -.1*z),
    'step': dict(height=lambda x, z: .08*(z < -.6), zs=STEP_Z),
    'raise': dict(zs=np.linspace(-3.5, 1, 46), events=[{'time': 1.0, 'type': 'terrain_patch',
                                                        'rect': [[-1, -1.0], [1, -.5]], 'offset': .05}]),
}


def package(tmp, glb, contract):
    out = tmp/'package'
    report = game_package.build(glb, contract, out)
    assert report['passed'], report['issues']
    return out, json.loads((out/'character.runtime.json').read_text())


def play(pkg, spec):
    trace = game_runtime.run(pkg, spec)
    return trace, game_check.check(pkg, spec, trace)


@pytest.fixture(scope='module')
def biped(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('runtime-biped')
    glb = legged(tmp, BIPED_LEGS, BIPED_GAITS, (.4, .25, .3))
    return package(tmp, glb, biped_contract(glb))


@pytest.fixture(scope='module')
def turret(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('runtime-turret')
    return package(tmp, *K.turret(tmp, part, rig, terrain, write_glb))


@pytest.fixture(scope='module')
def arm(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('runtime-arm')
    return package(tmp, *K.arm(tmp, part, rig, terrain, write_glb))


def test_frames_land_on_event_times_and_the_end():
    spec = K.still(.5, dt=(.03, .05), events=[{'time': .07, 'type': 'set_ground_enabled', 'enabled': False}])
    times = game_scenario.frame_times(spec)
    assert times.tolist() == pytest.approx([0, .03, .07, .1, .15, .18, .23, .26, .31, .34, .39, .42, .47, .5])


@pytest.mark.parametrize('kind', sorted(TERRAINS))
def test_ground_adaptation_holds_and_without_it_fails(biped, kind):
    pkg, runtime = biped
    _, on = play(pkg, K.walk(runtime, **TERRAINS[kind]))
    assert on['passed'], on['issues']
    # a contact keeps the animation's clearance (up to the build's 2 mm gap at a stance edge), and a look-ahead lift
    # that fades over blend_s after touchdown adds a little on a slope: the 3 mm gate holds
    assert on['gates']['penetration_mm']['value'] < .1
    _, off = play(pkg, K.walk(runtime, ground=False, **TERRAINS[kind]))
    assert not off['gates']['contact_gap_mm']['passed'] and not off['gates']['penetration_mm']['passed']
    if kind == 'raise':
        assert on['gates']['recovery_mm']['passed'] and not off['gates']['recovery_mm']['passed']


def test_flat_playback_matches_the_clip_with_the_ground_off(biped):
    pkg, runtime = biped
    _, report = play(pkg, K.walk(runtime, ground=False))
    assert report['passed'], report['issues']
    assert report['gates']['playback_mm']['value'] < .01


def test_variable_dt_and_a_slower_root_hold(biped):
    pkg, runtime = biped
    _, report = play(pkg, K.walk(runtime, dt=(1/30, 1/90, 1/45, 1/120, 1/60), **TERRAINS['slope']))
    assert report['passed'], report['issues']
    # the root at 0.5 m/s under a 0.6 m/s walk: the latched feet do not slide
    _, report = play(pkg, K.walk(runtime, root_speed=.5))
    assert report['passed'] and report['gates']['slip_mps']['value'] < .02


def test_deterministic_trace(biped):
    pkg, runtime = biped
    spec = K.walk(runtime, duration=1.0, **TERRAINS['slope'])
    first, second = (json.dumps(game_runtime.run(pkg, spec), sort_keys=True) for _ in range(2))
    assert first == second


def test_four_legs_tilt_on_a_cross_slope(tmp_path):
    glb = legged(tmp_path, QUAD_LEGS, {'walk': QUAD_WALK, 'run': QUAD_RUN}, (.5, .8, .25))
    pkg, runtime = package(tmp_path, glb, quad_contract(glb))
    spec = K.walk(runtime, height=lambda x, z: -.08*z+.1*x, duration=2.4)
    _, report = play(pkg, spec)
    assert report['passed'], report['issues']
    _, off = play(pkg, K.walk(runtime, height=lambda x, z: -.08*z+.1*x, duration=2.4, ground=False))
    assert not off['gates']['contact_gap_mm']['passed']


def test_a_limb_target_is_reached_and_an_unreachable_one_clamped(arm):
    pkg, runtime = arm
    # the runtime follows the game's weight as given, so the game ramps it in over a second
    ramp = lambda point: [{'time': .1+.05*k, 'type': 'set_limb_target', 'name': 'arm', 'point': point,
                           'weight': round(.05*(k+1), 2)} for k in range(20)]
    spec = K.limits(K.still(2.2, events=ramp([.1, .2, -.3])), runtime)
    trace, report = play(pkg, spec)
    assert report['passed'], report['issues']
    assert report['gates']['limb_target_error_mm']['value'] < 1
    far = K.limits(K.still(2.2, events=ramp([0, .9, -1.5])), runtime, joint_speed_mps_max=20)   # straightens fast
    trace, report = play(pkg, far)
    assert report['passed'] and report['gates']['stretch_mm']['value'] < .01
    assert report['gates']['limb_target_error_mm']['value'] is None        # out of reach: not tracked
    tip = game_scenario.matrix(trace['frames'][-1]['joints'][trace['joints'].index('a5')])[:3, 3]
    assert tip == pytest.approx([0, .9, -.8], abs=.002)                       # straight at it


def test_aim_limits_and_turn_rate(turret):
    pkg, runtime = turret
    at = lambda yaw, pitch: (np.array([0, .3, 0])+3*game_scenario.aim_direction(yaw, pitch, [0, 0, -1])).tolist()
    events = [{'time': .2, 'type': 'set_aim', 'name': 'gun', 'point': at(60, 5)},
              {'time': 1.5, 'type': 'set_aim', 'name': 'gun', 'point': at(150, 30)}]
    trace, report = play(pkg, K.limits(K.still(3.0, events=events), runtime, settle_s=.9))
    assert report['passed'], report['issues']
    axes = [game_scenario.matrix(f['joints'][trace['joints'].index('turret')])[:3, :3] @ [0, 0, -1] for f in trace['frames']]
    angles = np.array([game_scenario.aim_angles(a, [0, 0, -1]) for a in axes])
    at_time = lambda t: angles[int(np.argmin(np.abs(np.array([f['time'] for f in trace['frames']])-t)))]
    assert at_time(1.0) == pytest.approx([60, 5], abs=.01)      # 60 degrees at 90 a second: 0.67 s
    # the frame at 0.2 s turns too: 0.3 s and one frame of 1/60 s, along the line to (60, 5)
    assert at_time(.5)[0] == pytest.approx(90*(.3+1/60)*60/math.hypot(60, 5), abs=.05)
    assert angles[-1] == pytest.approx([90, 10], abs=.01)                    # clamped to the limits


@pytest.mark.parametrize('change,match', [
    (lambda s: s.update(colour='red'), 'unknown key'),
    (lambda s: s.update(dt=[0]), 'dt'),
    (lambda s: s['events'].append({'time': .1, 'type': 'trigger', 'name': 'dance'}), 'no trigger'),
    (lambda s: s['events'].append({'time': .1, 'type': 'wave'}), 'type'),
    (lambda s: s['criteria'].pop('slip_mps_max'), 'criteria'),
    (lambda s: s['criteria'].update(made_up_max=1), 'criteria'),
    (lambda s: s.update(root=[[0, 0, 0, 0, 0]]), 'root'),
    (lambda s: s['terrain'].update(z=[1, 0]), 'terrain'),
    (lambda s: s.update(expect={'clip': 'moonwalk', 'offset_s': 0}), 'moonwalk'),
])
def test_scenario_refusals(biped, change, match):
    pkg, runtime = biped
    spec = K.walk(runtime, duration=1.0)
    change(spec)
    with pytest.raises(ValueError, match=match):
        game_runtime.run(pkg, spec)


def test_cli_run_check_and_preview(biped, tmp_path, capsys):
    pkg, runtime = biped
    scenario = tmp_path/'scenario.json'
    scenario.write_text(json.dumps(K.walk(runtime, duration=1.0, **TERRAINS['step'])))
    trace = tmp_path/'trace.json'
    base = ['--package', str(pkg), '--scenario', str(scenario)]
    assert cli.main(['game-run', *base, '--out', str(trace)]) == 0
    assert cli.main(['game-run', *base, '--out', str(trace)]) == 2          # a trace is never overwritten
    assert cli.main(['game-check', *base, '--trace', str(trace), '--report', str(tmp_path/'r.json')]) == 0
    assert json.loads((tmp_path/'r.json').read_text())['passed']
    assert cli.main(['game-preview', *base, '--trace', str(trace), '--out', str(tmp_path/'look'), '--phases', '3',
                     '--size', '96']) == 0
    assert (tmp_path/'look'/'filmstrip.png').is_file() and (tmp_path/'look'/'run.gif').is_file()
    off = json.loads(scenario.read_text())
    off['events'].append({'time': 0, 'type': 'set_ground_enabled', 'enabled': False})
    off['events'].sort(key=lambda e: e['time'])
    scenario.write_text(json.dumps(off))
    assert cli.main(['game-run', *base, '--out', str(tmp_path/'off.json')]) == 0
    assert cli.main(['game-check', *base, '--trace', str(tmp_path/'off.json')]) == 1


def test_example_scenarios_and_trial_contract_parse():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]/'examples'/'game-ready'
    for path in sorted(root.glob('*.scenario.json')):
        game_scenario.contract(json.loads(path.read_text()))
    from mesh_jig import game
    game.contract(json.loads((root/'character.trial.json').read_text()))


def test_a_toe_cap_is_kept_out_of_a_step(tmp_path, monkeypatch):
    glb = legged(tmp_path, BIPED_LEGS, BIPED_GAITS, (.4, .25, .3), toes=True)
    pkg, runtime = package(tmp_path, glb, biped_contract(glb))
    leg = runtime['limbs'][0]
    points, heights = np.array(leg['clearance']['points']), leg['clearance']['heights']
    assert 0 < len(points) <= 32 and len(heights) == len(points)
    # the toe cap's front reaches 22 cm ahead of the ankle (glTF -Z) and 15 mm up from the sole
    assert points[:, 2].min() == pytest.approx(-.22, abs=1e-6)
    assert min(h for p, h in zip(points, heights) if p[2] < -.159) == pytest.approx(.015, abs=1e-6)
    spec = K.walk(runtime, **TERRAINS['step'])
    _, report = play(pkg, spec)
    assert report['gates']['penetration_mm']['passed'] and report['gates']['joint_speed_mps']['passed'], report['gates']
    original = game_runtime.Runtime.__init__

    def soles_only(self, package):
        original(self, package)
        for limb in self.spec['limbs']:
            if 'clearance' in limb:
                limb['clearance']['points'] = []
        self.spec['ground']['lookahead_s'] = 0.0
    monkeypatch.setattr(game_runtime.Runtime, '__init__', soles_only)
    _, report = play(pkg, spec)
    assert not report['gates']['penetration_mm']['passed']
    assert report['readings']['deepest_penetration']['node'].startswith('toe.')


def test_a_leg_may_be_a_chain_of_four_joints(tmp_path):
    glb = legged(tmp_path, BIPED_LEGS, BIPED_GAITS, (.4, .25, .3), hipj=True)
    spec = biped_contract(glb)
    for limb, (name, _, _) in zip(spec['limbs'], BIPED_LEGS):
        limb.update(joints=['hip.'+name, 'thigh.'+name, 'shin.'+name, 'foot.'+name], solver='chain', bend_degrees_max=130)
        del limb['pole'], limb['bend_degrees']
    pkg, runtime = package(tmp_path, glb, spec)
    assert runtime['limbs'][0]['segment_lengths_m'] == pytest.approx([.08, .4, .44])
    for kind in ('slope', 'step'):
        _, on = play(pkg, K.walk(runtime, **TERRAINS[kind]))
        assert on['passed'], (kind, on['issues'])
        _, off = play(pkg, K.walk(runtime, ground=False, **TERRAINS[kind]))
        assert not off['gates']['penetration_mm']['passed']


def test_the_scenario_hash_is_the_file_bytes(biped, tmp_path):
    import hashlib
    pkg, runtime = biped
    path = tmp_path/'scenario.json'
    path.write_text(json.dumps(K.walk(runtime, duration=.5), indent=3)+'\n')      # any layout: the bytes count
    scenario, digest = game_scenario.load(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    trace = tmp_path/'trace.json'
    assert cli.main(['game-run', '--package', str(pkg), '--scenario', str(path), '--out', str(trace)]) == 0
    assert json.loads(trace.read_text())['hashes']['scenario_sha256'] == digest
    assert cli.main(['game-check', '--package', str(pkg), '--scenario', str(path), '--trace', str(trace)]) == 0
    path.write_text(json.dumps(scenario, indent=1)+'\n')                          # same scenario, other bytes
    assert cli.main(['game-check', '--package', str(pkg), '--scenario', str(path), '--trace', str(trace)]) == 2


def test_an_aim_that_carries_a_limb_does_not_pull_it_off_its_target(tmp_path):
    glb, spec = K.arm(tmp_path, part, rig, terrain, write_glb)
    spec['aims'] = [{'name': 'look', 'joints': ['body'], 'axis': [0, 1, 0], 'yaw_degrees': [-60, 60],
                     'pitch_degrees': [-20, 20], 'degrees_per_s': 90}]
    pkg, runtime = package(tmp_path, glb, spec)
    side = (np.array([0, 1, 0])+3*game_scenario.aim_direction(40, 0, [0, 0, -1])).tolist()
    events = [{'time': .2, 'type': 'set_aim', 'name': 'look', 'point': side}]
    events += [{'time': round(.1+.05*k, 6), 'type': 'set_limb_target', 'name': 'arm', 'point': [.1, .2, -.3],
                'weight': round(.05*(k+1), 2)} for k in range(20)]
    _, report = play(pkg, K.limits(K.still(2.2, events=sorted(events, key=lambda e: e['time'])), runtime))
    # the aim turns the body (and the arm under it) 40 degrees; the arm is solved after, so it still reaches
    assert report['passed'], report['issues']
    assert report['gates']['limb_target_error_mm']['value'] < 1 and report['gates']['aim_error_deg']['value'] < 1


TOGGLE = [{'time': 1.1, 'type': 'set_ground_enabled', 'enabled': False},
          {'time': 1.8, 'type': 'set_ground_enabled', 'enabled': True}]


def test_turning_the_ground_off_and_on_mid_walk_does_not_pop(tmp_path):
    # a biped with toe joints, so a turn of a foot moves a joint, circling tightly on a slope: late in a stance the
    # lock has turned the foot well away from the animation, and the ground fading out must turn it back gradually
    glb = legged(tmp_path, BIPED_LEGS, BIPED_GAITS, (.4, .25, .3), toej=True)
    pkg, runtime = package(tmp_path, glb, biped_contract(glb))
    spec = K.circle(runtime, .75, lambda x, z: -.1*z, events=TOGGLE)
    spec['criteria']['joint_speed_mps_max'] = 2.5          # steady 1.74 m/s; a foot that snaps round reads 3.8
    _, report = play(pkg, spec)
    assert report['gates']['joint_speed_mps']['passed'], report['gates']['joint_speed_mps']


def test_turning_the_ground_off_and_on_with_tilt_does_not_pop(tmp_path):
    glb = legged(tmp_path, QUAD_LEGS, {'walk': QUAD_WALK, 'run': QUAD_RUN}, (.5, .8, .25), toej=True)
    pkg, runtime = package(tmp_path, glb, quad_contract(glb))
    spec = K.walk(runtime, height=lambda x, z: -.08*z+.1*x, duration=2.4, events=TOGGLE)
    spec['criteria']['joint_speed_mps_max'] = 3.0          # steady 1.67 m/s; a tilt that resets at once reads 7
    _, report = play(pkg, spec)
    assert report['gates']['joint_speed_mps']['passed'], report['gates']['joint_speed_mps']


@pytest.mark.parametrize('kind', ['step', 'raise'])
def test_a_clock_a_nanosecond_off_gives_the_same_gate_values(biped, kind):
    """An engine's frame clock differs from the scenario's in the last bits. Whether a frame lands exactly on a stance
    window's edge must not move a gate: the lock holds while any contact point touches, so planting a frame earlier or
    later changes little. Before that, on the armour, one frame's later lock raised slip from 0.055 to 0.078 m/s;
    on this biped the old rule moves only penetration, by fractions of a micrometre, but that is enough to fail here."""
    pkg, runtime = biped
    spec = K.walk(runtime, **TERRAINS[kind])
    n = len(game_scenario.frame_times(spec))-1
    reports = []
    for seconds in (-1e-9, 0.0, 1e-9):
        nudged = copy.deepcopy(spec)
        nudged['dt'] = [d+seconds/n for d in spec['dt']]          # the last frame lands `seconds` off
        reports.append(play(pkg, nudged)[1]['gates'])
    for gate, g in reports[1].items():
        for other in (reports[0], reports[2]):
            if g['value'] is not None:
                assert abs(other[gate]['value']-g['value']) <= max(.01*abs(g['value']), 1e-4), (gate, g, other[gate])
