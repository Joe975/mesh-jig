"""The checker against runtimes broken on purpose: each defect fails its own named gate on a scenario the reference
holds. The checker imports no runtime and refuses a trace that is not of this package and scenario."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest

import gamekit as K
from mesh_jig import game_check, game_runtime, game_scenario
from test_game_runtime import TERRAINS, arm, biped, play, turret  # noqa: F401  (fixtures)

R = game_runtime.Runtime
FLIP = np.diag([1., 1, -1, 1])


def cached_ground(original):
    def step(self, dt, root, ground):
        self.__dict__.setdefault('frozen_ground', ground)
        return original(self, dt, root, self.frozen_ground)
    return step


def stretching_chain(original):
    def chain(self, world, limb, target):
        original(self, world, limb, target)
        self.translate(world, limb['joints'][-1], target-world[limb['joints'][-1]][:3, 3])
    return chain


def unlocked(self, state, points, support, planted, root_yaw, dt, far):
    return np.zeros(2), 0.0


def far_target(runtime):
    events = [{'time': .1+.05*k, 'type': 'set_limb_target', 'name': 'arm', 'point': [0, .9, -1.5],
               'weight': round(.05*(k+1), 2)} for k in range(20)]
    return K.limits(K.still(2.2, events=events), runtime, joint_speed_mps_max=20)   # it straightens fast once out of reach


def wide_aim(runtime):
    point = (np.array([0, .3, 0])+3*game_scenario.aim_direction(150, 0, [0, 0, -1])).tolist()
    return K.limits(K.still(3.0, events=[{'time': .2, 'type': 'set_aim', 'name': 'gun', 'point': point}]), runtime)


DEFECTS = {
    'no stance lock': ('biped', 'lock', lambda o: unlocked, lambda rt: K.walk(rt, root_speed=.5), 'slip_mps'),
    'stretches bones': ('arm', 'chain', stretching_chain, far_target, 'stretch_mm'),
    'ignores terrain': ('biped', 'ground_height', lambda o: lambda self, ground, x, z: 0.0,
                        lambda rt: K.walk(rt, **TERRAINS['step']), 'contact_gap_mm'),
    'snaps a fade': ('biped', 'oneshot_weight', lambda o: lambda self, f, tau, d: 1.0,
                     lambda rt: K.walk(rt, events=[{'time': .5, 'type': 'play', 'name': 'salute'}]), 'joint_speed_mps'),
    'wrong clip speed': ('biped', 'step', lambda o: lambda self, dt, root, ground: o(self, dt*1.1, root, ground),
                         lambda rt: K.walk(rt, ground=False), 'playback_mm'),
    'flipped axis': ('biped', 'place', lambda o: lambda self, root, model: root @ FLIP @ model,
                     lambda rt: K.walk(rt, ground=False), 'playback_mm'),
    'ignores aim limits': ('turret', 'clamp_aim', lambda o: lambda self, aim, yaw, pitch: (yaw, pitch), wide_aim, 'limits_deg'),
    'never recovers': ('biped', 'step', cached_ground, lambda rt: K.walk(rt, **TERRAINS['raise']), 'recovery_mm'),
}


@pytest.mark.parametrize('name', sorted(DEFECTS))
def test_each_defect_fails_its_own_gate(name, request, monkeypatch):
    fixture, method, broken, scenario, gate = DEFECTS[name]
    pkg, runtime = request.getfixturevalue(fixture)
    spec = scenario(runtime)
    _, reference = play(pkg, spec)
    assert reference['passed'], reference['issues']
    monkeypatch.setattr(R, method, broken(getattr(R, method)))
    _, report = play(pkg, spec)
    assert not report['gates'][gate]['passed'], report['gates'][gate]
    assert not report['passed']


def test_ignoring_terrain_also_fails_penetration(biped, monkeypatch):
    pkg, runtime = biped
    monkeypatch.setattr(R, 'ground_height', lambda self, ground, x, z: 0.0)
    _, report = play(pkg, K.walk(runtime, **TERRAINS['step']))
    assert not report['gates']['penetration_mm']['passed']


def test_the_checker_imports_no_runtime():
    source = (Path(game_check.__file__)).read_text(encoding='utf-8')
    imported = {alias.name for node in ast.walk(ast.parse(source)) if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    assert 'game_runtime' not in imported and 'game_runtime' not in source


def test_a_trace_of_another_scenario_or_shape_is_refused(biped):
    pkg, runtime = biped
    spec = K.walk(runtime, duration=1.0)
    trace = game_runtime.run(pkg, spec)
    other = K.walk(runtime, duration=1.0, root_speed=.5)
    with pytest.raises(ValueError, match='hashes differ'):
        game_check.check(pkg, other, trace)
    short = json.loads(json.dumps(trace))
    short['frames'] = short['frames'][:-1]
    with pytest.raises(ValueError, match='frames'):
        game_check.check(pkg, spec, short)
    renamed = json.loads(json.dumps(trace))
    renamed['joints'] = list(reversed(renamed['joints']))
    with pytest.raises(ValueError, match='joints'):
        game_check.check(pkg, spec, renamed)


def test_the_trace_format(biped):
    pkg, runtime = biped
    spec = K.walk(runtime, duration=.5, dt=(.1,))
    trace = game_runtime.run(pkg, spec)
    assert set(trace) == {'format', 'version', 'runtime', 'hashes', 'joints', 'matrix_layout', 'frames'}
    assert trace['joints'] == [j['name'] for j in runtime['skeleton']['joints']]
    frame = trace['frames'][-1]
    assert frame['time'] == .5 and len(frame['joints']) == len(trace['joints'])
    root = game_scenario.matrix(frame['root'])
    # column-major: the translation is entries 12, 13, 14; the root walks toward -Z
    assert frame['root'][12:15] == pytest.approx(root[:3, 3]) and root[2, 3] < -.25
    hips = game_scenario.matrix(frame['joints'][trace['joints'].index('hips')])
    assert hips[3].tolist() == [0, 0, 0, 1] and hips[1, 3] == pytest.approx(.8, abs=.02)


def test_the_porting_page_names_only_functions_that_exist():
    import re
    from mesh_jig import game_runtime, game_scenario
    page = (Path(__file__).resolve().parents[1]/'docs'/'GAME_READY_ENGINES.md').read_text(encoding='utf-8')
    cited = set(re.findall(r'`Runtime\.(\w+)`', page)) | set(re.findall(r'Runtime\.(\w+)', page))
    assert {'animate', 'adapt', 'lock', 'foot_height', 'two_bone', 'chain', 'aim', 'place', 'step'} <= cited
    for name in cited:
        assert hasattr(game_runtime.Runtime, name), name
    for name in set(re.findall(r'game_scenario\.(\w+)', page)):
        assert hasattr(game_scenario, name), name
    for name in set(re.findall(r'game_runtime\.(\w+)', page)) - {'py'}:
        assert hasattr(game_runtime, name), name
    for key in ('format', 'runtime', 'hashes', 'joints', 'matrix_layout', 'frames', 'scenario_sha256', 'column-major'):
        assert key in page
