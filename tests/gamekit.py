"""Scenarios for the game-ready runtime and checker tests, written in-test in glTF axes."""
import numpy as np

from mesh_jig import game_scenario

LIMITS = {'penetration_mm_max': 3, 'stretch_mm_max': .5, 'joint_speed_mps_max': 8, 'contact_gap_mm_max': 3,
          'slip_mps_max': .05, 'support_fraction_min': .95, 'playback_mm_max': .5, 'limit_violation_deg_max': .5,
          'aim_error_deg_max': 1, 'limb_target_error_mm_max': 2, 'settle_s': .6, 'recovery_mm_max': 4, 'grace_s': .3}


def field(x, z, height):
    """A grid over x and z with height(x, z) at each point."""
    return {'x': list(x), 'z': list(z), 'heights': [[float(height(a, b)) for a in x] for b in z]}


def walk(runtime, *, clip='walk', height=lambda x, z: 0.0, xs=None, zs=None, duration=2.0, dt=(1/60,), events=(),
         root_speed=None, ground=True, expect=True, rest_ground=0.0, grid=None):
    """A character walking toward -Z (its forward) at the clip's measured speed, the root kept on the ground under
    it (as a game's controller would), with its speed parameter set to the clip's measured value."""
    speed = runtime['clips'][clip]['travel_speed_mps']
    v = speed if root_speed is None else root_speed
    xs = np.linspace(-1, 1, 3) if xs is None else xs
    zs = np.linspace(-v*duration-1.5, 1, 40) if zs is None else zs
    terrain = grid or field(xs, zs, height)
    ground_now = game_scenario.Ground({'terrain': terrain})
    keys = [[t, 0.0, float(ground_now.height(0.0, -v*t))-rest_ground, -v*t, 0.0] for t in np.arange(0, duration+1e-9, .05)]
    keys[-1][0] = duration
    first = [{'time': 0, 'type': 'set_parameter', 'name': 'speed', 'value': speed}]
    if not ground:
        first.append({'time': 0, 'type': 'set_ground_enabled', 'enabled': False})
    spec = {'version': 1, 'terrain': terrain, 'dt': list(dt), 'duration': duration, 'root': keys,
            'events': first+sorted(events, key=lambda e: e['time']), 'criteria': {}}
    if expect:
        point = next(p for layer in runtime['graph']['layers'] if layer['node']['type'] == 'blend1d'
                     for p in layer['node']['points'] if p.get('clip') == clip)
        spec['expect'] = {'clip': clip, 'offset_s': point['phase_offset']*runtime['clips'][clip]['duration_s']}
    return limits(spec, runtime)


def circle(runtime, radius, height, duration=2.4, events=(), clip='walk'):
    """A character walking a left-hand circle of this radius at the clip's measured speed, the root on the ground."""
    import math
    v = runtime['clips'][clip]['travel_speed_mps']
    terrain = field(np.linspace(-4, 2, 25), np.linspace(-4, 2, 25), height)
    ground = game_scenario.Ground({'terrain': terrain})
    keys = []
    for t in np.arange(0, duration+1e-9, .05):
        yaw = v/radius*t
        x, z = -radius+radius*math.cos(yaw), -radius*math.sin(yaw)
        keys.append([round(float(t), 6), x, float(ground.height(x, z)), z, math.degrees(yaw)])
    spec = {'version': 1, 'terrain': terrain, 'dt': [1/60], 'duration': duration, 'root': keys,
            'events': [{'time': 0, 'type': 'set_parameter', 'name': 'speed', 'value': v}]+list(events), 'criteria': {}}
    return limits(spec, runtime)


def still(duration=2.0, terrain=None, events=(), dt=(1/60,)):
    """A character standing at the origin on flat ground."""
    terrain = terrain or field([-3, 3], [-3, 3], lambda x, z: 0.0)
    return {'version': 1, 'terrain': terrain, 'dt': list(dt), 'duration': duration,
            'root': [[0, 0, 0, 0, 0], [duration, 0, 0, 0, 0]], 'events': list(events), 'criteria': {}}


TURRET = '''import math
def skeleton(model):
    return [dict(name='root', head=(0, 0, 0), tail=(0, 0, .1), parent=None),
            dict(name='turret', head=(0, 0, .3), tail=(0, 0, .5), parent='root')]
def weights(name, position, piece):
    return 'root' if name == 'base' else 'turret'
def pose(clip, t):
    return {}
'''

ARM = '''import math
def skeleton(model):
    out = [dict(name='root', head=(0, 0, 0), tail=(0, 0, .1), parent=None),
           dict(name='body', head=(0, 0, 1.0), tail=(0, 0, 1.2), parent='root')]
    for k, z in enumerate((.9, .7, .5, .3, .1)):
        out.append(dict(name=f'a{k+1}', head=(0, 0, z), tail=(0, 0, z-.2), parent='body' if k == 0 else f'a{k}'))
    return out
def weights(name, position, piece):
    return 'body' if name == 'body' else 'a'+name[-1]
def pose(clip, t):
    return {j: [('X', 10*math.sin(2*math.pi*t))] for j in ('a1', 'a2', 'a3', 'a4')}
'''


def turret(tmp, part, rig, terrain, write_glb):
    """A turret whose clip holds still, with one aim: yaw -90..90, pitch -10..10, 90 degrees a second."""
    source = write_glb(tmp/'turret-source.glb', [part('base', (.6, .6, .2), (0, 0, .1)), part('gun', (.1, .5, .1), (0, .2, .35))])
    (tmp/'turret.py').write_text(TURRET)
    path = tmp/'turret.glb'
    rig.build(source, tmp/'turret.py', {'fps': 30, 'rigid': True, 'max_influences': 1,
                                        'clips': {'idle': {'seconds': 1, 'loop': True}}}, path)
    return path, {'version': 1, 'rig_sha256': terrain.digest(path),
                  'graph': {'layers': [{'name': 'base', 'node': {'type': 'clip', 'clip': 'idle'}}]},
                  'aims': [{'name': 'gun', 'joints': ['turret'], 'axis': [0, 1, 0], 'yaw_degrees': [-90, 90],
                            'pitch_degrees': [-10, 10], 'degrees_per_s': 90}],
                  'criteria': {'loop_seam_mm_max': 2}}


def arm(tmp, part, rig, terrain, write_glb):
    """A body with one five-joint arm hanging from it, swaying; a chain limb, bends up to 45 degrees a joint."""
    zs = [.9, .7, .5, .3, .1]
    parts = [part('body', (.3, .3, .2), (0, 0, 1.0))]+[part(f'seg{k+1}', (.08, .08, .19), (0, 0, z-.1)) for k, z in enumerate(zs[:4])]
    source = write_glb(tmp/'arm-source.glb', parts)
    (tmp/'arm.py').write_text(ARM)
    path = tmp/'arm.glb'
    rig.build(source, tmp/'arm.py', {'fps': 30, 'rigid': True, 'max_influences': 1,
                                     'clips': {'sway': {'seconds': 1, 'loop': True}}}, path)
    return path, {'version': 1, 'rig_sha256': terrain.digest(path),
                  'graph': {'layers': [{'name': 'base', 'node': {'type': 'clip', 'clip': 'sway'}}]},
                  'limbs': [{'name': 'arm', 'role': 'reach', 'joints': ['a1', 'a2', 'a3', 'a4', 'a5'], 'solver': 'chain',
                             'bend_degrees_max': 45}],
                  'criteria': {'loop_seam_mm_max': 2}}


def limits(spec, runtime, **changes):
    need = game_scenario.needed_criteria(spec, runtime)
    spec['criteria'] = {k: changes.get(k, LIMITS[k]) for k in sorted(need)}
    return spec
