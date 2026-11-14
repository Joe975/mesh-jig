"""Portable face controller. Blender metres, +Y forward, +Z up.

Only math/numpy: this file is also embedded in the rig export script. Public
callers use pose_at with a validated face contract; no mutable module state.
"""
import math
import numpy as np


def selection(region, name, position):
    if name not in region['nodes']:
        return np.zeros(len(position))
    if 'box' in region:
        lo, hi = np.asarray(region['box'], float)
        return np.all((position >= lo) & (position <= hi), axis=1).astype(float)
    if 'center' in region:
        r = np.linalg.norm((position-np.asarray(region['center'])) / np.asarray(region['radius']), axis=1)
        u = np.clip((1-r) / (1-region.get('inner', 0.5)), 0, 1)
        return u*u*(3-2*u)
    return np.ones(len(position))


def make_skeleton(config):
    bones = [dict(name='face_root', head=[0, 0, 0], tail=[0, 0, .01], parent=None)]
    for name, control in config['controls'].items():
        pivot = np.asarray(control['pivot'])
        parent = 'face_root' if name == 'neck' else 'face_neck'
        bones.append(dict(name='face_'+name, head=pivot.tolist(), tail=(pivot+[0, 0, .01]).tolist(), parent=parent))
    return bones


def make_weights(config, name, position):
    controls = config['controls']
    head = selection(controls['neck']['region'], name, position)
    # Optional linear attachment blend (e.g. a continuous neck on a fused skin).
    if 'ramp' in controls['neck']:
        axis, lo, hi = controls['neck']['ramp']
        u = np.clip((position[:, int(axis)]-lo)/(hi-lo), 0, 1)
        head *= u*u*(3-2*u)
    if name in controls['neck'].get('rigid_nodes', []):
        head = np.ones(len(position))
    selected = {key: selection(c['region'], name, position) for key, c in controls.items() if key != 'neck'}
    total = sum(selected.values(), np.zeros(len(position)))
    scale = np.maximum(total, 1)
    result = {'face_root': 1-head, 'face_neck': head*(1-np.minimum(total, 1))}
    for key, value in selected.items():
        result['face_'+key] = head*value/scale
    return result


def angles(direction, yaw_limit, pitch_limit):
    x, y, z = direction
    if np.linalg.norm(direction) < 1e-9:
        return 0., 0.
    yaw = np.clip(math.degrees(math.atan2(-x, y)), -yaw_limit, yaw_limit)
    pitch = np.clip(math.degrees(math.atan2(z, math.hypot(x, y))), -pitch_limit, pitch_limit)
    return float(yaw), float(pitch)


def rotation(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return rz @ rx


def pose_at(config, state, t=0., target=None, intensity=1.):
    """Return rig.py bone deltas. Target is model-space, not camera/world space.

    Expression states hold their pose. Talking is a two-second periodic motion;
    it is deliberately not audio-driven lip sync. Call once per runtime frame.
    """
    states = ('look_at', 'talk', 'smile', 'anger', 'sadness', 'confusion', 'yell')
    if state not in states or not math.isfinite(t) or not math.isfinite(intensity) or not 0 <= intensity <= 1:
        raise ValueError('invalid face state, time or intensity (0..1)')
    c = config['controls']
    amount = config['face_scale_m']
    result = {}
    neck = []
    jaw = 0.
    corners = [0., 0.]
    brows = [0., 0.]
    tilt = [0., 0.]
    if state == 'look_at':
        target = np.asarray(config['look_target'] if target is None else target, float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValueError('look target must be a finite three-component model-space point')
        pivot = np.asarray(c['neck']['pivot'])
        yaw, pitch = angles(target-pivot, 80, 60)
        yaw, pitch = np.clip(yaw*.65, -45, 45)*intensity, np.clip(pitch*.65, -25, 25)*intensity
        neck = [('X', float(pitch)), ('Z', float(yaw))]
        rot = rotation(yaw, pitch)
        for key in ('eye_l', 'eye_r'):
            if key in c:
                eye = pivot + rot @ (np.asarray(c[key]['pivot'])-pivot)
                ey, ep = angles(rot.T @ (target-eye), 30, 25)
                result['face_'+key] = {'rotate': [('X', ep*intensity), ('Z', ey*intensity)]}
    elif state == 'talk':
        jaw = -config.get('jaw_open_degrees', 28)*(.12+.09*math.sin(6*math.pi*t)+.03*math.sin(10*math.pi*t))
        neck = [('X', 1.5*math.sin(2*math.pi*t)), ('Z', math.sin(math.pi*t))]
    elif state == 'smile':
        corners = [.035, .035]
        brows = [.008, .008]
    elif state in ('anger', 'yell'):
        brows, tilt = [-.018, -.018], [15., -15.]
        corners = [-.012, -.012]
        if state == 'yell':
            jaw = -config.get('jaw_open_degrees', 28)
            neck = [('X', 5.)]
    elif state == 'sadness':
        corners, brows, tilt = [-.035, -.035], [.012, .012], [-14., 14.]
        neck = [('X', -5.)]
    else:
        brows, tilt = [.035, -.012], [8., 8.]
        neck = [('Y', 7.), ('Z', 3.)]
    if state != 'look_at':
        neck = [(axis, value*intensity) for axis, value in neck]
    result['face_neck'] = {'rotate': neck}
    if 'jaw' in c:
        result['face_jaw'] = {'rotate': [('X', jaw*intensity)]}
    for i, side in enumerate(('l', 'r')):
        if 'mouth_'+side in c:
            result['face_mouth_'+side] = {'move': (0, 0, corners[i]*amount*intensity)}
        if 'brow_'+side in c:
            result['face_brow_'+side] = {'move': (0, 0, brows[i]*amount*intensity), 'rotate': [('Y', tilt[i]*intensity)]}
    return result
