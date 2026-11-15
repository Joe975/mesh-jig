"""The reference runtime for a game-ready package: the executable specification an engine port follows.

Plain numpy, deterministic. It reads `character.glb` for the skeleton, skin and clips and `character.runtime.json`
for everything else. All positions are glTF axes (Y up, metres, right-handed) in world space. It never moves the
character: the game gives the root's world matrix each frame, and the runtime only writes joint matrices.

The interface: set_parameter, trigger, play, set_aim, set_limb_target, set_ground_enabled, and
step(dt, root_matrix, ground) -> {joint name: 4x4 world matrix}, where ground(x, z) -> (height, normal, walkable).

Each frame, in order:
 1. Advance the graph's clocks by dt and evaluate it, layer by layer, into joint local transforms.
 2. Forward kinematics: world = root matrix x the joint's model-space matrix.
 3. Ground adaptation, if on and the package has legs: decide which legs are planted from the clips' stance
    windows; while planted, lock the contact in world XZ and yaw by cancelling the drift of its points that stay on
    the ground (a correction that fades back to zero after lift-off or past relock_m); keep each contact's lowest
    point as high above the real ground as the animation has it above its own; move the body joint up or down
    within the drop and rise limits and the legs' reach; tilt it to the planted contacts with three or more legs;
    solve each leg; turn each tip back to its animated (or locked) world orientation.
 4. Aims, within their yaw and pitch limits and turn rate.
 5. Game-given limb targets, by weight (after aims, whose chains may carry a limb).
Every smoothing step is written in seconds, so a frame rate change does not change the result's speed.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from . import game_measure, game_scenario, terrain

NAME, VERSION = 'mesh-jig reference runtime', 1
FABRIK_ITERATIONS = 30         # fixed, so every port does the same work
REACH = 0.999                  # a chain is never asked to straighten fully: a straight knee has no pole
LOOKAHEAD_SAMPLES = 8          # points along a swinging foot's look-ahead where the ground is read


# --- small geometry -------------------------------------------------------------------------------------------------

def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v/n if n > 1e-12 else v


def rotation_between(a, b):
    """The shortest rotation (3x3) that turns direction a onto direction b. Twist about the direction is kept."""
    a, b = unit(a), unit(b)
    v, c = np.cross(a, b), float(np.clip(a @ b, -1, 1))
    if c > 1-1e-12:
        return np.eye(3)
    if c < -1+1e-9:                                  # opposite: turn half way round any perpendicular axis
        axis = unit(np.cross(a, np.eye(3)[np.argmin(np.abs(a))]))
        return 2*np.outer(axis, axis)-np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3)+k+k @ k/(1+c)


def axis_angle(axis, angle):
    x, y, z = unit(axis)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c+x*x*(1-c), x*y*(1-c)-z*s, x*z*(1-c)+y*s],
                     [y*x*(1-c)+z*s, c+y*y*(1-c), y*z*(1-c)-x*s],
                     [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]])


def yaw_matrix(radians):
    return axis_angle([0, 1, 0], radians)


def smooth(dt, seconds):
    """The share of the way to a target covered in dt by a follower with this time constant (dt-correct)."""
    return 1.0 if seconds <= 0 else 1-math.exp(-dt/seconds)


def blend_trs(a, b, w):
    """Lerp translation and scale, shortest-arc nlerp rotation (as the build's blend)."""
    return game_measure.blend({0: a}, {0: b}, w)[0]


# --- the runtime ----------------------------------------------------------------------------------------------------

class Runtime:
    def __init__(self, package):
        package = Path(package)
        self.spec = json.loads((package/'character.runtime.json').read_text(encoding='utf-8'))
        self.body = game_measure.Body(package/self.spec['glb'])
        joints = self.spec['skeleton']['joints']
        self.names = [j['name'] for j in joints]
        self.parent = {j['name']: j['parent'] for j in joints}
        self.children = {n: [c for c in self.names if self.parent[c] == n] for n in self.names}
        self.params = {k: v['default'] for k, v in self.spec['parameters'].items()}
        self.pending_triggers, self.pending_plays = set(), set()
        self.clocks = {}                                       # graph node path -> state
        self.aim_targets, self.limb_targets, self.aim_state = {}, {}, {}
        self.ground_on, self.ground_weight = True, 1.0
        self.legs = [l for l in self.spec['limbs'] if l['role'] == 'leg']
        self.leg_state = {l['name']: {} for l in self.legs}
        self.body_offset, self.tilt = 0.0, np.eye(3)
        self.anim_ground = float(self.body.rest[:, 2].min())   # the rest pose's lowest vertex: model-space height
        self.first = True

    # the interface ----------------------------------------------------------------------------------------------
    def set_parameter(self, name, value):
        self.params[name] = float(value)

    def trigger(self, name):
        self.pending_triggers.add(name)

    def play(self, name):
        self.pending_plays.add(name)

    def set_aim(self, name, point):
        self.aim_targets[name] = None if point is None else np.asarray(point, float)

    def set_limb_target(self, name, point, weight=1.0):
        self.limb_targets[name] = None if point is None else (np.asarray(point, float), float(weight))

    def set_ground_enabled(self, enabled):
        self.ground_on = bool(enabled)

    def step(self, dt, root, ground):
        if self.first:                    # the ground weight starts where the calls before the first frame left it
            self.ground_weight = 1.0 if self.ground_on else 0.0
        pose, votes = self.animate(dt)
        model = self.body.world(pose)
        world = {n: self.place(root, model[self.body.node[n]]) for n in self.names}
        self.adapt(world, root, votes, ground, dt)
        for aim in self.spec['aims']:            # aims before limb targets: an aim's chain may carry a limb
            self.aim(world, root, aim, dt)
        for limb in self.spec['limbs']:
            target = self.limb_targets.get(limb['name'])
            if target is not None and target[1] > 0:
                tip = world[limb['joints'][-1]][:3, 3]
                self.solve(world, root, limb, tip+target[1]*(target[0]-tip))
        self.first = False
        return world

    def place(self, root, model):
        """2. A joint's world matrix: the root's world matrix times its model-space matrix. Both are glTF axes; an
        engine that is left-handed flips one axis of every runtime vector the same way, once, on import."""
        return root @ model

    # 1. the graph -----------------------------------------------------------------------------------------------
    def clock(self, path, dt):
        """Seconds since this node started: 0 on its first frame, then dt more each frame."""
        t = self.clocks.get(path)
        self.clocks[path] = t = 0.0 if t is None else t+dt
        return t

    def reset(self, path):
        for key in [k for k in self.clocks if k == path or k.startswith(path+'/') or k.startswith(path+'#')]:
            del self.clocks[key]

    def clip_time(self, clip, t):
        c = self.spec['clips'][clip]
        return t % c['duration_s'] if c['loop'] else min(t, c['duration_s'])

    def node(self, node, path, dt):
        """(pose or None, weight, [(clip, clip time, share)]): joint local transforms and the weight they are laid
        over the layers below with. None adds nothing (a null state, or one-shots when none is playing)."""
        kind = node['type']
        if kind == 'clip':
            t = self.clip_time(node['clip'], self.clock(path, dt))
            return self.body.full(node['clip'], t), 1.0, [(node['clip'], t, 1.0)]
        if kind == 'blend1d':
            return self.blend1d(node, path, dt)
        if kind == 'states':
            return self.states(node, path, dt)
        return self.oneshots(node, path, dt)

    def blend1d(self, node, path, dt):
        points, x = node['points'], self.params[node['parameter']]
        values = [p['value'] for p in points]
        if x <= values[0]:
            i, w = 0, 0.0
        elif x >= values[-1]:
            i, w = len(points)-2, 1.0
        else:
            i = max(k for k in range(len(points)-1) if values[k] <= x)
            w = (x-values[i])/(values[i+1]-values[i])
        a, b = points[i], points[i+1]
        t = self.clock(path, dt)       # the blend's own clock: for points off the cycle and unsynced blends
        if node.get('sync'):
            # Points on the cycle share one phase; its frequency lerps between them. A point off the cycle (an idle:
            # no leg touches down in it) runs on its own clock and adds nothing to the frequency.
            dur = lambda p: self.spec['clips'][p['clip']]['duration_s']
            on = [(p, s) for p, s in ((a, 1-w), (b, w)) if p['cycle']]
            freq = sum(s/dur(p) for p, s in on) if len(on) == 2 else 1/dur(on[0][0]) if on else 0.0
            phase = self.clocks.get(path+'#phase')
            self.clocks[path+'#phase'] = phase = 0.0 if phase is None else phase+dt*freq
            times = [((phase+p['phase_offset']) % 1)*dur(p) if p['cycle'] else t % dur(p) for p in (a, b)]
            poses = [self.body.full(p['clip'], tt) for p, tt in zip((a, b), times)]
            made = [(a['clip'], times[0], 1-w), (b['clip'], times[1], w)]
        else:
            poses, made = [], []
            for k, (p, s) in zip((i, i+1), ((a, 1-w), (b, w))):
                if 'clip' in p:
                    tt = self.clip_time(p['clip'], t)
                    poses.append(self.body.full(p['clip'], tt))
                    made.append((p['clip'], tt, s))
                else:          # a nested node's own weight is not used inside a blend
                    q, _, inner = self.node(p['node'], f'{path}/{k}', dt)
                    poses.append(q if q is not None else {j: dict(v) for j, v in self.body.rest_trs.items()})
                    made += [(c, tt, s*share) for c, tt, share in inner]
        return game_measure.blend(poses[0], poses[1], w), 1.0, made

    def ended(self, node, path):
        """A state that plays one clip has ended when its clock passes the clip's duration."""
        if node is None or node['type'] != 'clip':
            return False
        return self.clocks.get(path, 0.0) >= self.spec['clips'][node['clip']]['duration_s']

    def states(self, node, path, dt):
        st = self.clocks.setdefault(path+'#states', {'current': node['start'], 'previous': None, 'elapsed': 0.0})
        if st['previous'] is not None:
            st['elapsed'] += dt
        for tr in node['transitions']:
            if tr['from'] != st['current']:
                continue
            fire = tr['on'] in self.pending_triggers if 'on' in tr else self.ended(node['states'][tr['from']], f'{path}/{tr["from"]}')
            if fire:
                st.update(previous=st['current'], current=tr['to'], elapsed=0.0, fade=tr['fade_s'])
                self.reset(f'{path}/{tr["to"]}')
                break
        u = 1.0 if st['previous'] is None else self.fade(st['elapsed'], st['fade'])
        cur = node['states'][st['current']]
        b = (None, 0.0, []) if cur is None else self.node(cur, f'{path}/{st["current"]}', dt)
        if st['previous'] is None or u >= 1:
            st['previous'] = None
            return b
        prev = node['states'][st['previous']]
        a = (None, 0.0, []) if prev is None else self.node(prev, f'{path}/{st["previous"]}', dt)
        made = [(c, t, s*(1-u)) for c, t, s in a[2]]+[(c, t, s*u) for c, t, s in b[2]]
        if a[0] is None and b[0] is None:
            return None, 0.0, []
        if a[0] is None:                      # fading in over the layers below
            return b[0], u, made
        if b[0] is None:                      # fading out to the layers below
            return a[0], 1-u, made
        return game_measure.blend(a[0], b[0], u), 1.0, made

    def fade(self, elapsed, seconds):
        """Linear cross-fade progress, 0 to 1."""
        return min(1.0, elapsed/seconds)

    def oneshots(self, node, path, dt):
        for name in sorted(self.pending_plays):
            if name in node['clips']:
                self.clocks[path+'#oneshot'] = {'clip': name, 'tau': None}       # a replay restarts it
        st = self.clocks.get(path+'#oneshot')
        if st is None:
            return None, 0.0, []
        st['tau'] = 0.0 if st['tau'] is None else st['tau']+dt
        d = self.spec['clips'][st['clip']]['duration_s']
        if st['tau'] > d:
            del self.clocks[path+'#oneshot']
            return None, 0.0, []
        f = node['clips'][st['clip']]
        return self.body.full(st['clip'], st['tau']), self.oneshot_weight(f, st['tau'], d), [(st['clip'], st['tau'], 1.0)]

    def oneshot_weight(self, f, tau, d):
        """Fades in from the start and out before the end, linearly."""
        return max(0.0, min(1.0, tau/f['fade_in_s'], (d-tau)/f['fade_out_s']))

    def animate(self, dt):
        """The layers, in order: a later layer replaces the joints in its mask by its weight. Also, for each leg,
        the clips that made its tip joint's pose, with their shares (the stance vote)."""
        pose = {i: dict(v) for i, v in self.body.rest_trs.items()}
        votes = {l['name']: [] for l in self.legs}
        for k, layer in enumerate(self.spec['graph']['layers']):
            p, w, made = self.node(layer['node'], str(k), dt)
            if p is None or w <= 0:
                continue
            mask = layer['mask_joints']
            for i in ([self.body.node[n] for n in mask] if mask else p):
                pose[i] = blend_trs(pose[i], p[i], w)
            for leg in self.legs:
                if mask is None or leg['joints'][-1] in mask:
                    votes[leg['name']] = [(c, t, s*(1-w)) for c, t, s in votes[leg['name']]]+[(c, t, s*w) for c, t, s in made]
        self.pending_triggers.clear()
        self.pending_plays.clear()
        return pose, votes

    # 3. ground adaptation ---------------------------------------------------------------------------------------
    def planted(self, leg, votes):
        """Planted when the clips that make the leg's pose, by share, say stance at their current phase."""
        total, yes = 0.0, 0.0
        for clip, t, share in votes:
            c = self.spec['clips'][clip]
            phase = t/c['duration_s']
            windows = c['legs'].get(leg['name'], {}).get('stance_phase', [])
            total += share
            yes += share*any(a <= phase <= b or a <= phase+1 <= b for a, b in windows)
        return total > 0 and yes >= total/2

    def ground_height(self, ground, x, z):
        return ground(float(x), float(z))[0]

    def lock(self, state, points, support, planted, root_yaw, dt, far):
        """The planted-foot lock, as a world correction added to the animated contact: a horizontal shift `shift`
        and a turn `turn` about +Y. While the leg is planted, the shift cancels the world drift of the contact
        points that are on the animation's ground in this frame and the last (so a foot rolling onto its toe keeps
        its toe still), and the turn cancels the root's change of yaw. The shift is never longer than relock_m:
        past it the foot slides, only as fast as the animation pulls it further. When the leg cannot reach, and
        whenever it is not planted, both fade to zero with time constant blend_s: the foot returns to the animation
        without a jump."""
        g = self.spec['ground']
        if planted and state.get('planted'):
            common = support & state['support']
            if common.any():
                state['shift'] = state['shift']-(points[common][:, [0, 2]]-state['points'][common][:, [0, 2]]).mean(axis=0)
            state['turn'] -= root_yaw-state['root_yaw']
        if not planted or far:
            fade = 1-smooth(dt, g['blend_s'])
            state['shift'], state['turn'] = state['shift']*fade, state['turn']*fade
        length = float(np.linalg.norm(state['shift']))
        if length > g['relock_m']:
            state['shift'] = state['shift']*(g['relock_m']/length)
        state.update(points=points, support=support, planted=planted, root_yaw=root_yaw)
        return state['shift'], state['turn']

    def reach(self, leg):
        """How far a leg's tip may be from its first joint: the bend-limited reach of a two-bone leg."""
        lengths = leg['segment_lengths_m']
        if leg['solver'] == 'two_bone':
            l1, l2 = lengths
            hi = math.radians(leg['bend_degrees'][1])
            return min(REACH*(l1+l2), math.sqrt(l1*l1+l2*l2-2*l1*l2*math.cos(hi)))
        return REACH*sum(lengths)

    def foot_height(self, leg, state, ground, tip_xz, rot, lift, planted, dt):
        """The tip's height. Each point of the foot (its contact points and its clearance points) asks for the ground
        under it; the tip goes as low as keeps every point at least `lift` above what it asks for. While the leg is
        not planted, a point also looks ahead along the foot's horizontal velocity for lookahead_s, read at
        LOOKAHEAD_SAMPLES points. Ground a share s of the way there that stands above the ground continued from just
        behind the point (one sample's spacing back, so a uniform slope stands above nothing) counts for (1 - s) of
        that rise: a riser coming up lifts the foot gradually instead of at once, and a slope does not lift it. The
        look-ahead's weight rises over blend_s after lift-off and falls over blend_s after touchdown, so the height
        never jumps."""
        g = self.spec['ground']
        local = np.asarray(leg['contact']['points'], float)
        if leg['clearance']['points']:
            local = np.r_[local, np.asarray(leg['clearance']['points'], float)]
        offsets = local @ rot.T
        previous = state.get('tip_xz')
        if previous is not None and dt > 0:
            state['velocity'] = (tip_xz-previous)/dt
        state['tip_xz'] = tip_xz.copy()
        step = dt/g['blend_s'] if g['blend_s'] > 0 else 1.0
        state['ahead'] = ahead = max(0.0, state.get('ahead', 0.0)-step) if planted else min(1.0, state.get('ahead', 0.0)+step)
        dx, dz = state.get('velocity', np.zeros(2))*g['lookahead_s']
        best = -math.inf
        for o in offsets:
            x, z = tip_xz[0]+o[0], tip_xz[1]+o[2]
            here = need = self.ground_height(ground, x, z)
            if ahead > 0 and dx*dx+dz*dz > 1e-12:
                back = self.ground_height(ground, x-dx/LOOKAHEAD_SAMPLES, z-dz/LOOKAHEAD_SAMPLES)
                for k in range(1, LOOKAHEAD_SAMPLES+1):
                    s = k/LOOKAHEAD_SAMPLES
                    rise = self.ground_height(ground, x+s*dx, z+s*dz)-(here+k*(here-back))
                    need = max(need, here+ahead*(1-s)*rise)
            best = max(best, need+lift-o[1])
        return best

    def adapt(self, world, root, votes, ground, dt):
        g = self.spec['ground']
        if not self.legs or g is None:
            return
        rate = dt/g['blend_s'] if g['blend_s'] > 0 else 1.0
        self.ground_weight = min(1.0, self.ground_weight+rate) if self.ground_on else max(0.0, self.ground_weight-rate)
        gw = self.ground_weight
        if gw <= 0:
            for s in self.leg_state.values():
                s.clear()
            self.body_offset, self.tilt = 0.0, np.eye(3)
            return
        root_yaw = math.atan2(root[0, 2], root[2, 2])
        anim_ground = (root @ [0, self.anim_ground, 0, 1])[1]
        gap = self.spec['measurement']['contact_gap_m']
        plan = {}
        for leg in self.legs:
            tip, s = leg['joints'][-1], self.leg_state[leg['name']]
            s.setdefault('shift', np.zeros(2))
            s.setdefault('turn', 0.0)
            contact = np.asarray(leg['contact']['points'], float)
            w = world[tip]
            points = contact @ w[:3, :3].T+w[:3, 3]
            clearance = points[:, 1]-anim_ground                # each contact point's height in the animation
            hip = world[leg['joints'][0]][:3, 3]
            reach = self.reach(leg)
            planted = self.planted(leg, votes[leg['name']])
            far = np.linalg.norm(hip[[0, 2]]-points.mean(axis=0)[[0, 2]]-s['shift']) > reach
            # the lock holds while the clips say planted and after, while any contact point is still on the
            # animation's ground (a toe rolling off): fading it then would slide the toe
            touching = clearance <= gap
            shift, turn = self.lock(s, points, touching, planted or bool(touching.any()), root_yaw, dt, far)
            rot = yaw_matrix(turn) @ w[:3, :3]
            offsets = contact @ rot.T
            tip_xz = points.mean(axis=0)[[0, 2]]+shift-offsets.mean(axis=0)[[0, 2]]
            # the contact's lowest point as high above the real ground as the animation has it above its own ground
            # (never below it), with the rest of the foot kept out of the ground too
            lift = max(0.0, float(clearance.min()))
            y = self.foot_height(leg, s, ground, tip_xz, rot, lift, planted, dt)
            plan[leg['name']] = (np.array([tip_xz[0], y, tip_xz[1]]), rot, planted, w[:3, 3].copy(), hip, reach)
        # the body: as low as the lowest planted target asks (smoothed), and at once low enough for every planted leg
        # to reach, within the drop and rise limits
        chosen = [p for p in plan.values() if p[2]] or list(plan.values())
        want = min(t[1]-a[1] for t, _, _, a, _, _ in chosen)
        bound = math.inf
        for t, _, _, _, hip, reach in chosen:
            flat = np.linalg.norm(t[[0, 2]]-hip[[0, 2]])
            bound = min(bound, t[1]-hip[1]+math.sqrt(max(reach*reach-flat*flat, 0.0)))
        want = float(np.clip(want, -g['drop_m_max'], g['rise_m_max']))
        self.body_offset = want if self.first else self.body_offset+(want-self.body_offset)*smooth(dt, g['blend_s'])
        offset = float(np.clip(min(self.body_offset, bound), -g['drop_m_max'], g['rise_m_max']))
        body = self.spec['body']['joint']
        self.translate(world, body, np.array([0, gw*offset, 0]))
        if g['tilt']:
            planted = [p for p in plan.values() if p[2]]
            goal = np.eye(3)
            if len(planted) >= 3:
                fit = lambda pts: unit(np.cross(*(np.linalg.svd(pts-pts.mean(axis=0))[2][:2])))
                normal = fit(np.array([p[0] for p in planted]))
                normal = normal if normal[1] > 0 else -normal
                goal = rotation_between([0, 1, 0], normal)
            self.tilt = goal if self.first else self.slerp(self.tilt, goal, smooth(dt, g['blend_s']))
            # laid on by the ground weight, so it fades out with the adaptation and is gone when the weight is 0
            self.rotate(world, body, self.slerp(np.eye(3), self.tilt, gw), world[body][:3, 3])
        for leg in self.legs:
            target, rot, planted, _, _, _ = plan[leg['name']]
            tip = leg['joints'][-1]
            self.solve(world, root, leg, world[tip][:3, 3]+gw*(target-world[tip][:3, 3]))
            # the tip turns to its target rotation by the ground weight
            self.rotate(world, tip, self.slerp(np.eye(3), rot @ world[tip][:3, :3].T, gw), world[tip][:3, 3])

    @staticmethod
    def slerp(a, b, u):
        r = b @ a.T
        angle = math.acos(float(np.clip((np.trace(r)-1)/2, -1, 1)))
        if angle < 1e-9:
            return b
        axis = np.array([r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1]])
        return axis_angle(axis, angle*u) @ a

    # moving parts of the skeleton -------------------------------------------------------------------------------
    def subtree(self, name):
        out, stack = [], [name]
        while stack:
            n = stack.pop()
            out.append(n)
            stack += self.children[n]
        return out

    def translate(self, world, name, offset):
        for n in self.subtree(name):
            world[n] = world[n].copy()
            world[n][:3, 3] += offset

    def rotate(self, world, name, r, pivot):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = r, pivot-r @ pivot
        for n in self.subtree(name):
            world[n] = m @ world[n]

    # 3, 4. solvers ----------------------------------------------------------------------------------------------
    def solve(self, world, root, limb, target):
        if limb['solver'] == 'two_bone':
            self.two_bone(world, root, limb, target)
        elif limb['solver'] == 'chain':
            self.chain(world, limb, target)

    def two_bone(self, world, root, limb, target):
        """Analytic: the knee in the plane of hip, target and pole; the reach clamped so the knee's inner angle stays
        within bend_degrees. Each bone is turned by the shortest rotation onto its new direction."""
        a, b, c = limb['joints']
        pa, pb, pc = (world[n][:3, 3] for n in (a, b, c))
        l1, l2 = np.linalg.norm(pb-pa), np.linalg.norm(pc-pb)
        lo, hi = (math.radians(x) for x in limb['bend_degrees'])
        reach = lambda angle: math.sqrt(max(l1*l1+l2*l2-2*l1*l2*math.cos(angle), 1e-12))
        d = target-pa
        r = float(np.clip(np.linalg.norm(d), reach(lo), min(reach(hi), REACH*(l1+l2))))
        direction = unit(d) if np.linalg.norm(d) > 1e-9 else unit(pc-pa)
        pole = root[:3, :3] @ np.asarray(limb['pole'], float)
        side = pole-direction*(pole @ direction)
        if np.linalg.norm(side) < 1e-6:                          # pole along the leg: keep the knee where it is
            side = (pb-pa)-direction*((pb-pa) @ direction)
        cos_a = float(np.clip((l1*l1+r*r-l2*l2)/(2*l1*r), -1, 1))
        knee = pa+direction*l1*cos_a+unit(side)*l1*math.sqrt(1-cos_a*cos_a)
        self.rotate(world, a, rotation_between(pb-pa, knee-pa), pa)
        pb, pc = world[b][:3, 3], world[c][:3, 3]
        self.rotate(world, b, rotation_between(pc-pb, pa+direction*r-pb), pb)

    def fabrik(self, points, target):
        """FABRIK with the root pinned and a fixed number of passes; an out-of-reach target straightens the chain."""
        points = [p.copy() for p in points]
        lengths = [np.linalg.norm(b-a) for a, b in zip(points, points[1:])]
        root = points[0].copy()
        if np.linalg.norm(target-root) >= sum(lengths):
            direction = unit(target-root)
            for k, length in enumerate(lengths):
                points[k+1] = points[k]+direction*length
            return points
        for _ in range(FABRIK_ITERATIONS):
            points[-1] = target.copy()
            for k in range(len(points)-2, -1, -1):
                points[k] = points[k+1]+unit(points[k]-points[k+1])*lengths[k]
            points[0] = root.copy()
            for k in range(len(points)-1):
                points[k+1] = points[k]+unit(points[k+1]-points[k])*lengths[k]
        return points

    def chain(self, world, limb, target):
        """Solve the positions with FABRIK, then turn each joint, root first, by the shortest rotation that aims
        its segment at its solved next point; a bend past bend_degrees_max from the segment before is held there."""
        joints = limb['joints']
        goal = self.fabrik([world[n][:3, 3] for n in joints], target)
        limit = math.radians(limb.get('bend_degrees_max', 180))
        for k in range(len(joints)-1):
            here, nxt = world[joints[k]][:3, 3], world[joints[k+1]][:3, 3]
            want = unit(goal[k+1]-here)
            if k > 0:
                before = unit(here-world[joints[k-1]][:3, 3])
                angle = math.acos(float(np.clip(before @ want, -1, 1)))
                axis = np.cross(before, want)
                if angle > limit and np.linalg.norm(axis) > 1e-9:
                    want = unit(axis_angle(axis, limit) @ before)
            self.rotate(world, joints[k], rotation_between(nxt-here, want), here)

    # 5. aims ------------------------------------------------------------------------------------------------------
    def aim(self, world, root, aim, dt):
        """Turn the chain so the last joint's axis points at the target. Yaw and pitch are measured in the frame of
        the first joint's parent from the axis at rest. The aim keeps an offset from the animation; the offset moves
        at most degrees_per_s (a step in yaw and pitch together) toward the one that points at the target clamped
        to the limits, or toward zero with no target, so the animation itself is never filtered. The turn is shared
        equally by the chain's joints."""
        joints, last = aim['joints'], aim['joints'][-1]
        parent = self.parent[joints[0]]
        frame = (world[parent] if parent else root)[:3, :3]
        rest = self.body.rest_world[self.body.node[parent]][:3, :3] if parent else np.eye(3)
        a0 = unit(rest.T @ np.asarray(aim['axis_model'], float))
        now = unit(frame.T @ world[last][:3, :3] @ np.asarray(aim['axis_local'], float))
        animated = np.array(game_scenario.aim_angles(now, a0))
        target = self.aim_targets.get(aim['name'])
        if target is not None:
            yaw, pitch = game_scenario.aim_angles(unit(frame.T @ (target-world[last][:3, 3])), a0)
            goal = np.array(self.clamp_aim(aim, yaw, pitch))-animated
        else:
            goal = np.zeros(2)
        offset = self.aim_state.get(aim['name'], np.zeros(2))
        step = goal-offset
        most = aim['degrees_per_s']*dt
        if np.linalg.norm(step) > most:
            step *= most/np.linalg.norm(step)
        offset = offset+step
        if target is not None:                    # an aimed axis stays within the limits
            offset = np.array(self.clamp_aim(aim, *(animated+offset)))-animated
        self.aim_state[aim['name']] = offset
        final = animated+offset
        turn = frame @ rotation_between(now, game_scenario.aim_direction(final[0], final[1], a0)) @ frame.T
        angle = math.acos(float(np.clip((np.trace(turn)-1)/2, -1, 1)))
        if angle < 1e-9:
            return
        axis = np.array([turn[2, 1]-turn[1, 2], turn[0, 2]-turn[2, 0], turn[1, 0]-turn[0, 1]])
        for n in joints:
            self.rotate(world, n, axis_angle(axis, angle/len(joints)), world[n][:3, 3])

    def clamp_aim(self, aim, yaw, pitch):
        return (float(np.clip(yaw, *aim['yaw_degrees'])), float(np.clip(pitch, *aim['pitch_degrees'])))


# --- the scenario player ----------------------------------------------------------------------------------------

def apply(runtime, ground, event):
    kind = event['type']
    if kind == 'set_parameter':
        runtime.set_parameter(event['name'], event['value'])
    elif kind == 'trigger':
        runtime.trigger(event['name'])
    elif kind == 'play':
        runtime.play(event['name'])
    elif kind == 'set_aim':
        runtime.set_aim(event['name'], event['point'])
    elif kind == 'set_limb_target':
        runtime.set_limb_target(event['name'], event['point'], event['weight'])
    elif kind == 'set_ground_enabled':
        runtime.set_ground_enabled(event['enabled'])
    else:
        ground.patch(event['rect'], event['offset'])


def run(package, scenario, scenario_sha256=None):
    """Play a scenario: the player stands in for the game, moving the root, making the calls at their times and
    handing the runtime the ground as it is that frame. Returns the trace. `scenario_sha256` is the SHA-256 of the
    scenario file's bytes (game_scenario.load); a scenario held in memory uses game_scenario.digest_json."""
    package = Path(package)
    runtime = Runtime(package)
    game_scenario.resolve(scenario, runtime.spec)
    ground = game_scenario.Ground(scenario)
    times = game_scenario.frame_times(scenario)
    events, next_event, frames = scenario['events'], 0, []
    for k, t in enumerate(times):
        while next_event < len(events) and events[next_event]['time'] <= t+1e-9:
            apply(runtime, ground, events[next_event])
            next_event += 1
        root = game_scenario.root_matrix(*game_scenario.root_at(scenario, t))
        world = runtime.step(float(t-times[k-1]) if k else 0.0, root, ground.snapshot())
        frames.append({'time': round(float(t), 9), 'root': game_scenario.column_major(root),
                       'joints': [game_scenario.column_major(world[n]) for n in runtime.names]})
    return {'format': game_scenario.TRACE_FORMAT, 'version': 1, 'runtime': {'name': NAME, 'version': VERSION},
            'hashes': {'glb_sha256': terrain.digest(package/runtime.spec['glb']),
                       'runtime_sha256': terrain.digest(package/'character.runtime.json'),
                       'scenario_sha256': scenario_sha256 or game_scenario.digest_json(scenario)},
            'joints': runtime.names, 'matrix_layout': game_scenario.LAYOUT, 'frames': frames}
