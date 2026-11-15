## The reference runtime (src/mesh_jig/game_runtime.py) ported to GDScript, steps 2 to 5: forward kinematics, ground
## adaptation (stance lock, foot height with clearance points and look-ahead, body height and tilt, leg solves), aims,
## then game-given limb targets. Ported from the reference at commit 871e2ba4.
## Functions keep the reference's names and order so a change there can be ported by diff. Every smoothing step is
## in seconds. The blend graph (step 1) is mesh_jig_graph.gd; mesh_jig_modifier.gd runs this each skeleton update.
##
## It works on `world`, an Array of Transform3D (one world matrix per runtime joint, in skeleton.joints order), as the
## reference works on its dict of 4x4 matrices. It never moves the character: step() is given the root's world
## matrix (the GLB scene root's global transform, which the game moves) and returns joint world matrices.
extends RefCounted

const Body = preload("mesh_jig_body.gd")
const Graph = preload("mesh_jig_graph.gd")
const FABRIK_ITERATIONS := 30         # fixed, so every port does the same work
const REACH := 0.999                  # a chain is never asked to straighten fully: a straight knee has no pole
const LOOKAHEAD_SAMPLES := 8          # points along a swinging foot's look-ahead where the ground is read

var spec: Dictionary
var body: Body
var graph: Graph
var legs := []
var aim_targets := {}
var limb_targets := {}
var aim_state := {}
var ground_on := true
var ground_weight := 1.0
var leg_state := {}
var body_offset := 0.0
var tilt := Basis.IDENTITY
var first := true
var last_planted := {}                # leg name -> planted, for debug_state()


func _init(p_spec: Dictionary, p_body: Body, p_graph: Graph) -> void:
	spec = p_spec
	body = p_body
	graph = p_graph
	for l in spec["limbs"]:
		if l["role"] == "leg":
			legs.append(l)
			leg_state[l["name"]] = {}


# the interface (the graph's calls go to the graph) -------------------------------------------------------------
func set_aim(name: String, point) -> void:
	aim_targets[name] = point


func set_limb_target(name: String, point, weight: float = 1.0) -> void:
	limb_targets[name] = null if point == null else [point, float(weight)]


func set_ground_enabled(enabled: bool) -> void:
	ground_on = enabled


## One frame: the reference's Runtime.step. Returns the joint world matrices.
func step(dt: float, root: Transform3D, ground: Callable) -> Array[Transform3D]:
	if first:                    # the ground weight starts where the calls before the first frame left it
		ground_weight = 1.0 if ground_on else 0.0
	var r := graph.animate(dt)
	var model := body.model(r[0])
	var world: Array[Transform3D] = []
	for i in model.size():
		world.append(place(root, model[i]))
	adapt(world, root, r[1], ground, dt)
	for a in spec["aims"]:                  # aims before limb targets: an aim's chain may carry a limb
		aim(world, root, a, dt)
	for limb in spec["limbs"]:
		var target = limb_targets.get(limb["name"])
		if target != null and target[1] > 0:
			var tip := world[body.index[limb["joints"][-1]]].origin
			solve(world, root, limb, tip + target[1] * (target[0] - tip))
	first = false
	return world


## 2. A joint's world matrix: the root's world matrix times its model-space matrix. Both are glTF axes, which are
## Godot's axes (Y up, right-handed, -Z forward), so nothing flips.
func place(root: Transform3D, m: Transform3D) -> Transform3D:
	return root * m


# small geometry ------------------------------------------------------------------------------------------------
static func unit(v: Vector3) -> Vector3:
	var n := v.length()
	return v / n if n > 1e-12 else v


## Dot and cross products in double precision. Vector3 is single precision in Godot: a.dot(b) of two unit vectors
## less than 3.4e-4 rad apart rounds to exactly 1, so a small turn read from it vanishes (0.15 mm at the end of a
## 0.44 m shin, measured on the raise scenario). The reference runs in double precision throughout.
static func _dot(a: Vector3, b: Vector3) -> float:
	return float(a.x) * b.x + float(a.y) * b.y + float(a.z) * b.z


static func _cross(a: Vector3, b: Vector3) -> Vector3:
	return Vector3(float(a.y) * b.z - float(a.z) * b.y, float(a.z) * b.x - float(a.x) * b.z, float(a.x) * b.y - float(a.y) * b.x)


## A 3x3 from its rows, as numpy writes it.
static func rows(r0: Vector3, r1: Vector3, r2: Vector3) -> Basis:
	return Basis(Vector3(r0.x, r1.x, r2.x), Vector3(r0.y, r1.y, r2.y), Vector3(r0.z, r1.z, r2.z))


## The shortest rotation that turns direction a onto direction b. Twist about the direction is kept.
static func rotation_between(a: Vector3, b: Vector3) -> Basis:
	a = unit(a)
	b = unit(b)
	var v := _cross(a, b)
	var c := clampf(_dot(a, b), -1.0, 1.0)
	if c > 0.0 and _dot(v, v) < 1e-24:                    # the reference's c > 1-1e-12, read from the cross product
		return Basis.IDENTITY
	if c < -1.0 + 1e-9:                                  # opposite: turn half way round any perpendicular axis
		var m := absf(a.x)
		var e := Vector3(1, 0, 0)
		if absf(a.y) < m:
			m = absf(a.y)
			e = Vector3(0, 1, 0)
		if absf(a.z) < m:
			e = Vector3(0, 0, 1)
		var ax := unit(a.cross(e))
		return rows(Vector3(2 * ax.x * ax.x - 1, 2 * ax.x * ax.y, 2 * ax.x * ax.z),
				Vector3(2 * ax.y * ax.x, 2 * ax.y * ax.y - 1, 2 * ax.y * ax.z),
				Vector3(2 * ax.z * ax.x, 2 * ax.z * ax.y, 2 * ax.z * ax.z - 1))
	# I + K + K^2/(1+c), which is cI + K + v v^T/(1+c)
	var k := 1.0 / (1.0 + c)
	return rows(Vector3(c + v.x * v.x * k, -v.z + v.x * v.y * k, v.y + v.x * v.z * k),
			Vector3(v.z + v.y * v.x * k, c + v.y * v.y * k, -v.x + v.y * v.z * k),
			Vector3(-v.y + v.z * v.x * k, v.x + v.z * v.y * k, c + v.z * v.z * k))


static func axis_angle(axis: Vector3, angle: float) -> Basis:
	var u := unit(axis)
	var x := u.x
	var y := u.y
	var z := u.z
	var c := cos(angle)
	var s := sin(angle)
	return rows(Vector3(c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s),
			Vector3(y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s),
			Vector3(z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)))


static func yaw_matrix(radians: float) -> Basis:
	return axis_angle(Vector3(0, 1, 0), radians)


## The share of the way to a target covered in dt by a follower with this time constant (dt-correct).
static func smooth(dt: float, seconds: float) -> float:
	return 1.0 if seconds <= 0 else 1.0 - exp(-dt / seconds)


## The rotation angle of a rotation matrix, and its axis (unnormalised), as the reference reads them. The angle is
## taken with atan2 from the skew part as well as the trace: acos of the trace alone loses angles under 3.4e-4 rad in
## single precision.
static func _angle(r: Basis) -> float:
	var tr := float(r.x.x) + r.y.y + r.z.z
	var s := sqrt(_dot(_axis(r), _axis(r))) / 2.0
	return atan2(s, (tr - 1.0) / 2.0)


static func _axis(r: Basis) -> Vector3:
	# r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]; r[i,j] is column j, component i
	return Vector3(r.y.z - r.z.y, r.z.x - r.x.z, r.x.y - r.y.x)


# 3. ground adaptation --------------------------------------------------------------------------------------------
## Planted when the clips that make the leg's pose, by share, say stance at their current phase.
func planted(leg: Dictionary, votes: Array) -> bool:
	var total := 0.0
	var yes := 0.0
	for v in votes:
		var c: Dictionary = spec["clips"][v[0]]
		var phase: float = v[1] / c["duration_s"]
		var windows: Array = c["legs"].get(leg["name"], {}).get("stance_phase", [])
		total += v[2]
		var hit := false
		for w in windows:
			if (w[0] <= phase and phase <= w[1]) or (w[0] <= phase + 1 and phase + 1 <= w[1]):
				hit = true
				break
		if hit:
			yes += v[2]
	return total > 0 and yes >= total / 2


func ground_height(ground: Callable, x: float, z: float) -> float:
	return ground.call(x, z)["height"]


## The planted-foot lock, as a world correction added to the animated contact: a horizontal shift and a turn about
## +Y. While the leg is planted, the shift cancels the world drift of the contact points that are on the
## animation's ground in this frame and the last (so a foot rolling onto its toe keeps its toe still), and the turn
## cancels the root's change of yaw. The shift is never longer than relock_m: past it the foot slides, only as fast as
## the animation pulls it further. When the leg cannot reach, and whenever it is not planted, both fade to zero with
## time constant blend_s: the foot returns to the animation without a jump.
## Returns [shift (Vector2, world x and z), turn].
func lock(state: Dictionary, points: PackedVector3Array, support: PackedByteArray, is_planted: bool, root_yaw: float,
		dt: float, far: bool) -> Array:
	var g: Dictionary = spec["ground"]
	if is_planted and state.get("planted", false):
		var old: PackedVector3Array = state["points"]
		var was: PackedByteArray = state["support"]
		var sum := Vector2.ZERO
		var n := 0
		for i in points.size():
			if support[i] and was[i]:
				sum += Vector2(points[i].x - old[i].x, points[i].z - old[i].z)
				n += 1
		if n > 0:
			state["shift"] = state["shift"] - sum / n
		state["turn"] -= root_yaw - state["root_yaw"]
	if not is_planted or far:
		var f := 1.0 - smooth(dt, g["blend_s"])
		state["shift"] = state["shift"] * f
		state["turn"] = state["turn"] * f
	var shifted: Vector2 = state["shift"]
	var length := sqrt(float(shifted.x) * shifted.x + float(shifted.y) * shifted.y)
	if length > g["relock_m"]:
		state["shift"] = shifted * (g["relock_m"] / length)
	state["points"] = points
	state["support"] = support
	state["planted"] = is_planted
	state["root_yaw"] = root_yaw
	return [state["shift"], state["turn"]]


## How far a leg's tip may be from its first joint: the bend-limited reach of a two-bone leg.
func reach(leg: Dictionary) -> float:
	var lengths: Array = leg["segment_lengths_m"]
	if leg["solver"] == "two_bone":
		var l1: float = lengths[0]
		var l2: float = lengths[1]
		var hi := deg_to_rad(leg["bend_degrees"][1])
		return minf(REACH * (l1 + l2), sqrt(l1 * l1 + l2 * l2 - 2 * l1 * l2 * cos(hi)))
	var total := 0.0
	for x in lengths:
		total += x
	return REACH * total


## The tip's height. Each point of the foot (its contact points and its clearance points) asks for the ground under
## it; the tip goes as low as keeps every point at least `lift` above what it asks for. While the leg is not planted,
## a point also looks ahead along the foot's horizontal velocity for lookahead_s, read at LOOKAHEAD_SAMPLES points.
## Ground a share s of the way there that stands above the ground continued from just behind the point (one sample's
## spacing back, so a uniform slope stands above nothing) counts for (1 - s) of that rise: a riser coming up lifts the
## foot gradually instead of at once, and a slope does not lift it. The look-ahead's weight rises over blend_s after
## lift-off and falls over blend_s after touchdown, so the height never jumps. Kept in one function, as the
## reference's foot_height, so a change to the rule ports alone.
func foot_height(leg: Dictionary, state: Dictionary, ground: Callable, tip_xz: Vector2, rot: Basis, lift: float,
		is_planted: bool, dt: float) -> float:
	var g: Dictionary = spec["ground"]
	var offsets := PackedVector3Array()
	for p in leg["contact"]["points"]:
		offsets.append(rot * Vector3(p[0], p[1], p[2]))
	for p in leg.get("clearance", {}).get("points", []):
		offsets.append(rot * Vector3(p[0], p[1], p[2]))
	var previous = state.get("tip_xz")
	if previous != null and dt > 0:
		state["velocity"] = [(float(tip_xz.x) - previous[0]) / dt, (float(tip_xz.y) - previous[1]) / dt]
	state["tip_xz"] = [float(tip_xz.x), float(tip_xz.y)]
	var step_w: float = dt / g["blend_s"] if g["blend_s"] > 0 else 1.0
	var ahead: float = maxf(0.0, state.get("ahead", 0.0) - step_w) if is_planted else minf(1.0, state.get("ahead", 0.0) + step_w)
	state["ahead"] = ahead
	var velocity: Array = state.get("velocity", [0.0, 0.0])
	var dx: float = velocity[0] * g.get("lookahead_s", 0.0)
	var dz: float = velocity[1] * g.get("lookahead_s", 0.0)
	var best := -INF
	for o in offsets:
		var x := float(tip_xz.x) + o.x
		var z := float(tip_xz.y) + o.z
		var here := ground_height(ground, x, z)
		var need := here
		if ahead > 0 and dx * dx + dz * dz > 1e-12:
			var back := ground_height(ground, x - dx / LOOKAHEAD_SAMPLES, z - dz / LOOKAHEAD_SAMPLES)
			for k in range(1, LOOKAHEAD_SAMPLES + 1):
				var s := float(k) / LOOKAHEAD_SAMPLES
				var rise := ground_height(ground, x + s * dx, z + s * dz) - (here + k * (here - back))
				need = maxf(need, here + ahead * (1 - s) * rise)
		best = maxf(best, need + lift - o.y)
	return best


func adapt(world: Array[Transform3D], root: Transform3D, votes: Dictionary, ground: Callable, dt: float) -> void:
	var g = spec["ground"]
	if legs.is_empty() or g == null:
		return
	var rate: float = dt / g["blend_s"] if g["blend_s"] > 0 else 1.0
	ground_weight = minf(1.0, ground_weight + rate) if ground_on else maxf(0.0, ground_weight - rate)
	var gw := ground_weight
	if gw <= 0:
		for s in leg_state.values():
			s.clear()
		body_offset = 0.0
		tilt = Basis.IDENTITY
		return
	var root_yaw := atan2(root.basis.z.x, root.basis.z.z)
	var anim_ground := (root * Vector3(0, body.anim_ground, 0)).y
	var gap: float = spec["measurement"]["contact_gap_m"]
	var plan := {}
	for leg in legs:
		var tip: int = body.index[leg["joints"][-1]]
		var s: Dictionary = leg_state[leg["name"]]
		if not s.has("shift"):
			s["shift"] = Vector2.ZERO
		if not s.has("turn"):
			s["turn"] = 0.0
		var w := world[tip]
		var points := PackedVector3Array()
		var mean := Vector3.ZERO
		for p in leg["contact"]["points"]:
			var q := w * Vector3(p[0], p[1], p[2])
			points.append(q)
			mean += q
		mean /= points.size()
		var clearance := PackedFloat64Array()                # each contact point's height in the animation
		var support := PackedByteArray()
		for q in points:
			clearance.append(q.y - anim_ground)
			support.append(1 if q.y - anim_ground <= gap else 0)
		var hip := world[body.index[leg["joints"][0]]].origin
		var leg_reach := reach(leg)
		var is_planted := planted(leg, votes[leg["name"]])
		last_planted[leg["name"]] = is_planted
		var held: Vector2 = s["shift"]
		var far := (Vector2(hip.x, hip.z) - Vector2(mean.x, mean.z) - held).length() > leg_reach
		# the lock holds while the clips say planted and after, while any contact point is still on the animation's
		# ground (a toe rolling off): fading it then would slide the toe
		var touching := false
		for x in support:
			touching = touching or x == 1
		var locked := lock(s, points, support, is_planted or touching, root_yaw, dt, far)
		var shift: Vector2 = locked[0]
		var rot := yaw_matrix(locked[1]) * w.basis
		var offset_mean := Vector3.ZERO
		for p in leg["contact"]["points"]:
			offset_mean += rot * Vector3(p[0], p[1], p[2])
		offset_mean /= leg["contact"]["points"].size()
		var tip_xz := Vector2(mean.x, mean.z) + shift - Vector2(offset_mean.x, offset_mean.z)
		# the contact's lowest point as high above the real ground as the animation has it above its own ground (never
		# below it), with the rest of the foot kept out of the ground too
		var low := INF
		for c in clearance:
			low = minf(low, c)
		var y := foot_height(leg, s, ground, tip_xz, rot, maxf(0.0, low), is_planted, dt)
		plan[leg["name"]] = [Vector3(tip_xz.x, y, tip_xz.y), rot, is_planted, w.origin, hip, leg_reach]
	# the body: as low as the lowest planted target asks (smoothed), and at once low enough for every planted leg to
	# reach, within the drop and rise limits
	var chosen := []
	for p in plan.values():
		if p[2]:
			chosen.append(p)
	if chosen.is_empty():
		chosen = plan.values()
	var want := INF
	for p in chosen:
		want = minf(want, p[0].y - p[3].y)
	var bound := INF
	for p in chosen:
		var flat := Vector2(p[0].x - p[4].x, p[0].z - p[4].z).length()
		bound = minf(bound, p[0].y - p[4].y + sqrt(maxf(p[5] * p[5] - flat * flat, 0.0)))
	want = clampf(want, -g["drop_m_max"], g["rise_m_max"])
	body_offset = want if first else body_offset + (want - body_offset) * smooth(dt, g["blend_s"])
	var offset := clampf(minf(body_offset, bound), -g["drop_m_max"], g["rise_m_max"])
	var body_joint: int = body.index[spec["body"]["joint"]]
	translate(world, body_joint, Vector3(0, gw * offset, 0))
	if g["tilt"]:
		var pts := PackedVector3Array()
		for p in plan.values():
			if p[2]:
				pts.append(p[0])
		var goal := Basis.IDENTITY
		if pts.size() >= 3:
			var normal := plane_normal(pts)
			normal = normal if normal.y > 0 else -normal
			goal = rotation_between(Vector3(0, 1, 0), normal)
		tilt = goal if first else slerp(tilt, goal, smooth(dt, g["blend_s"]))
		# laid on by the ground weight, so it fades out with the adaptation and is gone when the weight is 0
		rotate(world, body_joint, slerp(Basis.IDENTITY, tilt, gw), world[body_joint].origin)
	for leg in legs:
		var p: Array = plan[leg["name"]]
		var tip: int = body.index[leg["joints"][-1]]
		solve(world, root, leg, world[tip].origin + gw * (p[0] - world[tip].origin))
		# the tip turns to its target rotation by the ground weight
		rotate(world, tip, slerp(Basis.IDENTITY, p[1] * world[tip].basis.transposed(), gw), world[tip].origin)


## The unit normal of the plane fitted through the points (least squares): the reference takes it from an SVD; here
## it is the covariance's eigenvector of least eigenvalue (the same direction), by Jacobi rotations.
static func plane_normal(pts: PackedVector3Array) -> Vector3:
	var c := Vector3.ZERO
	for p in pts:
		c += p
	c /= pts.size()
	var a := [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
	for p in pts:
		var d := p - c
		for i in 3:
			for j in 3:
				a[i][j] += d[i] * d[j]
	var v := [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
	for sweep in 50:
		var off: float = absf(a[0][1]) + absf(a[0][2]) + absf(a[1][2])
		if off < 1e-15:
			break
		for pq in [[0, 1], [0, 2], [1, 2]]:
			var p: int = pq[0]
			var q: int = pq[1]
			if absf(a[p][q]) < 1e-18:
				continue
			var theta: float = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
			var t: float = signf(theta) / (absf(theta) + sqrt(theta * theta + 1.0)) if theta != 0.0 else 1.0
			var cs := 1.0 / sqrt(t * t + 1.0)
			var sn := t * cs
			for k in 3:
				var akp: float = a[k][p]
				var akq: float = a[k][q]
				a[k][p] = cs * akp - sn * akq
				a[k][q] = sn * akp + cs * akq
			for k in 3:
				var apk: float = a[p][k]
				var aqk: float = a[q][k]
				a[p][k] = cs * apk - sn * aqk
				a[q][k] = sn * apk + cs * aqk
			for k in 3:
				var vkp: float = v[k][p]
				var vkq: float = v[k][q]
				v[k][p] = cs * vkp - sn * vkq
				v[k][q] = sn * vkp + cs * vkq
	var least := 0
	for i in [1, 2]:
		if a[i][i] < a[least][least]:
			least = i
	return unit(Vector3(v[0][least], v[1][least], v[2][least]))


static func slerp(a: Basis, b: Basis, u: float) -> Basis:
	var r := b * a.transposed()
	var angle := _angle(r)
	if angle < 1e-9:
		return b
	return axis_angle(_axis(r), angle * u) * a


# moving parts of the skeleton ------------------------------------------------------------------------------------
func subtree(i: int) -> Array:
	var out := []
	var stack := [i]
	while not stack.is_empty():
		var n: int = stack.pop_back()
		out.append(n)
		stack.append_array(body.children[n])
	return out


func translate(world: Array[Transform3D], i: int, offset: Vector3) -> void:
	for n in subtree(i):
		var m := world[n]
		m.origin += offset
		world[n] = m


func rotate(world: Array[Transform3D], i: int, r: Basis, pivot: Vector3) -> void:
	var m := Transform3D(r, pivot - r * pivot)
	for n in subtree(i):
		world[n] = m * world[n]


# 3, 4. solvers ---------------------------------------------------------------------------------------------------
func solve(world: Array[Transform3D], root: Transform3D, limb: Dictionary, target: Vector3) -> void:
	if limb["solver"] == "two_bone":
		two_bone(world, root, limb, target)
	elif limb["solver"] == "chain":
		chain(world, limb, target)


static func _reach_at(l1: float, l2: float, angle: float) -> float:
	return sqrt(maxf(l1 * l1 + l2 * l2 - 2 * l1 * l2 * cos(angle), 1e-12))


## Analytic: the knee in the plane of hip, target and pole; the reach clamped so the knee's inner angle stays within
## bend_degrees. Each bone is turned by the shortest rotation onto its new direction.
func two_bone(world: Array[Transform3D], root: Transform3D, limb: Dictionary, target: Vector3) -> void:
	var a: int = body.index[limb["joints"][0]]
	var b: int = body.index[limb["joints"][1]]
	var c: int = body.index[limb["joints"][2]]
	var pa := world[a].origin
	var pb := world[b].origin
	var pc := world[c].origin
	var l1 := (pb - pa).length()
	var l2 := (pc - pb).length()
	var lo := deg_to_rad(limb["bend_degrees"][0])
	var hi := deg_to_rad(limb["bend_degrees"][1])
	var d := target - pa
	var r := clampf(d.length(), _reach_at(l1, l2, lo), minf(_reach_at(l1, l2, hi), REACH * (l1 + l2)))
	var direction := unit(d) if d.length() > 1e-9 else unit(pc - pa)
	var pv: Array = limb["pole"]
	var pole := root.basis * Vector3(pv[0], pv[1], pv[2])
	var side := pole - direction * pole.dot(direction)
	if side.length() < 1e-6:                          # pole along the leg: keep the knee where it is
		side = (pb - pa) - direction * (pb - pa).dot(direction)
	var cos_a := clampf((l1 * l1 + r * r - l2 * l2) / (2 * l1 * r), -1.0, 1.0)
	var knee := pa + direction * l1 * cos_a + unit(side) * l1 * sqrt(1 - cos_a * cos_a)
	rotate(world, a, rotation_between(pb - pa, knee - pa), pa)
	pb = world[b].origin
	pc = world[c].origin
	rotate(world, b, rotation_between(pc - pb, pa + direction * r - pb), pb)


## FABRIK with the root pinned and a fixed number of passes; an out-of-reach target straightens the chain.
static func fabrik(points: PackedVector3Array, target: Vector3) -> PackedVector3Array:
	points = PackedVector3Array(points)
	var lengths := PackedFloat64Array()
	var total := 0.0
	for k in points.size() - 1:
		lengths.append((points[k + 1] - points[k]).length())
		total += lengths[k]
	var root := points[0]
	if (target - root).length() >= total:
		var direction := unit(target - root)
		for k in lengths.size():
			points[k + 1] = points[k] + direction * lengths[k]
		return points
	for _pass in FABRIK_ITERATIONS:
		points[points.size() - 1] = target
		for k in range(points.size() - 2, -1, -1):
			points[k] = points[k + 1] + unit(points[k] - points[k + 1]) * lengths[k]
		points[0] = root
		for k in points.size() - 1:
			points[k + 1] = points[k] + unit(points[k + 1] - points[k]) * lengths[k]
	return points


## Solve the positions with FABRIK, then turn each joint, root first, by the shortest rotation that aims its segment
## at its solved next point; a bend past bend_degrees_max from the segment before is held there.
func chain(world: Array[Transform3D], limb: Dictionary, target: Vector3) -> void:
	var joints := []
	var start := PackedVector3Array()
	for n in limb["joints"]:
		joints.append(body.index[n])
		start.append(world[body.index[n]].origin)
	var goal := fabrik(start, target)
	var limit := deg_to_rad(limb.get("bend_degrees_max", 180))
	for k in joints.size() - 1:
		var here := world[joints[k]].origin
		var nxt := world[joints[k + 1]].origin
		var want := unit(goal[k + 1] - here)
		if k > 0:
			var before := unit(here - world[joints[k - 1]].origin)
			var angle := acos(clampf(before.dot(want), -1.0, 1.0))
			var axis := before.cross(want)
			if angle > limit and axis.length() > 1e-9:
				want = unit(axis_angle(axis, limit) * before)
		rotate(world, joints[k], rotation_between(nxt - here, want), here)


# 5. aims ---------------------------------------------------------------------------------------------------------
## Yaw and pitch, degrees, of direction v relative to an aim's axis at rest, both in the frame of the aim's first
## joint's parent (game_scenario.aim_angles).
static func aim_angles(v: Vector3, a: Vector3) -> Vector2:
	var vh := Vector3(v.x, 0, v.z)
	var ah := Vector3(a.x, 0, a.z)
	var yaw := rad_to_deg(atan2(ah.cross(vh).y, ah.dot(vh)))
	var pitch := rad_to_deg(asin(clampf(v.y, -1.0, 1.0)) - asin(clampf(a.y, -1.0, 1.0)))
	return Vector2(yaw, pitch)


## The unit direction with these aim angles: the inverse of aim_angles (game_scenario.aim_direction).
static func aim_direction(yaw: float, pitch: float, a: Vector3) -> Vector3:
	var ah := Vector3(a.x, 0, a.z)
	ah = ah / ah.length() if ah.length() > 1e-12 else Vector3(0, 0, -1)
	var y := deg_to_rad(yaw)
	var h := Vector3(cos(y) * ah.x + sin(y) * ah.z, 0, -sin(y) * ah.x + cos(y) * ah.z)
	var e := asin(clampf(a.y, -1.0, 1.0)) + deg_to_rad(pitch)
	return h * cos(e) + Vector3(0, 1, 0) * sin(e)


func clamp_aim(a: Dictionary, yaw: float, pitch: float) -> Vector2:
	return Vector2(clampf(yaw, a["yaw_degrees"][0], a["yaw_degrees"][1]), clampf(pitch, a["pitch_degrees"][0], a["pitch_degrees"][1]))


## Turn the chain so the last joint's axis points at the target. Yaw and pitch are measured in the frame of the first
## joint's parent from the axis at rest. The aim keeps an offset from the animation; the offset moves at most
## degrees_per_s (a step in yaw and pitch together) toward the one that points at the target clamped to the limits,
## or toward zero with no target, so the animation itself is never filtered. The turn is shared equally by the
## chain's joints.
func aim(world: Array[Transform3D], root: Transform3D, a: Dictionary, dt: float) -> void:
	var joints := []
	for n in a["joints"]:
		joints.append(body.index[n])
	var last: int = joints[-1]
	var parent := body.parent[joints[0]]
	var frame := (world[parent] if parent >= 0 else root).basis
	var rest := body.rest_model[parent].basis if parent >= 0 else Basis.IDENTITY
	var am: Array = a["axis_model"]
	var al: Array = a["axis_local"]
	var a0 := unit(rest.transposed() * Vector3(am[0], am[1], am[2]))
	var now := unit(frame.transposed() * (world[last].basis * Vector3(al[0], al[1], al[2])))
	var animated := aim_angles(now, a0)
	var target = aim_targets.get(a["name"])
	var goal := Vector2.ZERO
	if target != null:
		var ang := aim_angles(unit(frame.transposed() * (target - world[last].origin)), a0)
		goal = clamp_aim(a, ang.x, ang.y) - animated
	var offset: Vector2 = aim_state.get(a["name"], Vector2.ZERO)
	var step_v := goal - offset
	var most: float = a["degrees_per_s"] * dt
	if step_v.length() > most:
		step_v *= most / step_v.length()
	offset = offset + step_v
	if target != null:                    # an aimed axis stays within the limits
		var both := animated + offset
		offset = clamp_aim(a, both.x, both.y) - animated
	aim_state[a["name"]] = offset
	var final := animated + offset
	var turn := frame * rotation_between(now, aim_direction(final.x, final.y, a0)) * frame.transposed()
	var angle := _angle(turn)
	if angle < 1e-9:
		return
	var axis := _axis(turn)
	for n in joints:
		rotate(world, n, axis_angle(axis, angle / joints.size()), world[n].origin)


## What ground adaptation and the aims are doing, for a person debugging a character.
func debug_state() -> Dictionary:
	var aims := {}
	for k in aim_state:
		aims[k] = {"offset_yaw_pitch_degrees": aim_state[k], "target": aim_targets.get(k)}
	return {"ground_on": ground_on, "ground_weight": ground_weight, "body_offset_m": body_offset,
			"planted": last_planted.duplicate(), "aims": aims, "limb_targets": limb_targets.duplicate()}
