## The blend graph of character.runtime.json, evaluated in script: step 1 of the reference runtime
## (src/mesh_jig/game_runtime.py, Runtime.clock .. Runtime.animate), function for function with the same names.
##
## Why not AnimationTree: a synced blend1d shares one normalised phase whose frequency lerps between its two points,
## each clip at phase + its phase_offset, and a point with cycle false runs on its own clock. Godot's
## AnimationNodeBlendSpace1D with sync = true plays every point on its own clock (measured on Godot 4.7.2: after
## 0.3 s at blend 0.5 the pose is 0.04 deg from an own-clock blend and 0.90 deg from the shared-phase one), and the
## stance votes that drive ground adaptation need each clip's time and share, which an AnimationTree does not report.
## debug_state() says what every node is doing.
extends RefCounted

const Body = preload("mesh_jig_body.gd")

var spec: Dictionary
var body: Body
var params := {}
var pending_triggers := {}
var pending_plays := {}
var clocks := {}                       # graph node path -> state
var legs := []                         # the runtime file's limbs with role leg
var last_layers := []                  # per layer: [weight, made] of the last frame, for debug_state()


func _init(p_spec: Dictionary, p_body: Body) -> void:
	spec = p_spec
	body = p_body
	for k in spec["parameters"]:
		params[k] = float(spec["parameters"][k]["default"])
	for l in spec["limbs"]:
		if l["role"] == "leg":
			legs.append(l)


# the interface ---------------------------------------------------------------------------------------------------
func set_parameter(name: String, value: float) -> void:
	params[name] = float(value)


func trigger(name: String) -> void:
	pending_triggers[name] = true


func play(name: String) -> void:
	pending_plays[name] = true


# 1. the graph ----------------------------------------------------------------------------------------------------
## Seconds since this node started: 0 on its first frame, then dt more each frame.
func clock(path: String, dt: float) -> float:
	var t = clocks.get(path)
	t = 0.0 if t == null else t + dt
	clocks[path] = t
	return t


func reset(path: String) -> void:
	for key in clocks.keys():
		if key == path or key.begins_with(path + "/") or key.begins_with(path + "#"):
			clocks.erase(key)


func clip_time(clip: String, t: float) -> float:
	var c: Dictionary = spec["clips"][clip]
	return fposmod(t, c["duration_s"]) if c["loop"] else minf(t, c["duration_s"])


## [pose or null, weight, [[clip, clip time, share]]]: joint local transforms and the weight they are laid over the
## layers below with. null adds nothing (a null state, or one-shots when none is playing).
func node(n: Dictionary, path: String, dt: float) -> Array:
	var kind: String = n["type"]
	if kind == "clip":
		var t := clip_time(n["clip"], clock(path, dt))
		return [body.full(n["clip"], t), 1.0, [[n["clip"], t, 1.0]]]
	if kind == "blend1d":
		return blend1d(n, path, dt)
	if kind == "states":
		return states(n, path, dt)
	return oneshots(n, path, dt)


func _dur(p: Dictionary) -> float:
	return spec["clips"][p["clip"]]["duration_s"]


func blend1d(n: Dictionary, path: String, dt: float) -> Array:
	var points: Array = n["points"]
	var x: float = params[n["parameter"]]
	var values := []
	for p in points:
		values.append(float(p["value"]))
	var i := 0
	var w := 0.0
	if x <= values[0]:
		i = 0
		w = 0.0
	elif x >= values[-1]:
		i = points.size() - 2
		w = 1.0
	else:
		for k in points.size() - 1:
			if values[k] <= x:
				i = k
		w = (x - values[i]) / (values[i + 1] - values[i])
	var a: Dictionary = points[i]
	var b: Dictionary = points[i + 1]
	var t := clock(path, dt)       # the blend's own clock: for points off the cycle and unsynced blends
	var poses := []
	var made := []
	if n.get("sync", false):
		# Points on the cycle share one phase; its frequency lerps between them. A point off the cycle (an idle: no
		# leg touches down in it) runs on its own clock and adds nothing to the frequency.
		var on := []
		for pair in [[a, 1.0 - w], [b, w]]:
			if pair[0]["cycle"]:
				on.append(pair)
		var freq := 0.0
		if on.size() == 2:
			for pair in on:
				freq += pair[1] / _dur(pair[0])
		elif on.size() == 1:
			freq = 1.0 / _dur(on[0][0])
		var phase = clocks.get(path + "#phase")
		phase = 0.0 if phase == null else phase + dt * freq
		clocks[path + "#phase"] = phase
		var times := []
		for p in [a, b]:
			times.append(fposmod(phase + p["phase_offset"], 1.0) * _dur(p) if p["cycle"] else fposmod(t, _dur(p)))
		poses = [body.full(a["clip"], times[0]), body.full(b["clip"], times[1])]
		made = [[a["clip"], times[0], 1.0 - w], [b["clip"], times[1], w]]
	else:
		var ks := [i, i + 1]
		var shares := [1.0 - w, w]
		for m in 2:
			var p: Dictionary = points[ks[m]]
			if p.has("clip"):
				var tt := clip_time(p["clip"], t)
				poses.append(body.full(p["clip"], tt))
				made.append([p["clip"], tt, shares[m]])
			else:           # a nested node's own weight is not used inside a blend
				var r := node(p["node"], "%s/%d" % [path, ks[m]], dt)
				poses.append(r[0] if r[0] != null else body.rest())
				for c in r[2]:
					made.append([c[0], c[1], shares[m] * c[2]])
	return [blend(poses[0], poses[1], w), 1.0, made]


## A state that plays one clip has ended when its clock passes the clip's duration.
func ended(n, path: String) -> bool:
	if n == null or n["type"] != "clip":
		return false
	return clocks.get(path, 0.0) >= spec["clips"][n["clip"]]["duration_s"]


func states(n: Dictionary, path: String, dt: float) -> Array:
	if not clocks.has(path + "#states"):
		clocks[path + "#states"] = {"current": n["start"], "previous": null, "elapsed": 0.0}
	var st: Dictionary = clocks[path + "#states"]
	if st["previous"] != null:
		st["elapsed"] += dt
	for tr in n["transitions"]:
		if tr["from"] != st["current"]:
			continue
		var fire: bool = pending_triggers.has(tr["on"]) if tr.has("on") else ended(n["states"][tr["from"]], "%s/%s" % [path, tr["from"]])
		if fire:
			st["previous"] = st["current"]
			st["current"] = tr["to"]
			st["elapsed"] = 0.0
			st["fade"] = float(tr["fade_s"])
			reset("%s/%s" % [path, tr["to"]])
			break
	var u := 1.0 if st["previous"] == null else fade(st["elapsed"], st["fade"])
	var cur = n["states"][st["current"]]
	var b: Array = [null, 0.0, []] if cur == null else node(cur, "%s/%s" % [path, st["current"]], dt)
	if st["previous"] == null or u >= 1.0:
		st["previous"] = null
		return b
	var prev = n["states"][st["previous"]]
	var a: Array = [null, 0.0, []] if prev == null else node(prev, "%s/%s" % [path, st["previous"]], dt)
	var made := []
	for c in a[2]:
		made.append([c[0], c[1], c[2] * (1.0 - u)])
	for c in b[2]:
		made.append([c[0], c[1], c[2] * u])
	if a[0] == null and b[0] == null:
		return [null, 0.0, []]
	if a[0] == null:                      # fading in over the layers below
		return [b[0], u, made]
	if b[0] == null:                      # fading out to the layers below
		return [a[0], 1.0 - u, made]
	return [blend(a[0], b[0], u), 1.0, made]


## Linear cross-fade progress, 0 to 1.
func fade(elapsed: float, seconds: float) -> float:
	return minf(1.0, elapsed / seconds)


func oneshots(n: Dictionary, path: String, dt: float) -> Array:
	var names := pending_plays.keys()
	names.sort()
	for name in names:
		if n["clips"].has(name):
			clocks[path + "#oneshot"] = {"clip": name, "tau": null}       # a replay restarts it
	var st = clocks.get(path + "#oneshot")
	if st == null:
		return [null, 0.0, []]
	st["tau"] = 0.0 if st["tau"] == null else st["tau"] + dt
	var d: float = spec["clips"][st["clip"]]["duration_s"]
	if st["tau"] > d:
		clocks.erase(path + "#oneshot")
		return [null, 0.0, []]
	var f: Dictionary = n["clips"][st["clip"]]
	return [body.full(st["clip"], st["tau"]), oneshot_weight(f, st["tau"], d), [[st["clip"], st["tau"], 1.0]]]


## Fades in from the start and out before the end, linearly.
func oneshot_weight(f: Dictionary, tau: float, d: float) -> float:
	return maxf(0.0, minf(1.0, minf(tau / f["fade_in_s"], (d - tau) / f["fade_out_s"])))


## The layers, in order: a later layer replaces the joints in its mask by its weight. Also, for each leg, the clips
## that made its tip joint's pose, with their shares (the stance vote). Returns [pose, votes].
func animate(dt: float) -> Array:
	var pose := body.rest()
	var votes := {}
	for l in legs:
		votes[l["name"]] = []
	last_layers = []
	var layers: Array = spec["graph"]["layers"]
	for k in layers.size():
		var layer: Dictionary = layers[k]
		var r := node(layer["node"], str(k), dt)
		last_layers.append([r[1] if r[0] != null else 0.0, r[2]])
		var p = r[0]
		var w: float = r[1]
		if p == null or w <= 0.0:
			continue
		var mask = layer.get("mask_joints")
		var ids := []
		if mask:
			for name in mask:
				ids.append(body.index[name])
		else:
			ids = range(body.names.size())
		for i in ids:
			blend_trs(pose, p, i, w)
		for leg in legs:
			if mask == null or mask.has(leg["joints"][-1]):
				var mixed := []
				for c in votes[leg["name"]]:
					mixed.append([c[0], c[1], c[2] * (1.0 - w)])
				for c in r[2]:
					mixed.append([c[0], c[1], c[2] * w])
				votes[leg["name"]] = mixed
	pending_triggers.clear()
	pending_plays.clear()
	return [pose, votes]


# blending: lerp translation and scale, shortest-arc nlerp rotation (game_measure.blend) -----------------------------
static func nlerp(qa: Quaternion, qb: Quaternion, w: float) -> Quaternion:
	if qa.dot(qb) < 0.0:
		qb = -qb
	var q := qa * (1.0 - w) + qb * w
	return q.normalized()


## Joint local transforms: any share w of b.
static func blend(a: Array, b: Array, w: float) -> Array:
	var tt := PackedVector3Array()
	var rr: Array[Quaternion] = []
	var ss := PackedVector3Array()
	for i in a[0].size():
		tt.append(a[0][i] * (1.0 - w) + b[0][i] * w)
		rr.append(nlerp(a[1][i], b[1][i], w))
		ss.append(a[2][i] * (1.0 - w) + b[2][i] * w)
	return [tt, rr, ss]


## One joint of `pose` laid toward the same joint of `p` by w, in place.
static func blend_trs(pose: Array, p: Array, i: int, w: float) -> void:
	pose[0][i] = pose[0][i] * (1.0 - w) + p[0][i] * w
	pose[1][i] = nlerp(pose[1][i], p[1][i], w)
	pose[2][i] = pose[2][i] * (1.0 - w) + p[2][i] * w


## What each layer did on the last frame, for a person debugging a character: weights, clips at their times and
## shares, the states each states node is in, and every parameter.
func debug_state() -> Dictionary:
	var layers := []
	var spec_layers: Array = spec["graph"]["layers"]
	for k in last_layers.size():
		var made := []
		for c in last_layers[k][1]:
			made.append("%s @ %.3f s x %.2f" % [c[0], c[1], c[2]])
		var row := {"name": spec_layers[k]["name"], "weight": last_layers[k][0], "clips": made}
		var st = clocks.get(str(k) + "#states")
		if st != null:
			row["state"] = st["current"]
			row["fading_from"] = st["previous"]
		layers.append(row)
	return {"parameters": params.duplicate(), "layers": layers}
