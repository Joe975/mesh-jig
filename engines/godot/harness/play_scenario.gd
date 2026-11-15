## Plays a mesh-jig scenario on a game-ready package with the Godot addon and writes the trace game-check reads.
##
##   godot --headless --xr-mode off --script play_scenario.gd -- <package dir> <scenario.json> <trace out>
##         [bake fps] [exact numbers.json]
##
## bake fps: the rate the GLB's clips are resampled at on load (Godot's glTF import resamples every track); give the
## clips' key rate (mesh-jig passes it), or 120, the default, which keeps 24, 30, 40 and 60 Hz keys.
## exact numbers: the scenario with every number written as the hex of its IEEE-754 double (mesh-jig writes it).
## Godot's JSON parser is not correctly rounded: it reads 0.016666666666666666 (1/60 as Python writes it) one unit in
## the last place low, the frame clock drifts by such units, and a stance decision that falls exactly on a window's
## edge can flip. Without the file the scenario's numbers are Godot's own reading.
## No project is needed: the GLB is loaded at run time with GLTFDocument (no editor import), the addon's scripts by
## path. It stands in for a game: it places the character at the scenario's root each frame, makes the scenario's
## calls at their times, hands the character the scenario's heightfield as its ground query, and drives the clock
## itself (the skeleton in manual mode, advanced by each scenario step), so real frame timing does not matter. Each
## frame's joints are read from the skeleton after its modifiers ran (skeleton_updated). Exit 0 on success, 2 on bad
## input.
extends SceneTree

const Character = preload("../addons/mesh_jig_character/mesh_jig_character.gd")
const Heightfield = preload("../addons/mesh_jig_character/ground_heightfield.gd")
const NAME := "godot-mesh-jig-character"
const VERSION := 1
const LAYOUT := "4x4 world matrices as 16 numbers in column-major order (glTF), so entries 12, 13, 14 are the translation; glTF axes (Y up, right-handed), metres"

var character: Character
var captured = null


func _init() -> void:
	process_frame.connect(_run, CONNECT_ONE_SHOT)


func _fail(text: String) -> void:
	printerr("play_scenario: ", text)
	quit(2)


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() < 3 or args.size() > 5:
		_fail("usage: -- <package dir> <scenario.json> <trace out> [bake fps] [exact numbers.json]")
		return
	var package := args[0].replace("\\", "/")
	var scenario_path := args[1].replace("\\", "/")
	var out_path := args[2].replace("\\", "/")
	var runtime_path := package.path_join("character.runtime.json")
	var spec = JSON.parse_string(FileAccess.get_file_as_string(runtime_path))
	var scenario = JSON.parse_string(FileAccess.get_file_as_string(scenario_path))
	if not (spec is Dictionary) or not (scenario is Dictionary):
		_fail("cannot read %s or %s" % [runtime_path, scenario_path])
		return
	var glb_path := package.path_join(spec["glb"])
	var doc := GLTFDocument.new()
	var state := GLTFState.new()
	if doc.append_from_file(glb_path, state) != OK:
		_fail("cannot load " + glb_path)
		return
	if args.size() == 5:
		scenario = exact(scenario, JSON.parse_string(FileAccess.get_file_as_string(args[4])))
	var bake := float(args[3]) if args.size() >= 4 else 120.0
	var model: Node3D = doc.generate_scene(state, bake, false, false)   # keep every track
	var ground := Heightfield.new(scenario["terrain"])
	character = Character.new()
	character.name = "Character"
	character.manual_clock = true
	character.spec = spec
	var clips_path := package.path_join("godot/character.clips.json")
	if FileAccess.file_exists(clips_path):
		character.clip_data = JSON.parse_string(FileAccess.get_file_as_string(clips_path))
	character.ground_query = ground.as_callable()
	character.add_child(model)
	root.add_child(character)                 # the tree is live: _ready builds the runtime from spec
	if character.runtime == null:
		_fail("the character did not build (see the errors above)")
		return
	character.skeleton.skeleton_updated.connect(_capture)
	var times := frame_times(scenario)
	var events: Array = scenario["events"]
	var next_event := 0
	var frames := PackedStringArray()
	for k in times.size():
		var t: float = times[k]
		while next_event < events.size() and float(events[next_event]["time"]) <= t + 1e-9:
			apply(ground, events[next_event])
			next_event += 1
		var root_m := root_matrix(scenario, t)
		character.global_transform = root_m
		captured = null
		character.advance(t - times[k - 1] if k > 0 else 0.0)
		await process_frame
		if captured == null:
			_fail("the skeleton did not update on frame %d" % k)
			return
		var joints := PackedStringArray()
		for m in captured:
			joints.append(column_major(m))
		frames.append('{"time":%s,"root":%s,"joints":[%s]}' % [String.num(t, 9), column_major(root_m), ",".join(joints)])
	var hashes := {"glb_sha256": FileAccess.get_sha256(glb_path), "runtime_sha256": FileAccess.get_sha256(runtime_path),
			"scenario_sha256": FileAccess.get_sha256(scenario_path)}
	var head := {"format": "mesh-jig game trace", "version": 1, "runtime": {"name": NAME, "version": VERSION},
			"hashes": hashes, "joints": Array(character.body.names), "matrix_layout": LAYOUT}
	var text := JSON.stringify(head, "", true)
	text = text.substr(0, text.length() - 1) + ',"frames":[' + ",".join(frames) + "]}\n"
	var f := FileAccess.open(out_path, FileAccess.WRITE)
	if f == null:
		_fail("cannot write " + out_path)
		return
	f.store_string(text)
	f.close()
	print("%d frames to %s" % [times.size(), out_path.get_file()])
	quit(0)


## The parsed scenario with each number replaced by the double whose IEEE-754 bits the exact file gives (16 hex
## digits, little-endian byte order as PackedFloat64Array stores it).
static func exact(parsed, bits):
	if parsed is Dictionary:
		var out := {}
		for k in parsed:
			out[k] = exact(parsed[k], bits[k])
		return out
	if parsed is Array:
		var out := []
		for i in parsed.size():
			out.append(exact(parsed[i], bits[i]))
		return out
	if parsed is float or parsed is int:
		return String(bits).hex_decode().decode_double(0)
	return parsed


## The joints' world matrices after the modifiers ran: the skeleton's world transform times each bone's global pose.
func _capture() -> void:
	var sk := character.skeleton
	var out: Array[Transform3D] = []
	for b in character.body.bone:
		out.append(sk.global_transform * sk.get_bone_global_pose(b))
	captured = out


func apply(ground: Heightfield, e: Dictionary) -> void:
	var kind: String = e["type"]
	if kind == "set_parameter":
		character.set_parameter(e["name"], e["value"])
	elif kind == "trigger":
		character.trigger(e["name"])
	elif kind == "play":
		character.play(e["name"])
	elif kind == "set_aim":
		character.set_aim(e["name"], point(e["point"]))
	elif kind == "set_limb_target":
		character.set_limb_target(e["name"], point(e["point"]), e["weight"])
	elif kind == "set_ground_enabled":
		character.set_ground_enabled(e["enabled"])
	else:
		ground.patch(e["rect"], e["offset"])


static func point(v):
	return null if v == null else Vector3(v[0], v[1], v[2])


## Frame times: 0, then the dt sequence in turn, a step cut short to land exactly on an event time or the end
## (game_scenario.frame_times).
static func frame_times(s: Dictionary) -> PackedFloat64Array:
	var marks := {}
	for e in s["events"]:
		marks[float(e["time"])] = true
	marks[float(s["duration"])] = true
	var sorted_marks := marks.keys()
	sorted_marks.sort()
	var times := PackedFloat64Array([0.0])
	var t := 0.0
	var k := 0
	var end := float(s["duration"])
	var dts: Array = s["dt"]
	while t < end - 1e-12:
		var step := float(dts[k % dts.size()])
		k += 1
		var nxt := t + step
		for m in sorted_marks:
			if m > t + 1e-12:
				if m <= nxt + 1e-9:
					nxt = m
				break
		times.append(nxt)
		t = nxt
	return times


## The root's world matrix at time t: position and yaw (degrees about +Y) linear between keys.
static func root_matrix(s: Dictionary, t: float) -> Transform3D:
	var keys: Array = s["root"]
	var v := [0.0, 0.0, 0.0, 0.0]
	for c in 4:
		v[c] = _interp(keys, t, c + 1)
	var a := deg_to_rad(v[3])
	var basis := Basis(Vector3(cos(a), 0, -sin(a)), Vector3(0, 1, 0), Vector3(sin(a), 0, cos(a)))
	return Transform3D(basis, Vector3(v[0], v[1], v[2]))


static func _interp(keys: Array, t: float, c: int) -> float:
	if t <= float(keys[0][0]):
		return float(keys[0][c])
	for i in keys.size() - 1:
		var t0 := float(keys[i][0])
		var t1 := float(keys[i + 1][0])
		if t <= t1:
			return float(keys[i][c]) + (t - t0) * (float(keys[i + 1][c]) - float(keys[i][c])) / (t1 - t0)
	return float(keys[-1][c])


static func column_major(m: Transform3D) -> String:
	var b := m.basis
	var o := m.origin
	var v := [b.x.x, b.x.y, b.x.z, 0.0, b.y.x, b.y.y, b.y.z, 0.0, b.z.x, b.z.y, b.z.z, 0.0, o.x, o.y, o.z, 1.0]
	var parts := PackedStringArray()
	for x in v:
		parts.append(String.num(x, 7))
	return "[" + ",".join(parts) + "]"
