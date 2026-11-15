## Headless checks of the addon's game-facing helpers on a package (tests/test_game_godot.py runs it):
##   godot --headless --xr-mode off --script test_addon.gd -- <package dir> <socket name>
## A ground query whose object the game keeps no reference to stays alive; socket() follows its joint as the runtime
## poses it. Prints FAIL lines and exits 1 on a failure, 0 when every check holds.
extends SceneTree

const Character = preload("../addons/mesh_jig_character/mesh_jig_character.gd")
const Heightfield = preload("../addons/mesh_jig_character/ground_heightfield.gd")

var failures := 0


func _init() -> void:
	process_frame.connect(_run, CONNECT_ONE_SHOT)


func check(ok: bool, what: String) -> void:
	if not ok:
		failures += 1
		printerr("FAIL ", what)


func _run() -> void:
	var args := OS.get_cmdline_user_args()
	var package := args[0].replace("\\", "/")
	var doc := GLTFDocument.new()
	var state := GLTFState.new()
	doc.append_from_file(package.path_join("character.glb"), state)
	var c := Character.new()
	c.spec = JSON.parse_string(FileAccess.get_file_as_string(package.path_join("character.runtime.json")))
	c.clip_data = JSON.parse_string(FileAccess.get_file_as_string(package.path_join("godot/character.clips.json")))
	c.manual_clock = true
	c.add_child(doc.generate_scene(state, 30.0, false, false))
	root.add_child(c)
	# a ground query made inline: nothing but the character holds the heightfield
	c.set_ground_query(Heightfield.new({"x": [-5, 5], "z": [-5, 5], "heights": [[0.0, 0.0], [0.0, 0.0]]}).as_callable())
	var at: Node3D = c.socket(args[1])
	check(at != null and c.socket(args[1]) == at, "socket() returns one node per socket")
	var s: Dictionary = {}
	for x in c.spec["sockets"]:
		if x["name"] == args[1]:
			s = x
	var joint: int = c.body.bone[c.body.index[s["joint"]]]
	if not c.spec["oneshots"].is_empty():
		c.play(c.spec["oneshots"][0])
	var expected := [Transform3D.IDENTITY]
	c.skeleton.skeleton_updated.connect(func(): expected[0] = c.skeleton.global_transform * c.skeleton.get_bone_global_pose(joint))
	var worst := 0.0
	var moved := 0.0
	var rest: Vector3 = (c.skeleton.global_transform * c.skeleton.get_bone_global_rest(joint)).origin
	for k in 40:
		c.advance(1.0 / 60.0)
		await process_frame
		var o: Array = s["offset_local"]
		var want: Vector3 = expected[0] * Vector3(o[0], o[1], o[2])
		worst = maxf(worst, at.global_position.distance_to(want))
		moved = maxf(moved, expected[0].origin.distance_to(rest))
	check(c.ground_query.is_valid(), "the ground query is still valid after 40 frames")
	check(worst < 1e-5, "the socket follows its joint as posed (worst %.6f m)" % worst)
	check(moved > 1e-3, "the joint moved, so the check means something (%.6f m)" % moved)
	print("socket worst %.7f m over 40 frames; joint moved %.4f m; failures %d" % [worst, moved, failures])
	quit(1 if failures else 0)
