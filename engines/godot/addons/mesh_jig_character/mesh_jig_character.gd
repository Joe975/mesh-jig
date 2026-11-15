## A mesh-jig game-ready character: the GLB scene as a child, driven by character.runtime.json. It blends the clips by
## the package's graph, keeps the feet on the ground, solves limb targets and aims, every frame, after any animation.
## It never moves itself: move its parent (a CharacterBody3D, say) and it reads that each frame. In the generated
## character.tscn this node's origin is the rest pose's lowest point (the Model child is raised by that much), so put
## it where the character stands: the bottom of the body's collision shape.
##
## The calls, as every mesh-jig runtime names them: set_parameter(name, value), trigger(name), play(name),
## set_aim(name, point or null), set_limb_target(name, point or null, weight), set_ground_enabled(enabled).
## Points are world positions. The ground is a Callable (x, z) -> {height, normal, walkable}: a physics raycast
## (ground_raycast.gd, made as a child named Ground) unless the game sets ground_query.
## No class_name: every package carries a copy of this addon, and two scripts with one global class name clash.
extends Node3D

const Body = preload("mesh_jig_body.gd")
const Graph = preload("mesh_jig_graph.gd")
const Runtime = preload("mesh_jig_runtime.gd")
const Modifier = preload("mesh_jig_modifier.gd")
const RaycastGround = preload("ground_raycast.gd")

const KNOWN := {
	"": ["format", "version", "glb", "axes", "skeleton", "parameters", "triggers", "oneshots", "clips", "graph", "body",
		"limbs", "ground", "aims", "sockets", "blending", "measurement", "capabilities", "hashes"],
	"limb": ["name", "role", "solver", "joints", "segment_lengths_m", "pole", "bend_degrees", "bend_degrees_max", "contact",
		"clearance"],
	"contact": ["joint", "vertices", "points"],
	"ground": ["drop_m_max", "rise_m_max", "tilt", "relock_m", "blend_s", "lookahead_s"],
}

@export var runtime_file: JSON                     ## the package's character.runtime.json
@export var clips_file: JSON                       ## the package's godot/character.clips.json: the GLB's tracks, exact
@export_flags_3d_physics var ground_collision_mask := 1   ## what the default raycast ground hits
@export var ground_ray_up := 1.0                   ## metres above the character's origin each ground ray starts
@export var ground_ray_length := 3.0               ## metres each ground ray reaches down
@export var manual_clock := false                  ## true: nothing moves until advance(dt) is called (tests, replays)

var ground_query: Callable                         ## (x, z) -> {height, normal, walkable}; set before or after ready
var spec: Dictionary
var clip_data: Dictionary                          ## clips_file's data; a harness may set it instead
var model_root: Node3D
var skeleton: Skeleton3D
var body: Body
var graph: Graph
var runtime: Runtime
var modifier: Modifier
var warnings: PackedStringArray = []
var _ground_owner: Object                          # keeps the ground query's object alive


func _ready() -> void:
	if Engine.is_editor_hint():
		return
	if spec.is_empty() and runtime_file != null:
		spec = runtime_file.data
	if clip_data.is_empty() and clips_file != null:
		clip_data = clips_file.data
	if spec.is_empty():
		push_error("MeshJigCharacter %s: no runtime_file" % name)
		return
	setup(spec)


## Build everything from a parsed runtime file. _ready calls it with runtime_file's data; a harness may call it with
## a dictionary it read itself, once the model is a child and this node is in the tree.
func setup(p_spec: Dictionary) -> void:
	spec = p_spec
	_warn_unknown()
	for n in get_children():
		if n is Node3D and not n.find_children("*", "Skeleton3D", true, false).is_empty():
			model_root = n
			break
	if model_root == null:
		push_error("MeshJigCharacter %s: no child scene with a Skeleton3D (instance character.glb under it)" % name)
		return
	skeleton = model_root.find_children("*", "Skeleton3D", true, false)[0]
	var players := model_root.find_children("*", "AnimationPlayer", true, false)
	var player: AnimationPlayer = players[0] if not players.is_empty() else null
	if player != null:
		player.stop()                   # this character's runtime poses every joint; the player only holds the clips
	body = Body.new()
	body.setup(spec, model_root, skeleton, player, clip_data)
	for w in body.warnings:
		_warning(w)
	graph = Graph.new(spec, body)
	runtime = Runtime.new(spec, body, graph)
	modifier = Modifier.new()
	modifier.name = "MeshJigModifier"
	modifier.runtime = runtime
	modifier.model_root = model_root
	if not ground_query.is_valid():
		var ray := RaycastGround.new()
		ray.name = "Ground"
		ray.collision_mask = ground_collision_mask
		ray.ray_up = ground_ray_up
		ray.ray_length = ground_ray_length
		ray.reference = model_root
		ray.rest_ground_y = body.anim_ground
		var up := get_parent()
		while up != null:                   # the game's own body (a CharacterBody3D above this node) is no ground
			if up is CollisionObject3D:
				ray.exclude.append(up.get_rid())
			up = up.get_parent()
		add_child(ray)
		ground_query = ray.as_callable()
	_ground_owner = ground_query.get_object()
	modifier.ground_query = ground_query
	skeleton.add_child(modifier)
	if manual_clock:
		skeleton.modifier_callback_mode_process = Skeleton3D.MODIFIER_CALLBACK_MODE_PROCESS_MANUAL


## A Callable to a method holds no reference to its object: this keeps the object (a ground_heightfield.gd, say) alive
## as long as the character uses it, so a game may pass `Heightfield.new(terrain).as_callable()` and keep nothing.
func set_ground_query(query: Callable) -> void:
	ground_query = query
	_ground_owner = query.get_object()
	if modifier != null:
		modifier.ground_query = query


## With manual_clock: run one frame of dt seconds at the skeleton's next update (skeleton_updated then carries the
## final pose). Godot's skeleton runs its modifiers deferred, so the pose is there after the current frame.
func advance(dt: float) -> void:
	skeleton.advance(dt)


# the interface -------------------------------------------------------------------------------------------------
func set_parameter(param: String, value: float) -> void:
	graph.set_parameter(param, value)


func trigger(trigger_name: String) -> void:
	graph.trigger(trigger_name)


func play(oneshot: String) -> void:
	graph.play(oneshot)


## point: a world Vector3, or null to let the aim return to the animation.
func set_aim(aim_name: String, point) -> void:
	runtime.set_aim(aim_name, point)


## point: a world Vector3, or null to release the limb. The weight is applied as given: ramp it for a smooth move.
func set_limb_target(limb: String, point, weight: float = 1.0) -> void:
	runtime.set_limb_target(limb, point, weight)


func set_ground_enabled(enabled: bool) -> void:
	runtime.set_ground_enabled(enabled)


## A socket from the runtime file as a node that follows its joint as the runtime poses it: a BoneAttachment3D on the
## joint (made once, named "Socket <name>") with a child at the socket's offset. Parent a weapon to the child.
func socket(socket_name: String) -> Node3D:
	var made := skeleton.get_node_or_null("Socket " + socket_name)
	if made != null:
		return made.get_child(0)
	for s in spec["sockets"]:
		if s["name"] == socket_name:
			var attach := BoneAttachment3D.new()
			attach.name = "Socket " + socket_name
			attach.bone_name = skeleton.get_bone_name(body.bone[body.index[s["joint"]]])
			var at := Node3D.new()
			at.name = socket_name
			var o: Array = s["offset_local"]
			at.position = Vector3(o[0], o[1], o[2])
			attach.add_child(at)
			skeleton.add_child(attach)
			return at
	push_error("MeshJigCharacter %s: no socket named %s" % [name, socket_name])
	return null


## What the graph, the feet and the aims did on the last frame: parameters, each layer's weight and clips at their
## times, states, planted legs, body offset, aim offsets. Print it, or show it on a label, to see why it moves so.
func debug_state() -> Dictionary:
	if runtime == null:
		return {}
	var out := graph.debug_state()
	out.merge(runtime.debug_state())
	out["step_usec"] = modifier.step_usec
	return out


func _warning(text: String) -> void:
	warnings.append(text)
	push_warning("MeshJigCharacter %s: %s" % [name, text])


func _warn_unknown() -> void:
	for key in spec:
		if not KNOWN[""].has(key):
			_warning("character.runtime.json key %s is not known to this addon version and is ignored" % key)
	for limb in spec.get("limbs", []):
		for key in limb:
			if not KNOWN["limb"].has(key):
				_warning("limb %s key %s is not known to this addon version and is ignored" % [limb["name"], key])
		if limb.has("contact"):
			for key in limb["contact"]:
				if not KNOWN["contact"].has(key):
					_warning("limb %s contact key %s is not known to this addon version and is ignored" % [limb["name"], key])
	if spec.get("ground") is Dictionary:
		for key in spec["ground"]:
			if not KNOWN["ground"].has(key):
				_warning("ground key %s is not known to this addon version and is ignored" % key)
