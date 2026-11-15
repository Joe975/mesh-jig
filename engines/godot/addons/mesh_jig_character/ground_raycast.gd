## The ground for a game: a physics ray straight down at (x, z), from ray_up above the character's origin to
## ray_length below that start, against collision_mask. A MeshJigCharacter makes one of these as its ground query
## unless the game sets another Callable, and excludes every collision body above the character in the tree (the
## game's CharacterBody3D). A miss returns the character's own rest ground (the animation plays as authored there)
## and walkable false.
@tool
extends Node3D

@export_flags_3d_physics var collision_mask := 1
@export var ray_up := 1.0           ## metres above the character's origin where each ray starts
@export var ray_length := 3.0       ## metres each ray reaches down from its start
@export var exclude: Array[RID] = []   ## bodies the rays ignore (the character's own body, say)

var reference: Node3D               # the character's model root: rays start ray_up above its origin
var rest_ground_y := 0.0            # the model's rest ground (model-space height): what a miss reports


func query(x: float, z: float) -> Dictionary:
	var origin := reference.global_transform if reference != null else global_transform
	var miss_height := (origin * Vector3(0, rest_ground_y, 0)).y
	var space := get_world_3d().direct_space_state if is_inside_tree() else null
	if space == null:
		return {"height": miss_height, "normal": Vector3.UP, "walkable": false}
	var start := Vector3(x, origin.origin.y + ray_up, z)
	var ray := PhysicsRayQueryParameters3D.create(start, start - Vector3(0, ray_length, 0), collision_mask, exclude)
	var hit := space.intersect_ray(ray)
	if hit.is_empty():
		return {"height": miss_height, "normal": Vector3.UP, "walkable": false}
	return {"height": hit["position"].y, "normal": hit["normal"], "walkable": true}


func as_callable() -> Callable:
	return query
