## Runs the character's runtime each skeleton update, after any animation, and writes the result to the skeleton as
## local bone poses. The runtime itself (mesh_jig_runtime.gd, with mesh_jig_graph.gd) is a RefCounted so it can keep
## the reference's function names: Node3D, which SkeletonModifier3D extends, already has rotate() and translate().
## In Godot's default mode the skeleton passes the process delta; MeshJigCharacter.advance() passes a chosen one.
extends SkeletonModifier3D

const Runtime = preload("mesh_jig_runtime.gd")

var runtime: Runtime
var model_root: Node3D                # the GLB scene's root: its global transform is the root matrix
var ground_query: Callable            # (x, z) -> {height, normal, walkable}
var step_usec := 0                    # how long the last update took (runtime step and bone writes), microseconds


func _process_modification_with_delta(delta: float) -> void:
	var skeleton := get_skeleton()
	if skeleton == null or runtime == null or runtime.body.bone.is_empty():
		return
	var started := Time.get_ticks_usec()
	write(skeleton, runtime.step(delta, model_root.global_transform, ground_query))
	step_usec = Time.get_ticks_usec() - started


## The world matrices written to the skeleton as local bone poses (each joint's pose in its parent's frame).
func write(skeleton: Skeleton3D, world: Array[Transform3D]) -> void:
	var to_skeleton := skeleton.global_transform.affine_inverse()
	var global: Array[Transform3D] = []
	for i in world.size():
		global.append(to_skeleton * world[i])
	for i in world.size():
		var p := runtime.body.parent[i]
		var local := global[i] if p < 0 else global[p].affine_inverse() * global[i]
		var b := runtime.body.bone[i]
		skeleton.set_bone_pose_position(b, local.origin)
		skeleton.set_bone_pose_rotation(b, local.basis.get_rotation_quaternion())
		skeleton.set_bone_pose_scale(b, local.basis.get_scale())
