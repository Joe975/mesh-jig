## The character's skeleton and clips, read from the imported GLB scene, character.runtime.json and the package's
## character.clips.json (every joint track of the GLB, key for key): the part of the reference's game_measure.Body the
## runtime uses. Clips are sampled from character.clips.json as the reference samples the GLB (linear, slerp), not from
## the Animation resources Godot's scene import makes, which resample and drop keys (see game_godot.py). Joints are indexed in the runtime file's skeleton.joints order.
## A pose is [translations, rotations, scales], one entry per joint, in each joint's parent frame (glTF axes).
## Model space is the GLB scene root's frame; a joint's model matrix = skeleton-in-model x its skeleton global pose.
extends RefCounted

var names: PackedStringArray = []        # runtime joint names, in skeleton.joints order
var index := {}                         # name -> joint index
var parent: PackedInt32Array = []       # joint index -> parent joint index, -1 for a root
var children := []                      # joint index -> Array of child joint indices
var bone: PackedInt32Array = []         # joint index -> Skeleton3D bone index
var rest_pose := []                     # [PackedVector3Array, Array[Quaternion], PackedVector3Array]
var rest_model: Array[Transform3D] = [] # rest model-space matrices
var skeleton_in_model := Transform3D.IDENTITY
var clips := {}                         # clip name -> per joint [translation, rotation, scale] channel or null
var from_player := false                # no clips file: sampled from the AnimationPlayer (Godot's imported clips)
var anim_ground := 0.0                  # the rest pose's lowest vertex: model-space height (glTF y)
var warnings: PackedStringArray = []


func setup(spec: Dictionary, model_root: Node3D, skeleton: Skeleton3D, player: AnimationPlayer, clip_data: Dictionary = {}) -> void:
	var joints: Array = spec["skeleton"]["joints"]
	for k in joints.size():
		names.append(joints[k]["name"])
		index[joints[k]["name"]] = k
	parent.resize(joints.size())
	children.resize(joints.size())
	var t := PackedVector3Array()
	var r: Array[Quaternion] = []
	var s := PackedVector3Array()
	for k in joints.size():
		var j: Dictionary = joints[k]
		parent[k] = -1 if j["parent"] == null else int(index[j["parent"]])
		children[k] = []
		var rt: Array = j["rest_translation"]
		var rr: Array = j["rest_rotation"]
		var rs: Array = j["rest_scale"]
		t.append(Vector3(rt[0], rt[1], rt[2]))
		r.append(Quaternion(rr[0], rr[1], rr[2], rr[3]).normalized())
		s.append(Vector3(rs[0], rs[1], rs[2]))
	for k in joints.size():
		if parent[k] >= 0:
			children[parent[k]].append(k)
	rest_pose = [t, r, s]
	skeleton_in_model = _relative(model_root, skeleton)
	bone = map_bones(skeleton)
	rest_model = model(rest_pose)
	anim_ground = rest_ground(model_root, skeleton)
	if not clip_data.is_empty():
		read_clip_data(spec, clip_data)
	elif player != null:
		from_player = true
		warnings.append("no character.clips.json: clips are sampled from Godot's imported animations, which can differ from the GLB")
		read_clips(spec, skeleton, player)


static func _relative(ancestor: Node3D, node: Node3D) -> Transform3D:
	"""node's transform in ancestor's frame, composed up the parent chain (works outside the tree too)."""
	var m := Transform3D.IDENTITY
	var n: Node = node
	while n != null and n != ancestor:
		if n is Node3D:
			m = (n as Node3D).transform * m
		n = n.get_parent()
	return m


static func _base(bone_name: String) -> String:
	"""Godot's glTF import makes bone names unique ("head" beside a mesh node "head" becomes "head_2") and replaces
	':' and '/'. The base name undoes that, for matching."""
	var b := bone_name.replace(":", "_").replace("/", "_")
	var at := b.rfind("_")
	if at > 0 and b.substr(at + 1).is_valid_int():
		b = b.substr(0, at)
	return b


func map_bones(skeleton: Skeleton3D) -> PackedInt32Array:
	"""Each runtime joint's bone, matched parent first: among the bones under the matched parent bone, the one of the
	same name, else of the same base name, nearest the joint's rest translation."""
	var out := PackedInt32Array()
	out.resize(names.size())
	out.fill(-1)
	var taken := {}
	var order: Array = []
	for k in names.size():
		if parent[k] < 0:
			order.append(k)
	var at := 0
	while at < order.size():
		var k: int = order[at]
		at += 1
		order.append_array(children[k])
		var want_parent := -1 if parent[k] < 0 else out[parent[k]]
		var best := -1
		var best_score := INF
		for b in skeleton.get_bone_count():
			if taken.has(b) or skeleton.get_bone_parent(b) != want_parent:
				continue
			var bn := skeleton.get_bone_name(b)
			var score := 0.0 if bn == names[k] else (1.0 if _base(bn) == _base(names[k]) else INF)
			if score == INF:
				continue
			score += skeleton.get_bone_rest(b).origin.distance_to(rest_pose[0][k])
			if score < best_score:
				best_score = score
				best = b
		if best < 0:
			push_error("mesh_jig_character: no bone in the skeleton matches joint %s" % names[k])
			return PackedInt32Array()
		out[k] = best
		taken[best] = true
	return out


## Each clip's tracks from character.clips.json: per joint, [translation, rotation, scale], each null (the joint
## holds its rest value) or {interpolation, times (PackedFloat64Array), values (PackedFloat64Array), stride}.
func read_clip_data(spec: Dictionary, data: Dictionary) -> void:
	for clip in spec["clips"]:
		if not data["clips"].has(clip):
			push_error("mesh_jig_character: character.clips.json has no clip %s" % clip)
			continue
		var per := []
		for k in names.size():
			var row := [null, null, null]
			var src: Dictionary = data["clips"][clip].get(names[k], {})
			for slot in 3:
				var kind: String = ["translation", "rotation", "scale"][slot]
				if src.has(kind):
					var times := PackedFloat64Array(src[kind]["times"])
					row[slot] = {"interpolation": src[kind]["interpolation"], "times": times,
							"values": PackedFloat64Array(src[kind]["values"]), "stride": 4 if slot == 1 else 3}
			per.append(row)
		clips[clip] = per


## A channel at time t, as glTF samples it (game_measure.Clip.local): the key at or before t, and with LINEAR the
## next key by the share of the way to it, slerp for rotations (skin.slerp), lerp otherwise. Doubles throughout.
static func sample(ch: Dictionary, t: float) -> PackedFloat64Array:
	var times: PackedFloat64Array = ch["times"]
	var values: PackedFloat64Array = ch["values"]
	var n := ch["stride"] as int
	var lo := 0
	var hi := times.size()
	while lo < hi:                         # searchsorted(times, t, side='right')
		var mid := (lo + hi) / 2
		if times[mid] <= t:
			lo = mid + 1
		else:
			hi = mid
	var k := clampi(lo - 1, 0, times.size() - 1)
	var a := values.slice(k * n, k * n + n)
	if ch["interpolation"] != "LINEAR" or k >= times.size() - 1:
		return a
	var u := clampf((t - times[k]) / (times[k + 1] - times[k]), 0.0, 1.0)
	var b := values.slice((k + 1) * n, (k + 1) * n + n)
	var out := PackedFloat64Array()
	out.resize(n)
	if n == 3:
		for i in 3:
			out[i] = a[i] * (1.0 - u) + b[i] * u
		return out
	var dot := a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
	if dot < 0.0:
		for i in 4:
			b[i] = -b[i]
		dot = -dot
	dot = minf(dot, 1.0)
	if dot > 0.9995:
		for i in 4:
			out[i] = a[i] + u * (b[i] - a[i])
	else:
		var angle := acos(dot)
		var sa := sin((1.0 - u) * angle) / sin(angle)
		var sb := sin(u * angle) / sin(angle)
		for i in 4:
			out[i] = sa * a[i] + sb * b[i]
	var length := sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3])
	for i in 4:
		out[i] /= length
	return out


func read_clips(spec: Dictionary, skeleton: Skeleton3D, player: AnimationPlayer) -> void:
	"""For each clip of the runtime file, its Animation and, per joint, its position, rotation and scale tracks
	(-1: none, the joint holds its rest value, as the GLB has it)."""
	var by_bone := {}
	for k in names.size():
		by_bone[skeleton.get_bone_name(bone[k])] = k
	for clip in spec["clips"]:
		var anim: Animation = null
		for lib in player.get_animation_library_list():
			var key := String(clip) if String(lib) == "" else "%s/%s" % [lib, clip]
			if player.has_animation(key):
				anim = player.get_animation(key)
				break
		if anim == null:
			push_error("mesh_jig_character: the model has no animation named %s" % clip)
			continue
		var tracks := []
		for k in names.size():
			tracks.append([-1, -1, -1])
		for tr in anim.get_track_count():
			var sub := anim.track_get_path(tr).get_concatenated_subnames()
			if not by_bone.has(sub):
				continue
			var slot: int = {Animation.TYPE_POSITION_3D: 0, Animation.TYPE_ROTATION_3D: 1, Animation.TYPE_SCALE_3D: 2}.get(anim.track_get_type(tr), -1)
			if slot >= 0:
				tracks[by_bone[sub]][slot] = tr
		clips[clip] = {"animation": anim, "tracks": tracks}


## Every joint's local transform in the clip at time t (no wrapping); a joint with no track holds its rest value.
func full(clip: String, t: float) -> Array:
	if not from_player:
		var per: Array = clips[clip]
		var t3 := PackedVector3Array(rest_pose[0])
		var r4: Array[Quaternion] = rest_pose[1].duplicate()
		var s3 := PackedVector3Array(rest_pose[2])
		for k in names.size():
			var row: Array = per[k]
			if row[0] != null:
				var v := sample(row[0], t)
				t3[k] = Vector3(v[0], v[1], v[2])
			if row[1] != null:
				var q := sample(row[1], t)
				r4[k] = Quaternion(q[0], q[1], q[2], q[3])
			if row[2] != null:
				var v := sample(row[2], t)
				s3[k] = Vector3(v[0], v[1], v[2])
		return [t3, r4, s3]
	var c: Dictionary = clips[clip]
	var anim: Animation = c["animation"]
	var tracks: Array = c["tracks"]
	var tt := PackedVector3Array(rest_pose[0])
	var rr: Array[Quaternion] = rest_pose[1].duplicate()
	var ss := PackedVector3Array(rest_pose[2])
	for k in names.size():
		var ids: Array = tracks[k]
		if ids[0] >= 0:
			tt[k] = anim.position_track_interpolate(ids[0], t)
		if ids[1] >= 0:
			rr[k] = anim.rotation_track_interpolate(ids[1], t)
		if ids[2] >= 0:
			ss[k] = anim.scale_track_interpolate(ids[2], t)
	return [tt, rr, ss]


func rest() -> Array:
	return [PackedVector3Array(rest_pose[0]), rest_pose[1].duplicate(), PackedVector3Array(rest_pose[2])]


func model(pose: Array) -> Array[Transform3D]:
	"""Forward kinematics: each joint's model-space matrix (parents come before children in the runtime file? not
	assumed: each joint is composed from its parent's matrix, computed on demand)."""
	var out: Array[Transform3D] = []
	out.resize(names.size())
	var done := PackedByteArray()
	done.resize(names.size())
	for k in names.size():
		_model(pose, k, out, done)
	return out


func _model(pose: Array, k: int, out: Array[Transform3D], done: PackedByteArray) -> void:
	if done[k]:
		return
	var local := Transform3D(Basis(pose[1][k]) * Basis.from_scale(pose[2][k]), pose[0][k])
	if parent[k] < 0:
		out[k] = skeleton_in_model * local
	else:
		_model(pose, parent[k], out, done)
		out[k] = out[parent[k]] * local
	done[k] = 1


func rest_ground(model_root: Node3D, skeleton: Skeleton3D) -> float:
	"""The rest pose's lowest vertex height in model space, as the reference reads it from the GLB: skinned meshes
	at the skeleton's rest (bone rest x bind pose, by weight), other meshes by their node transform."""
	var low := INF
	var rest_global := {}
	for b in skeleton.get_bone_count():
		rest_global[b] = skeleton.get_bone_global_rest(b)
	for node in model_root.find_children("*", "", true, false):
		var mesh = null
		if node is MeshInstance3D:
			mesh = node.mesh
		elif node is ImporterMeshInstance3D:
			mesh = node.mesh
		if mesh == null:
			continue
		var skin: Skin = node.skin
		var binds := []
		if skin != null:
			for i in skin.get_bind_count():
				var b := skin.get_bind_bone(i)
				if b < 0:
					b = skeleton.find_bone(skin.get_bind_name(i))
				binds.append(skeleton_in_model * rest_global.get(b, Transform3D.IDENTITY) * skin.get_bind_pose(i))
		var place := _relative(model_root, node)
		var count: int = mesh.get_surface_count()
		for si in count:
			var arrays: Array = mesh.get_surface_arrays(si) if mesh is ImporterMesh else mesh.surface_get_arrays(si)
			var verts: PackedVector3Array = arrays[Mesh.ARRAY_VERTEX]
			if skin == null or arrays[Mesh.ARRAY_BONES] == null:
				for v in verts:
					low = minf(low, (place * v).y)
				continue
			var bones: PackedInt32Array = arrays[Mesh.ARRAY_BONES]
			var weights: PackedFloat32Array = arrays[Mesh.ARRAY_WEIGHTS]
			var stride := bones.size() / maxi(verts.size(), 1)
			for vi in verts.size():
				var y := 0.0
				for q in stride:
					var w := weights[vi * stride + q]
					if w != 0.0:
						y += w * (binds[bones[vi * stride + q]] * verts[vi]).y
				low = minf(low, y)
	if low == INF:
		warnings.append("no mesh found: the rest ground is taken as 0")
		return 0.0
	return low
