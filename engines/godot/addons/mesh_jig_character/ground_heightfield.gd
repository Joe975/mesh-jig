## The ground from a mesh-jig scenario's heightfield (glTF axes, which are Godot's): each grid cell split along its
## (x0, z0)-(x1, z1) diagonal, terrain patches applied from their time on. A port of game_scenario.Ground, used by
## the headless harness; a game can use it too for a heightfield it does not have as collision.
## query(x, z) -> {height, normal, walkable}; outside the grid the nearest edge height, not walkable.
extends RefCounted

var xs := PackedFloat64Array()
var zs := PackedFloat64Array()
var heights := []                    # a PackedFloat64Array per z row


func _init(terrain: Dictionary) -> void:
	for x in terrain["x"]:
		xs.append(float(x))
	for z in terrain["z"]:
		zs.append(float(z))
	for row in terrain["heights"]:
		var r := PackedFloat64Array()
		for h in row:
			r.append(float(h))
		heights.append(r)


## Raise (or lower) the grid points inside the rectangle [[x0, z0], [x1, z1]] by offset.
func patch(rect: Array, offset: float) -> void:
	for j in zs.size():
		for i in xs.size():
			if zs[j] >= rect[0][1] and zs[j] <= rect[1][1] and xs[i] >= rect[0][0] and xs[i] <= rect[1][0]:
				heights[j][i] = heights[j][i] + offset


static func _cell(axis: PackedFloat64Array, v: float) -> int:
	# searchsorted(axis, v, side='right') - 1, clipped to a cell
	var lo := 0
	var hi := axis.size()
	while lo < hi:
		var mid := (lo + hi) / 2
		if axis[mid] <= v:
			lo = mid + 1
		else:
			hi = mid
	return clampi(lo - 1, 0, axis.size() - 2)


## Height at (x, z) inside the grid (terrain.height's piecewise planar surface).
func height(x: float, z: float) -> float:
	var i := _cell(xs, x)
	var j := _cell(zs, z)
	var u := (x - xs[i]) / (xs[i + 1] - xs[i])
	var v := (z - zs[j]) / (zs[j + 1] - zs[j])
	var h00: float = heights[j][i]
	var h01: float = heights[j][i + 1]
	var h11: float = heights[j + 1][i + 1]
	var h10: float = heights[j + 1][i]
	if u >= v:
		return h00 * (1 - u) + h01 * (u - v) + h11 * v
	return h00 * (1 - v) + h11 * u + h10 * (v - u)


func query(x: float, z: float) -> Dictionary:
	var xc := clampf(x, xs[0], xs[-1])
	var zc := clampf(z, zs[0], zs[-1])
	var i := _cell(xs, xc)
	var j := _cell(zs, zc)
	var sx := xs[i + 1] - xs[i]
	var sz := zs[j + 1] - zs[j]
	var u := (xc - xs[i]) / sx
	var v := (zc - zs[j]) / sz
	var a: float = heights[j][i]
	var b: float = heights[j][i + 1]
	var c: float = heights[j + 1][i + 1]
	var d: float = heights[j + 1][i]
	var h := 0.0
	var du := 0.0
	var dv := 0.0
	if u >= v:
		h = a * (1 - u) + b * (u - v) + c * v
		du = b - a
		dv = c - b
	else:
		h = a * (1 - v) + c * u + d * (v - u)
		du = c - d
		dv = d - a
	var n := Vector3(-du / sx, 1.0, -dv / sz).normalized()
	return {"height": h, "normal": n, "walkable": xc == x and zc == z}


## The Callable a MeshJigCharacter takes as its ground query.
func as_callable() -> Callable:
	return query
