"""Safe worked construction recipe, not a reconstruction of the pirate-cat reference.

Blender metres: +Z up, +Y front, anatomical left X<0. Parameters are intentionally
small and fixed for the regression example. Copy helpers into a model script; imports
remain within mesh-jig's existing allowlist. Exporting is the runner's responsibility.
"""
import bpy
import bmesh
import math
from mathutils import Vector
from mathutils.bvhtree import BVHTree

VOXEL_M = .012
SMOOTH_ITERATIONS = 3
SMOOTH_FACTOR = .35
SKIN_TRIANGLES = 6000
COAT = (.55, .20, .06, 1)
CREAM = (.8, .72, .5, 1)
TEAL = (.02, .22, .18, 1)


def activate(obj):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def ellipsoid(name, centre, radii):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=16, location=centre)
    obj = bpy.context.object
    obj.name = name
    obj.scale = radii
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def mesh_object(name, vertices, faces):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def ear(side):
    # Broad curved root overlaps scalp; a tapered wedge gives a deliberate feline tip.
    x = side * .13
    vertices = [(x-.085, -.05, .94), (x+.085, -.05, .94), (x+side*.02, 0, 1.15),
                (x-.07, .06, .94), (x+.07, .06, .94), (x+side*.02, .045, 1.13)]
    return mesh_object("Ear", vertices, [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)])


def curve_frames(points):
    """Minimal-rotation transported frames; never switch helper axes at individual rings."""
    p = [Vector(v) for v in points]
    tangents = [(p[min(i+1, len(p)-1)] - p[max(i-1, 0)]).normalized() for i in range(len(p))]
    helper = Vector((1, 0, 0)) if abs(tangents[0].x) < .9 else Vector((0, 1, 0))
    u = tangents[0].cross(helper).normalized()
    frames = []
    previous = tangents[0]
    for tangent in tangents:
        if previous.dot(tangent) < -.999:
            raise ValueError("curve reverses direction; add intermediate points")
        u = previous.rotation_difference(tangent) @ u
        u = (u - tangent * u.dot(tangent)).normalized()
        frames.append((u.copy(), tangent.cross(u).normalized()))
        previous = tangent
    return frames


def tube(name, points, radii, segments=12):
    frames = curve_frames(points)
    vertices, faces = [], []
    for p, radius, (u, v) in zip(points, radii, frames):
        centre = Vector(p)
        vertices.extend(tuple(centre + radius*(math.cos(2*math.pi*j/segments)*u + math.sin(2*math.pi*j/segments)*v))
                        for j in range(segments))
    for i in range(len(points)-1):
        for j in range(segments):
            k = (j+1) % segments
            faces.append((i*segments+j, i*segments+k, (i+1)*segments+k, (i+1)*segments+j))
    faces.extend([tuple(range(segments-1, -1, -1)), tuple(range((len(points)-1)*segments, len(points)*segments))])
    return mesh_object(name, vertices, faces)


def paint(obj, color, cream_front=False):
    # Rebuild attributes AFTER topology-changing modifiers. Paint the finished skin,
    # so a belly marking cannot become an opaque insert covering a belt or buckle.
    for old in list(obj.data.color_attributes):
        obj.data.color_attributes.remove(old)
    attr = obj.data.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="CORNER")
    for poly in obj.data.polygons:
        poly.use_smooth = True
        for li in poly.loop_indices:
            co = obj.data.vertices[obj.data.loops[li].vertex_index].co
            cream = cream_front and co.y > .1 and (co.z > .75 or (.28 < co.z < .65 and abs(co.x) < .12))
            attr.data[li].color = CREAM if cream else color


def fuse_skin(objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    skin = bpy.context.object
    skin.name = "Skin"
    # Join retains overlapping islands; voxel remesh performs the actual fusion.
    skin.data.remesh_voxel_size = VOXEL_M
    skin.data.remesh_voxel_adaptivity = 0
    bpy.ops.object.voxel_remesh()
    smooth = skin.modifiers.new("Controlled smooth", "SMOOTH")
    smooth.factor, smooth.iterations = SMOOTH_FACTOR, SMOOTH_ITERATIONS
    bpy.ops.object.modifier_apply(modifier=smooth.name)
    skin.data.calc_loop_triangles()
    count = len(skin.data.loop_triangles)
    if count > SKIN_TRIANGLES:
        decimate = skin.modifiers.new("Budget", "DECIMATE")
        decimate.ratio = SKIN_TRIANGLES / count
        bpy.ops.object.modifier_apply(modifier=decimate.name)
    paint(skin, COAT, cream_front=True)
    return skin


def fitted_vest(skin):
    # Every ring vertex is fitted to the FINAL skin, including the open-front edges.
    # Build both inner and outer surfaces, not only an offset centreline.
    bvh = BVHTree.FromPolygons([v.co for v in skin.data.vertices], [list(p.vertices) for p in skin.data.polygons])
    heights, angles = [.28+j*.02 for j in range(20)], [math.radians(35 + j*5) for j in range(59)]
    vertices, faces = [], []
    for offset in (.012, .022):
        for z in heights:
            for a in angles:
                direction = Vector((math.sin(a), math.cos(a), 0))
                radius = 0
                # Reserve the neighbouring loft interval too: a skin lobe can appear
                # between rings even when both end-ring probes pass.
                for sample_z in (z-.02, z-.01, z, z+.01, z+.02):
                    origin = Vector((0, 0, sample_z))
                    hit, normal, face, distance = bvh.ray_cast(origin, direction, 1)
                    if hit is None:
                        continue
                    # Concave arm/torso skin may exit and re-enter; fit the LAST exit.
                    cursor = hit + direction*.00001
                    for probe in range(16):
                        next_hit, next_normal, next_face, next_distance = bvh.ray_cast(cursor, direction, 1)
                        if next_hit is None:
                            break
                        hit = next_hit
                        cursor = hit + direction*.00001
                    radius = max(radius, (hit-origin).dot(direction))
                if radius == 0:
                    raise ValueError("vest probe missed skin; move the ring inside the final torso")
                # Radial clearance preserves this ring's height and its ray hit.
                # Normal offsets shift the probe in Z and can pull shell chords inside
                # the concave shoulder/arm transition. Check inter-ring spans too.
                vertices.append(tuple(Vector((0, 0, z)) + direction*(radius+offset)))
    cols, rows = len(angles), len(heights)
    layer = cols*rows
    for side in (0, 1):
        for r in range(rows-1):
            for c in range(cols-1):
                i = side*layer+r*cols+c
                faces.append((i, i+1, i+cols+1, i+cols))
    for r in range(rows-1):
        for c in (0, cols-1):
            i = r*cols+c
            faces.append((i, i+cols, i+cols+layer, i+layer))
    for r in (0, rows-1):
        for c in range(cols-1):
            i = r*cols+c
            faces.append((i, i+layer, i+1+layer, i+1))
    vest = mesh_object("Vest", vertices, faces)
    paint(vest, TEAL)
    return vest


def build():
    masses = [ellipsoid("Torso", (0, 0, .48), (.23, .17, .3)),
              ellipsoid("Head", (0, .02, .85), (.23, .18, .23))]
    for side in (-1, 1):
        masses.extend([ear(side), ellipsoid("Cheek", (side*.09, .17, .79), (.095, .075, .065)),
                       ellipsoid("Arm", (side*.25, 0, .52), (.11, .10, .21)),
                       ellipsoid("Leg", (side*.11, 0, .2), (.11, .11, .18))])
        # Flattened soles and elongated toes replace round paw balls; all overlap leg/foot.
        masses.append(ellipsoid("Paw", (side*.11, .045, .07), (.12, .15, .065)))
        for digit in (-1, 0, 1):
            masses.append(ellipsoid("Toe", (side*.11+digit*.05, .15, .065), (.035, .065, .05)))
    skin = fuse_skin(masses)
    # Preserve floor position after remesh/smoothing without normalizing the height.
    floor = min(v.co.z for v in skin.data.vertices)
    for v in skin.data.vertices:
        v.co.z -= floor
    fitted_vest(skin)
    tail = tube("Tail", [(0, -.12, .3), (0, -.3, .3), (0, -.42, .4), (0, -.43, .58)], [.07, .06, .045, .025])
    paint(tail, COAT)
