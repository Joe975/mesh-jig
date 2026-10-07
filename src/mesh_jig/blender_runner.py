"""The Blender runner: the only thing mesh-jig asks Blender to run. Never edited by the agent.

  blender --background --factory-startup --python blender_runner.py -- --script <model.py> --out <file.glb>
          [--variant <name>] [--join 1] [--atlas <atlas.png>]
          [--texture-out <atlas.png> --texture-size <n> --palette <palette.json>] [--blend <file.blend>]

Wipes the default scene, execs the model script (which may only define build() and use bpy / bmesh / mathutils /
math / random / numpy / openvdb: script_check refuses anything else before exec), calls build(), then exports every object as one GLB:
Y-up, modifiers applied, the active colour attribute exported as COLOR_0. `--variant` sets the global VARIANT the
script reads (one script, several fits); `--join 1` joins every mesh object into one named after the output file.
Prints JIG_BUILD_OK on success, JIG_BUILD_ERROR + a traceback otherwise (exit 1). Never renders.

`--atlas <png>` is the textured contract: the image is bound as the base colour of ONE material on every mesh and
embedded in the GLB. The script still may not open files; it authors the UV layer (`mesh.uv_layers.new()`), and a
mesh with no UV layer is a build error, because an atlas on a mesh without UVs renders as one texel. The atlas is a
file painted from a JSON plan (mesh_jig.paint), never something the script reads.

`--texture-out <png>` is the textured contract with `texture_from: "script"`: the script defines `texture(name,
position, normal)` beside `build()`, the runner unwraps every mesh into one atlas (the script authors no UVs),
asks `texture` for the colour of each texel's point on the model (texel_bake), writes the image there and binds it
as `--atlas` would. `--palette <json>` (name -> "#rrggbb") reaches the script as the global PALETTE, name ->
(r, g, b) sRGB 0..1, which is what `texture` returns.

`--blend <file>` also saves the scene as a Blender file, images packed: what is handed on to whoever textures,
rigs or animates the model. On a vertex-colour build each mesh that has no material is given one that reads its
colour attribute, so the file and the GLB show their colours in a viewport and an engine.

This file runs under Blender's own Python, so it imports nothing from the mesh_jig package except script_check and
texel_bake, which sit beside it and need only the standard library and the numpy that Blender ships.
"""
import os
import sys
import traceback

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from script_check import check_script  # noqa: E402

OK_MARK, ERR_MARK = "JIG_BUILD_OK", "JIG_BUILD_ERROR"


def _args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    opts = {}
    for i in range(0, len(argv) - 1, 2):
        opts[argv[i].lstrip("-")] = argv[i + 1]
    return opts


def reset_scene():
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o, do_unlink=True)
    for coll in (bpy.data.meshes, bpy.data.materials, bpy.data.curves, bpy.data.lights, bpy.data.cameras):
        for x in list(coll):
            coll.remove(x)


def join_meshes(name):
    """Apply modifiers and join every mesh object into one object called `name`."""
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.convert(target="MESH")   # bake modifiers first: join keeps only the active object's stack
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    obj.name = name
    obj.data.name = name
    return obj


def bind_atlas(meshes, atlas_path, name="atlas"):
    """One material reading `atlas_path` as base colour, on every mesh. Every mesh must already carry a UV layer
    (the script's job); refusing a mesh without one is what keeps a textured build from silently rendering flat."""
    if not os.path.isfile(atlas_path):
        raise RuntimeError(f"atlas not found: {atlas_path}")
    missing = [o.name for o in meshes if not o.data.uv_layers]
    if missing:
        raise RuntimeError(f"textured build: mesh(es) without a UV layer {missing[:6]}; the script must unwrap them "
                           "(mesh.uv_layers.new() and set every loop's uv)")
    img = bpy.data.images.load(atlas_path)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    nt.links.new(bsdf.inputs["Base Color"], tex.outputs["Color"])
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Metallic"].default_value = 0.0
    for o in meshes:
        o.data.materials.clear()
        o.data.materials.append(mat)
    return mat


def apply_modifiers(meshes):
    """Every mesh as its modifiers leave it: an atlas is laid out and painted on the surface that is exported."""
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.convert(target="MESH")


def unwrap_together(meshes, margin):
    """One UV layout over every mesh, islands packed into the one atlas, each sized by its area on the model."""
    for o in meshes:
        layer = o.data.uv_layers.get("atlas") or o.data.uv_layers.new(name="atlas")
        if layer is None:
            raise RuntimeError(f"mesh {o.name} cannot take another UV layer (it has {len(o.data.uv_layers)})")
        o.data.uv_layers.active = layer
        layer.active_render = True
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=margin, area_weight=0.0, correct_aspect=True,
                             scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    for o in meshes:                         # the layout is the only one the export and an engine should see
        for name in [layer.name for layer in o.data.uv_layers if layer.name != "atlas"]:
            o.data.uv_layers.remove(o.data.uv_layers[name])


def paint_atlas(meshes, paint_fn, size, path):
    """The script-painted atlas: unwrap, find each texel's point on the model, ask the script, write the PNG.
    -> how many texels the model covers."""
    import numpy as np
    import texel_bake
    unwrap_together(meshes, margin=3.0 / size)
    uvs, points, normals, part = [], [], [], []
    for k, o in enumerate(meshes):
        m = o.data
        m.calc_loop_triangles()
        n = len(m.loop_triangles)
        loops = np.zeros(n * 3, dtype=np.int32)
        m.loop_triangles.foreach_get("loops", loops)
        corner = np.zeros(n * 9)
        m.loop_triangles.foreach_get("split_normals", corner)
        uv = np.zeros(len(m.loops) * 2)
        m.uv_layers["atlas"].data.foreach_get("uv", uv)
        vertex = np.zeros(len(m.loops), dtype=np.int32)
        m.loops.foreach_get("vertex_index", vertex)
        co = np.zeros(len(m.vertices) * 3)
        m.vertices.foreach_get("co", co)
        world = np.array(o.matrix_world)
        co = co.reshape(-1, 3) @ world[:3, :3].T + world[:3, 3]
        turn = np.linalg.inv(world[:3, :3]).T
        uvs.append(uv.reshape(-1, 2)[loops].reshape(n, 3, 2))
        points.append(co[vertex[loops]].reshape(n, 3, 3))
        normals.append((corner.reshape(-1, 3) @ turn.T).reshape(n, 3, 3))
        part.append(np.full(n, k))
    covered, point, normal, tri = texel_bake.rasterise(np.concatenate(uvs), np.concatenate(points),
                                                       np.concatenate(normals), size)
    owner = np.where(tri >= 0, np.concatenate(part)[np.maximum(tri, 0)], -1)
    colour = texel_bake.paint(paint_fn, [o.name for o in meshes], owner, covered, point, normal)
    colour, reached = texel_bake.dilate(colour, covered, steps=max(4, size // 128))
    img = bpy.data.images.new("atlas_painted", size, size, alpha=False)
    img.pixels.foreach_set(texel_bake.pixels(colour, reached))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    return int(covered.sum())


def show_vertex_colour(meshes):
    """A mesh with a colour attribute and no material gets one that reads it."""
    mat = None
    for o in meshes:
        attrs = o.data.color_attributes
        if len(o.data.materials) or not len(attrs):
            continue
        if mat is None:
            mat = bpy.data.materials.new("vertex_colour")
            mat.use_nodes = True
            nt = mat.node_tree
            bsdf = nt.nodes["Principled BSDF"]
            node = nt.nodes.new("ShaderNodeVertexColor")
            node.layer_name = (attrs.active_color or attrs[0]).name
            nt.links.new(bsdf.inputs["Base Color"], node.outputs["Color"])
            bsdf.inputs["Roughness"].default_value = 0.8
            bsdf.inputs["Metallic"].default_value = 0.0
        o.data.materials.append(mat)


def save_blend(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    for img in bpy.data.images:
        if img.filepath and not img.packed_file:
            img.pack()
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True, check_existing=False)


def export_glb(path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    kw = dict(filepath=path, export_format="GLB",
              export_yup=True, export_apply=True, use_selection=False,
              export_materials="EXPORT", export_normals=True, export_texcoords=True,
              export_animations=False, export_lights=False, export_cameras=False, export_image_format="AUTO")
    try:   # Blender 4.2+: export the active colour attribute even when no material reads it
        bpy.ops.export_scene.gltf(export_vertex_color="ACTIVE", **kw)
    except TypeError:
        bpy.ops.export_scene.gltf(**kw)


def main():
    opts = _args()
    script, out, variant = opts["script"], opts["out"], opts.get("variant", "")
    join, atlas = opts.get("join", "0") == "1", opts.get("atlas", "")
    painted, blend = opts.get("texture-out", ""), opts.get("blend", "")
    try:
        bpy.ops.preferences.addon_enable(module="io_scene_gltf2")
    except Exception:
        pass
    reset_scene()
    with open(script, encoding="utf-8") as f:
        src = f.read()
    problems = check_script(src)
    if problems:
        raise RuntimeError("model script refused: " + "; ".join(problems[:6]))
    ns = {"__name__": "jig_model", "VARIANT": variant}
    if painted:
        import json
        with open(opts["palette"], encoding="utf-8") as f:
            ns["PALETTE"] = {name: tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
                             for name, h in json.load(f).items()}
    exec(compile(src, os.path.basename(script), "exec"), ns)
    if "build" not in ns:
        raise RuntimeError("model script defines no build()")
    if painted and not callable(ns.get("texture")):
        raise RuntimeError("this contract's texture is painted by the script: define `def texture(name, position, "
                           "normal):` at top level, returning one sRGB (r, g, b) row per position")
    ns["build"]()
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("build() created no mesh objects")
    if join:
        meshes = [join_meshes(os.path.splitext(os.path.basename(out))[0])]
    texels = 0
    if painted:
        if not join:
            apply_modifiers(meshes)
            meshes = [o for o in bpy.data.objects if o.type == "MESH"]
        texels = paint_atlas(meshes, ns["texture"], int(opts.get("texture-size", "1024")), painted)
        atlas = painted
    if atlas:
        bind_atlas(meshes, atlas)
    elif blend:
        show_vertex_colour(meshes)
    export_glb(out)
    if blend:
        save_blend(blend)
    print(f"{OK_MARK} objects={len(meshes)} out={out}" + (f" atlas={os.path.basename(atlas)}" if atlas else "")
          + (f" texels={texels}" if painted else "") + (f" blend={os.path.basename(blend)}" if blend else ""))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001 -- the error text is the agent's next input
        traceback.print_exc()
        print(f"{ERR_MARK}: {type(e).__name__}: {e}")
        sys.stdout.flush()
        sys.exit(1)
