"""Real Blender integration tests with synthetic geometry; no archived experiments."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from mesh_jig import build, cli, evaluate, glb, project

pytestmark = pytest.mark.blender

PLATE = """\
import bpy

CREAM = (0.807, 0.761, 0.631, 1.0)
RED = (0.539, 0.044, 0.024, 1.0)


def build():
    # a flat delta, four triangles in two colours: far too little to be the mech
    verts = [(0.0, 0.15, 0.0), (-0.12, -0.15, 0.0), (0.12, -0.15, 0.0), (0.0, -0.15, 0.02)]
    faces = [(0, 1, 3), (0, 3, 2), (0, 2, 1), (1, 2, 3)]
    mesh = bpy.data.meshes.new("plate")
    mesh.from_pydata(verts, [], faces)
    attr = mesh.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="CORNER")
    for poly in mesh.polygons:
        for li in poly.loop_indices:
            attr.data[li].color = RED if poly.index == 3 else CREAM
    bpy.context.scene.collection.objects.link(bpy.data.objects.new("plate", mesh))
"""










def test_a_textured_build_binds_the_painted_atlas(blender, tmp_path):
    spec = {"script": "tile.py", "glb": "tile.glb", "textured": True, "single_mesh": True}
    script = tmp_path / "tile.py"
    script.write_text(
        "import bpy\n\n\ndef build():\n"
        "    mesh = bpy.data.meshes.new('tile')\n"
        "    mesh.from_pydata([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)], [], [(0, 1, 2, 3)])\n"
        "    uv = mesh.uv_layers.new()\n"
        "    for i, co in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):\n"
        "        uv.data[i].uv = co\n"
        "    bpy.context.scene.collection.objects.link(bpy.data.objects.new('tile', mesh))\n")
    from PIL import Image
    atlas = tmp_path / "paint.png"
    Image.new("RGB", (8, 8), (10, 200, 30)).save(atlas)
    assert build.build_variants(script, tmp_path / "out", spec, atlas=atlas) == ""
    stats = glb.stats(tmp_path / "out" / "tile.glb")
    assert stats["images"] == 1 and stats["textured_primitives"] == 1 and stats["primitives_without_texcoord0"] == 0
    no_uv = tmp_path / "flat.py"
    no_uv.write_text(script.read_text().replace("    uv = mesh.uv_layers.new()\n", "").replace(
        "    for i, co in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):\n        uv.data[i].uv = co\n", ""))
    err = build.build_variants(no_uv, tmp_path / "out2", dict(spec, script="flat.py"), atlas=atlas)
    assert "without a UV layer" in err


def test_a_script_may_shape_a_field_with_numpy_and_mesh_it_with_openvdb(blender, tmp_path):
    """Both modules are in Blender's own Python. A ball of radius 0.6 m as a distance field on a 0.1 m grid: the
    points openvdb returns are in metres once the grid has its voxel size, measured from the array's first cell, and
    the script's own assert (inside Blender) holds them to the ball once that corner is taken off."""
    from test_script_check import FIELD
    script = tmp_path / "ball.py"
    script.write_text(FIELD)
    err = build.build_variants(script, tmp_path / "out", {"script": "ball.py", "glb": "ball.glb"})
    if "ModuleNotFoundError" in err:
        pytest.skip("this Blender's Python has no openvdb module")
    assert err == ""
    stats = glb.stats(tmp_path / "out" / "ball.glb")
    assert stats["triangles"] > 200
    assert stats["size"] == pytest.approx([1.2, 1.2, 1.2], abs=0.11), "0.6 m radius, to within a voxel"
    assert stats["centre"] == pytest.approx([0, 0, 0], abs=0.06)


PAINTED = """\
import bpy, bmesh
from mathutils import Vector

SRGB = (0xd0 / 255, 0x78 / 255, 0x29 / 255, 1.0)
LINEAR = tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in SRGB[:3]) + (1.0,)


def box(x, how, value):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=0.2)
    for v in bm.verts:
        v.co = v.co + Vector((x, 0.0, 0.0))
    layer = {"bmesh byte": bm.loops.layers.color, "bmesh float": bm.loops.layers.float_color}.get(how)
    if layer is not None:
        lay = layer.new("Color")
        for f in bm.faces:
            for loop in f.loops:
                loop[lay] = value
    mesh = bpy.data.meshes.new(how)
    bm.to_mesh(mesh)
    bm.free()
    if layer is None:
        attr = mesh.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="CORNER")
        for d in attr.data:
            d.color = value
    bpy.context.scene.collection.objects.link(bpy.data.objects.new(how, mesh))


def build():
    box(0.0, "mesh attribute", LINEAR)
    box(1.0, "bmesh float", LINEAR)
    box(2.0, "bmesh byte", LINEAR)
    box(3.0, "bmesh byte", SRGB)
"""


def test_which_values_a_colour_takes_depends_on_what_it_is_painted_through(blender, tmp_path):
    """What agent.colour_facts tells a builder: linear values through the mesh's colour attribute or a bmesh float
    layer, sRGB through a bmesh byte layer, where linear values come out too dark."""
    import numpy as np
    from mesh_jig import agent, render
    script = tmp_path / "boxes.py"
    script.write_text(PAINTED)
    assert build.build_variants(script, tmp_path / "out", {"script": "boxes.py", "glb": "boxes.glb"}) == ""
    mesh = render.load(tmp_path / "out" / "boxes.glb")
    at = np.rint(mesh.tris[:, :, 0].mean(axis=1)).astype(int)             # which box a triangle is in, by its x
    shown = {x: render.linear_to_srgb(mesh.colors[at == x].reshape(-1, 3).mean(axis=0)) for x in range(4)}
    orange = np.array([0xd0, 0x78, 0x29]) / 255
    for x in (0, 1, 3):
        assert np.abs(shown[x] - orange).max() < 0.01, (x, shown[x])
    assert shown[2][0] < orange[0] - 0.1 and shown[2][1] < orange[1] - 0.2, "linear through the byte layer: too dark"
    told = agent.colour_facts(project.Project(tmp_path, {"model": {}, "palette": {"orange": "#d07829"}}))
    assert "bm.loops.layers.color is a byte layer that holds sRGB values" in told
    assert "use the mesh attribute or bm.loops.layers.float_color" in told


BLOCK = """\
import bpy
import numpy as np


def build():
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    bpy.context.active_object.name = "block"
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.2, location=(0.0, 0.0, 0.8))
    bpy.context.active_object.name = "knob"


def texture(name, position, normal):
    if name == "knob":
        return PALETTE["teal"]
    colour = np.tile(np.array(PALETTE["cream"]), (len(position), 1))
    colour[position[:, 0] < 0] = PALETTE["red"]          # the half on the model's left
    return colour
"""
PAINTS = {"red": "#c32b28", "cream": "#f3be99", "teal": "#2c4b50"}


def test_a_script_paints_its_texture_by_where_a_point_is_on_the_model(blender, tmp_path):
    """`texture_from: "script"`: no UVs and no image in the script. The runner unwraps both meshes into one atlas,
    asks `texture` about each texel's point, and with `blend` saves the Blender file beside the GLB."""
    import numpy as np
    from PIL import Image
    from mesh_jig import render
    front = {"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0], "ortho": 2.0}
    spec = {"script": "block.py", "glb": "block.glb", "textured": True, "texture_from": "script", "texture_size": 256,
            "blend": True, "light": "camera", "shots": [front]}
    script = tmp_path / "block.py"
    script.write_text(BLOCK)
    out = tmp_path / "out"
    assert build.build_variants(script, out, spec, palette=PAINTS) == ""
    stats = glb.stats(out / "block.glb")
    assert stats["images"] == 1 and stats["textured_primitives"] == stats["primitives"] == 2
    assert stats["primitives_without_texcoord0"] == 0, "the runner made the UVs"
    assert Image.open(out / "atlas.png").size == (256, 256)
    assert (out / "block.blend").stat().st_size > 20_000, "the Blender file, its atlas packed"
    img = np.asarray(render.render(render.load(out / "block.glb", textured=True), front, spec=spec), dtype=int)
    want = {name: np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) for name, h in PAINTS.items()}
    # the front camera stands on the model's +Y side, so the model's left (x < 0) is on the picture's right
    assert np.abs(img[200, 320 + 45] - want["red"]).max() <= 3, img[200, 320 + 45]
    assert np.abs(img[200, 320 - 45] - want["cream"]).max() <= 3, img[200, 320 - 45]
    assert np.abs(img[180 - 144, 320] - want["teal"]).max() <= 6, "the knob, 0.8 m up, looks at the camera at its middle"
    # a script that defines no texture function is told what the contract needs
    bare = tmp_path / "bare.py"
    bare.write_text(BLOCK.split("def texture")[0])
    err = build.build_variants(bare, tmp_path / "out2", dict(spec, script="bare.py"), palette=PAINTS)
    assert "define `def texture(name, position, normal):`" in err
    wrong = tmp_path / "wrong.py"
    wrong.write_text(BLOCK.split("def texture")[0] + "def texture(name, position, normal):\n    return position[:, :2]\n")
    err = build.build_variants(wrong, tmp_path / "out3", dict(spec, script="wrong.py"), palette=PAINTS)
    assert "texture('block', ...) was given" in err and "one (r, g, b) row per position" in err


def test_a_vertex_colour_build_with_blend_saves_a_file_whose_meshes_show_their_colour(blender, tmp_path):
    spec = {"script": "plate.py", "glb": "plate.glb", "vertex_colors": True, "blend": True}
    script = tmp_path / "plate.py"
    script.write_text(PLATE)
    assert build.build_variants(script, tmp_path / "out", spec) == ""
    stats = glb.stats(tmp_path / "out" / "plate.glb")
    assert stats["materials"] == 1 and stats["primitives_without_color0"] == 0 and stats["images"] == 0
    assert (tmp_path / "out" / "plate.blend").stat().st_size > 10_000
    plain = tmp_path / "plain"
    assert build.build_variants(script, plain, {k: v for k, v in spec.items() if k != "blend"}) == ""
    assert glb.stats(plain / "plate.glb")["materials"] == 0 and not (plain / "plate.blend").exists()
