"""The containment check a model script passes before Blender executes it."""
import pytest

from mesh_jig.script_check import check_script

GOOD = "import bpy\nimport math\n\n\ndef build():\n    bpy.ops.mesh.primitive_cone_add(vertices=8)\n"


def test_a_plain_bpy_script_is_clean():
    assert check_script(GOOD) == []


@pytest.mark.parametrize("source, fragment", [
    ("import os\ndef build():\n    pass\n", "import of 'os'"),
    ("from subprocess import run\ndef build():\n    pass\n", "import of 'subprocess'"),
    ("def build():\n    open('x')\n", "forbidden builtin 'open'"),
    ("def build():\n    eval('1')\n", "forbidden builtin 'eval'"),
    ("import bpy\ndef build():\n    bpy.ops.wm.save_mainfile()\n", "forbidden API bpy.ops.wm"),
    ("import bpy\ndef build():\n    bpy.ops.render.render()\n", "forbidden API bpy.ops.render"),
    ("import bpy\ndef build():\n    bpy.app.version\n", "forbidden API bpy.app"),
    ("def build():\n    ().__class__\n", "dunder attribute"),
    ("def build(:\n", "syntax error"),
    ("import bpy\nif True:\n    def build():\n        pass\n", "no top-level `def build():`"),
    ("import bpy\n", "no top-level `def build():`"),
])
def test_each_escape_is_named_with_its_line(source, fragment):
    issues = check_script(source)
    assert any(fragment in i for i in issues), issues
    assert all(i.startswith("line ") for i in issues)


FIELD = """\
import bpy
import numpy as np
import openvdb as vdb
from numpy import linalg


def build():
    x, y, z = np.mgrid[-8:9, -8:9, -8:9].astype(np.float32)
    grid = vdb.FloatGrid(1.0)
    grid.copyFromArray(np.ascontiguousarray(np.sqrt(x * x + y * y + z * z) - 6.0))
    grid.transform = vdb.createLinearTransform(voxelSize=0.1)
    points, quads = grid.convertToQuads(isovalue=0.0)
    points = points - 0.8        # openvdb measures from the array's first cell, which is the field's -0.8 m corner
    mesh = bpy.data.meshes.new("ball")
    mesh.from_pydata(points.tolist(), [], quads.tolist())
    assert 0.5 < linalg.norm(points, axis=1).max() < 0.7
    bpy.context.scene.collection.objects.link(bpy.data.objects.new("ball", mesh))
"""


def test_numpy_and_openvdb_which_ship_in_blenders_python_may_be_imported():
    assert check_script(FIELD) == []


@pytest.mark.parametrize("source, fragment", [
    ("import numpy as np\ndef build():\n    np.save('x.npy', np.zeros(3))\n", "forbidden API numpy.save"),
    ("import numpy\ndef build():\n    numpy.fromfile('x')\n", "forbidden API numpy.fromfile"),
    ("from numpy import load\ndef build():\n    load('x.npy')\n", "forbidden API numpy.load"),
    ("from numpy import load as get\ndef build():\n    get('x.npy')\n", "forbidden API numpy.load"),
    ("import numpy.lib.npyio\ndef build():\n    pass\n", "forbidden API numpy.lib.npyio"),
    ("import numpy as np\ndef build():\n    np.ctypeslib.load_library('x', '.')\n", "forbidden API numpy.ctypeslib"),
    ("import numpy as np\ndef build():\n    np.zeros(3).tofile('x')\n", "forbidden method 'tofile'"),
    ("import openvdb as vdb\ndef build():\n    vdb.write('x.vdb', grids=[])\n", "forbidden API openvdb.write"),
    ("import openvdb\ndef build():\n    openvdb.readAll('x.vdb')\n", "forbidden API openvdb.readAll"),
    ("import scipy\ndef build():\n    pass\n", "import of 'scipy'"),
    # a module's forbidden parts are found under whatever name it was imported
    ("import bpy as b\ndef build():\n    b.ops.wm.save_mainfile()\n", "forbidden API bpy.ops.wm"),
    ("from bpy import ops\ndef build():\n    ops.render.render()\n", "forbidden API bpy.ops.render"),
])
def test_the_parts_of_an_allowed_module_that_reach_files_are_refused(source, fragment):
    issues = check_script(source)
    assert any(fragment in i for i in issues), issues
    assert all(i.startswith("line ") for i in issues)


NEW_IMAGE = "import bpy\ndef build():\n    im = bpy.data.images.new('x', 4, 4)\n    "


@pytest.mark.parametrize("source, fragment", [
    # a reference picture read into the model, or a picture written anywhere
    ("import bpy\ndef build():\n    bpy.data.images.load('refs/front.png')\n", "forbidden method 'load' (it reads a file)"),
    ("import bpy\ndef build():\n    images = bpy.data.images\n    images.load('refs/front.png')\n", "forbidden method 'load'"),
    (NEW_IMAGE + "im.filepath_raw = 'out.png'\n", "forbidden attribute 'filepath_raw'"),
    (NEW_IMAGE + "im.source = 'FILE'\n    im.filepath = 'refs/front.png'\n", "forbidden attribute 'filepath'"),
    (NEW_IMAGE + "im.save()\n", "forbidden method 'save' (it writes a file)"),
    (NEW_IMAGE + "im.save_render('out.png')\n", "forbidden method 'save_render'"),
    (NEW_IMAGE + "im.unpack(method='WRITE_LOCAL')\n", "forbidden method 'unpack'"),
    ("import bpy\ndef build():\n    bpy.ops.image.save_as(filepath='out.png')\n", "forbidden API bpy.ops.image"),
    ("import bpy\ndef build():\n    bpy.ops.image.save_all_modified()\n", "forbidden API bpy.ops.image"),
    ("import bpy\ndef build():\n    bpy.ops.image.open(filepath='refs/front.png')\n", "forbidden API bpy.ops.image"),
    # the other collections of bpy.data read a file the same way
    ("import bpy\ndef build():\n    bpy.data.fonts.load('a.ttf')\n", "forbidden method 'load'"),
    ("import bpy\ndef build():\n    bpy.data.volumes.load('a.vdb')\n", "forbidden method 'load'"),
    # an importer or exporter, whatever it is for: bpy.ops.export_scene, import_mesh, ...
    ("import bpy\ndef build():\n    bpy.ops.export_scene.gltf(filepath='out.glb')\n", "forbidden API bpy.ops.export_scene"),
    ("import bpy\ndef build():\n    bpy.ops.import_scene.gltf(filepath='a.glb')\n", "forbidden API bpy.ops.import_scene"),
    ("import bpy\ndef build():\n    bpy.ops.import_mesh.stl(filepath='a.stl')\n", "forbidden API bpy.ops.import_mesh"),
    ("from bpy.ops import export_scene\ndef build():\n    pass\n", "forbidden API bpy.ops.export_scene"),
    # any other operator told where a file is
    ("import bpy\ndef build():\n    bpy.ops.object.bake(type='DIFFUSE', filepath='out.png', save_mode='EXTERNAL')\n",
     "a call given `filepath=`"),
    ("import bpy\ndef build():\n    bpy.ops.uv.export_layout(filepath='out.svg')\n", "a call given `filepath=`"),
    ("import bpy\ndef build():\n    bpy.ops.font.open(directory='C:/Windows/Fonts')\n", "a call given `directory=`"),
    ("import bpy\ndef build():\n    bpy.context.scene.render.filepath = 'out'\n", "forbidden attribute 'filepath'"),
])
def test_a_script_reads_no_picture_and_writes_no_file_through_blender(source, fragment):
    issues = check_script(source)
    assert any(fragment in i for i in issues), issues
    assert all(i.startswith("line ") for i in issues)


def test_an_image_made_in_the_script_is_still_allowed():
    """A picture the script computes itself (a displacement texture, say) reaches no file."""
    assert check_script(NEW_IMAGE + "im.pixels.foreach_set([0.5] * 64)\n"
                        "    tex = bpy.data.textures.new('t', 'IMAGE')\n    tex.image = im\n") == []


def test_one_forbidden_call_is_named_once_not_as_an_api_and_as_a_method():
    issues = check_script("import numpy as np\ndef build():\n    np.load('x.npy')\n")
    assert issues == ["line 3: forbidden API numpy.load"]


def test_a_repeated_offence_on_one_line_is_reported_once():
    issues = check_script("def build():\n    open('a'); open('b')\n")
    assert len([i for i in issues if "open" in i]) == 1
