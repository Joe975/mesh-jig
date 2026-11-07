"""Model-script containment check, shared by the host and the Blender runner (blender_runner.py refuses a script
before exec).

Pure Python, standard library only, so it runs under Blender's bundled Python as well as the host's.
Denylist, not a sandbox: stops hallucinated or careless escapes in a file that Blender will exec.
"""
import ast

# numpy and openvdb ship in Blender's own Python (official builds), so a script that uses them needs nothing a
# Blender user does not have. The parts of them that reach files or load code are in FORBIDDEN_ATTRS.
ALLOWED_IMPORTS = {"bpy", "bmesh", "mathutils", "math", "random", "numpy", "openvdb"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "breakpoint", "input", "memoryview", "type"}
# attribute chains that reach files, processes, the UI, other .blend files or the renderer. A chain is read
# through the name a module was imported under, so `import numpy as np; np.save` is `numpy.save`.
FORBIDDEN_ATTRS = ("bpy.ops.wm", "bpy.ops.render", "bpy.ops.script", "bpy.ops.preferences", "bpy.ops.file",
                   "bpy.ops.text", "bpy.ops.image", "bpy.ops.sound", "bpy.app", "bpy.utils", "bpy.path",
                   "bpy.data.libraries", "bpy.data.texts",
                   "bpy.data.filepath", "bpy.context.preferences",
                   "numpy.load", "numpy.save", "numpy.savez", "numpy.savez_compressed", "numpy.savetxt",
                   "numpy.loadtxt", "numpy.genfromtxt", "numpy.fromfile", "numpy.fromregex", "numpy.memmap",
                   "numpy.DataSource", "numpy.lib", "numpy.ctypeslib", "numpy.f2py", "numpy.distutils",
                   "numpy.testing", "numpy.core", "numpy._core",
                   "openvdb.read", "openvdb.readAll", "openvdb.readMetadata", "openvdb.readAllGridMetadata",
                   "openvdb.readGridMetadata", "openvdb.write")
# Blender's importers and exporters are a family of operator groups: bpy.ops.export_scene, import_mesh, ...
FORBIDDEN_OPS_PREFIXES = ("export", "import")
# methods that reach a file whatever they are called on: an array's own, and those of an image or of a collection
# of bpy.data (`bpy.data.images.load`), which would put a reference picture in the model or a picture on the disk
FORBIDDEN_METHODS = {"tofile": "writes", "dump": "writes", "save": "writes", "save_render": "writes",
                     "unpack": "writes", "load": "reads"}
# where a datablock's file is: an image is read from the path it is given, and saved to it
FORBIDDEN_FIELDS = {"filepath", "filepath_raw"}
# an operator told where a file is, whichever it is (a bake saved outside the file, a UV layout, a font)
FORBIDDEN_KEYWORDS = {"filepath", "directory"}


def _attr_chain(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _forbidden(chain: str) -> bool:
    if any(chain == f or chain.startswith(f + ".") for f in FORBIDDEN_ATTRS):
        return True
    parts = chain.split(".")
    return parts[:2] == ["bpy", "ops"] and len(parts) > 2 and parts[2].startswith(FORBIDDEN_OPS_PREFIXES)


def _imported_as(tree: ast.AST) -> dict[str, str]:
    """The name each import binds -> what it names in full: `import numpy as np` gives np -> numpy, `from bpy
    import ops` gives ops -> bpy.ops."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    out[a.asname] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                out[a.asname or a.name] = f"{node.module}.{a.name}"
    return out


def check_script(source: str, *, required=("build",), allowed_imports=None) -> list[str]:
    """Reasons (with line numbers) a model script must not be run; [] when clean."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"line {e.lineno or 0}: syntax error: {e.msg}"]
    allowed_imports = ALLOWED_IMPORTS if allowed_imports is None else allowed_imports
    names = _imported_as(tree)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.Import):
                wanted = [a.name for a in node.names]
            else:
                wanted = [node.module or ""] + [f"{node.module}.{a.name}" for a in node.names if node.module]
            for n in wanted:
                root = n.split(".", 1)[0]
                if root not in allowed_imports:
                    out.append(f"line {node.lineno}: import of {n!r} not allowed (only {sorted(allowed_imports)})")
                    break                                        # one line for `from os import a, b`
                if _forbidden(n):
                    out.append(f"line {node.lineno}: forbidden API {n}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            out.append(f"line {node.lineno}: forbidden builtin {node.id!r}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                out.append(f"line {node.lineno}: dunder attribute access {node.attr!r}")
            chain = _attr_chain(node)
            if chain:
                root, _, rest = chain.partition(".")
                chain = names.get(root, root) + ("." + rest if rest else "")
            if chain and _forbidden(chain):
                out.append(f"line {node.lineno}: forbidden API {chain}")
            elif node.attr in FORBIDDEN_METHODS:
                out.append(f"line {node.lineno}: forbidden method {node.attr!r} "
                           f"(it {FORBIDDEN_METHODS[node.attr]} a file)")
            elif node.attr in FORBIDDEN_FIELDS:
                out.append(f"line {node.lineno}: forbidden attribute {node.attr!r} (it names a file)")
        elif isinstance(node, ast.Call):
            for k in node.keywords:
                if k.arg in FORBIDDEN_KEYWORDS:
                    out.append(f"line {node.lineno}: a call given `{k.arg}=` (it names a file)")
    for required_name in required:
        if not any(isinstance(n, ast.FunctionDef) and n.name == required_name for n in tree.body):
            out.append(f"line 0: no top-level `def {required_name}():` (the runner calls {required_name}() then exports)")
    seen: list[str] = []
    for o in out:
        if o not in seen:
            seen.append(o)
    return seen


def pydata_problem(vertex_count: int, edges, faces) -> str:
    """What in a `Mesh.from_pydata` call would crash Blender rather than raise ("" when nothing): a face or an edge
    naming a vertex index outside the list, or one that is not a sequence of whole numbers. Blender takes such a mesh
    without a word and dies later, in `bm.from_mesh` or a normals pass, with an access violation and no traceback,
    which leaves a builder nothing to repair from. The runner asks this before every call. A face naming a vertex
    twice is not a problem here: `pydata_faces` mends it."""
    bad, first = 0, ""
    for kind, items in (("face", faces), ("edge", edges)):
        for k, item in enumerate(items or ()):
            try:
                corners = [i.__index__() for i in item]
            except (AttributeError, TypeError):
                corners = None
            outside = [i for i in corners or () if not 0 <= i < vertex_count]
            why = ("is not a sequence of whole numbers" if corners is None
                   else f"names vertex {outside[0]}, and the list holds {vertex_count} (indices 0 to "
                        f"{vertex_count - 1})" if outside
                   else f"has {len(corners)} corners" if kind == "edge" and len(corners) != 2 else "")
            if why:
                bad += 1
                first = first or f"{kind} {k} {tuple(item) if corners is not None else item!r} {why}"
    if not bad:
        return ""
    return first + (f"; {bad - 1} more like it" if bad > 1 else "")


def pydata_faces(faces) -> tuple[list, int]:
    """(the faces with each vertex a face names twice kept once, how many faces that changed or dropped). A face left
    with fewer than three corners is dropped. A lathe or a sphere with a pole writes such faces; Blender keeps most of
    them and mends them the same way, but a quad naming two poles twice (0, 1, 2, 2) has hung it for minutes, which a
    build reports only as a timeout. Faces are assumed to be index sequences (`pydata_problem` holds them to that)."""
    out, mended = [], 0
    for face in faces or ():
        corners = [i.__index__() for i in face]
        kept = list(dict.fromkeys(corners))
        if len(kept) != len(corners):
            mended += 1
            if len(kept) < 3:
                continue
        out.append(kept if len(kept) != len(corners) else face)
    return out, mended
