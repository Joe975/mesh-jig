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


def check_script(source: str) -> list[str]:
    """Reasons (with line numbers) a model script must not be run; [] when clean."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"line {e.lineno or 0}: syntax error: {e.msg}"]
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
                if root not in ALLOWED_IMPORTS:
                    out.append(f"line {node.lineno}: import of {n!r} not allowed (only {sorted(ALLOWED_IMPORTS)})")
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
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build" for n in tree.body):
        out.append("line 0: no top-level `def build():` (the runner calls build() then exports)")
    seen: list[str] = []
    for o in out:
        if o not in seen:
            seen.append(o)
    return seen
