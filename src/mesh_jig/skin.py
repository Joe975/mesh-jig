"""Sample exported LINEAR/STEP joint animation and linear blend skins on the CPU.

The static renderer and its historic scores are deliberately unchanged. Quaternion
channels use shortest-arc slerp, including between keys, as glTF requires.
"""
from pathlib import Path

import numpy as np

from . import glb, render


def slerp(a, b, u):
    a, b = np.asarray(a, float), np.asarray(b, float)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    dot = min(dot, 1.0)
    if dot > .9995:
        q = a + u * (b - a)
    else:
        angle = np.arccos(dot)
        q = (np.sin((1-u)*angle)*a + np.sin(u*angle)*b) / np.sin(angle)
    return q / np.linalg.norm(q)


def worlds(doc, local=None):
    local, result = local or {}, {}
    visiting = set()
    def visit(i, parent):
        if i in visiting or i in result:
            raise ValueError("node hierarchy is cyclic or has multiple parents")
        visiting.add(i)
        result[i] = parent @ local.get(i, glb.node_matrix(doc['nodes'][i]))
        for child in doc['nodes'][i].get('children', []):
            visit(child, result[i])
        visiting.remove(i)
    for i in doc['scenes'][doc.get('scene', 0)]['nodes']:
        visit(i, np.eye(4))
    return result


def sample(doc, blob, clip, t):
    anim = next((a for a in doc.get('animations', []) if a.get('name') == clip), None)
    if anim is None:
        raise ValueError(f"no clip {clip!r}")
    parts = {}
    for channel in anim['channels']:
        sampler = anim['samplers'][channel['sampler']]
        times = glb.accessor(doc, blob, sampler['input']).ravel()
        values = glb.accessor(doc, blob, sampler['output'])
        interpolation = sampler.get('interpolation', 'LINEAR')
        if interpolation not in ('LINEAR', 'STEP'):
            raise ValueError(f"unsupported interpolation {interpolation}")
        if len(times) == 0 or np.any(np.diff(times) <= 0):
            raise ValueError("animation times must be nonempty and increasing")
        k = int(np.clip(np.searchsorted(times, t, side='right') - 1, 0, len(times)-1))
        v = values[k]
        if interpolation == 'LINEAR' and k < len(times)-1:
            u = float(np.clip((t-times[k])/(times[k+1]-times[k]), 0, 1))
            v = (slerp(values[k], values[k+1], u) if channel['target']['path'] == 'rotation'
                 else values[k]*(1-u) + values[k+1]*u)
        parts.setdefault(channel['target']['node'], {})[channel['target']['path']] = v.tolist()
    return worlds(doc, {i: glb.node_matrix({**{k: v for k, v in doc['nodes'][i].items() if k != 'matrix'}, **p})
                        for i, p in parts.items()})


def primitive_positions(doc, blob, ni, prim, world):
    attrs = prim['attributes']
    pos = glb.accessor(doc, blob, attrs['POSITION'])
    node = doc['nodes'][ni]
    if 'skin' not in node:
        return pos @ world[ni][:3, :3].T + world[ni][:3, 3], None
    spec = doc['skins'][node['skin']]
    inv = (glb.accessor(doc, blob, spec['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
           if 'inverseBindMatrices' in spec else np.repeat(np.eye(4)[None], len(spec['joints']), axis=0))
    joint = np.stack([world[j] @ inv[k] for k, j in enumerate(spec['joints'])])
    indices = glb.accessor(doc, blob, attrs['JOINTS_0']).astype(int)
    weights = glb.accessor(doc, blob, attrs['WEIGHTS_0'])
    blend = (joint[indices] * weights[..., None, None]).sum(axis=1)
    return np.einsum('nij,nj->ni', blend, np.c_[pos, np.ones(len(pos))])[:, :3], blend


def posed(path: str | Path, clip=None, t=0):
    doc, blob = glb.read_glb(path)
    world = worlds(doc) if clip is None else sample(doc, blob, clip, t)
    # Reuse colour/texture decoding in the static loader. Traversal/primitive order is identical.
    base = render.load(path, textured=bool(doc.get('images')), multiply=True)
    triangles, normals, offset = [], [], 0
    for ni, _ in glb.mesh_nodes(doc):
        for prim in doc['meshes'][doc['nodes'][ni]['mesh']]['primitives']:
            if prim.get('mode', 4) != 4:
                continue
            attrs = prim['attributes']
            pos, blend = primitive_positions(doc, blob, ni, prim, world)
            idx = (glb.accessor(doc, blob, prim['indices']).astype(int).ravel() if 'indices' in prim else np.arange(len(pos)))
            tri = idx.reshape(-1, 3)
            triangles.append(pos[tri])
            if 'NORMAL' in attrs:
                n = glb.accessor(doc, blob, attrs['NORMAL'])
                # glTF linear blend skinning also linearly blends the joint normal transforms.
                if blend is not None:
                    n = np.einsum('nij,nj->ni', blend[:, :3, :3], n)
                else:
                    n = n @ np.linalg.inv(world[ni][:3, :3])
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
                normals.append(n[tri])
            else:
                p = pos[tri]
                n = np.cross(p[:, 1]-p[:, 0], p[:, 2]-p[:, 0])
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
                normals.append(np.repeat(n[:, None], 3, axis=1))
            offset += len(tri)
    return render.Mesh(np.concatenate(triangles), base.colors, np.concatenate(normals),
                       base.uvs, base.tex_index, base.textures)
