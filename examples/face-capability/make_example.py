"""Offline schematic face: hinged lower lip, real recessed mouth, one visible eye.

Run from a mesh-jig development environment:
  python examples/face-capability/make_example.py local-work/face-example
Creates a new project only; no Blender, APIs or existing character assets needed.
This intentionally simple puppet is a functional regression, not an art reference.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from mesh_jig import face_capability as facial, glb, project, render, rig, rig_look


def geometry(blocked=False, with_body=False):
    """Boxes in Blender coordinates; the front is +Y, up is +Z."""
    parts = [
        ('forehead', (.85,.20,.46), (0,0,.65), (.44,.23,.12)),
        ('upper_lip', (.56,.06,.04), (0,.10,.39), (.65,.15,.12)),
        ('lower_lip', (.56,.06,.04), (0,.10,.35), (.65,.15,.12)),
        ('chin', (.60,.10,.15), (0,.10,.225), (.44,.23,.12)),
        ('interior', (.72,.04,.46), (0,-.24,.22), (.025,.007,.008)),
        ('cheek_l', (.17,.38,.48), (-.385,-.05,.25), (.44,.23,.12)),
        ('cheek_r', (.17,.38,.48), (.385,-.05,.25), (.44,.23,.12)),
        ('mouth_l', (.06,.05,.08), (-.29,.17,.36), (.65,.15,.12)),
        ('mouth_r', (.06,.05,.08), (.29,.17,.36), (.65,.15,.12)),
        ('brow_l', (.23,.05,.05), (-.22,.15,.82), (.08,.025,.01)),
        ('brow_r', (.23,.05,.05), (.22,.15,.82), (.08,.025,.01)),
        ('patch', (.26,.06,.20), (-.22,.15,.65), (.015,.02,.03)),
        ('eye_r', (.24,.08,.18), (.22,.16,.65), (.85,.85,.72)),
        ('iris_r', (.11,.025,.13), (.22,.211,.65), (.03,.30,.24)),
        ('pupil_r', (.045,.015,.08), (.22,.234,.65), (.008,.008,.008)),
    ]
    if blocked:
        # A lying open_mouth declaration cannot make this front plate a cavity.
        parts.append(('closed_front', (.58,.03,.49), (0,.07,.17), (.44,.23,.12)))
    if with_body:
        parts += [('body',(.65,.35,1.0),(0,0,-.5),(.09,.19,.32)),
                  ('leg_l',(.24,.29,.8),(-.20,0,-1.4),(.12,.08,.045)),
                  ('leg_r',(.24,.29,.8),(.20,0,-1.4),(.12,.08,.045))]
    return parts


def write_source(path, blocked=False, with_body=False):
    doc = {'asset':{'version':'2.0'}, 'scene':0, 'scenes':[{'nodes':[]}],
           'nodes':[], 'meshes':[], 'bufferViews':[], 'accessors':[], 'buffers':[{}]}
    blob = bytearray()
    # Vertex order and winding stay right-handed through Blender -> glTF conversion.
    faces = np.array([(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
                      (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)],np.uint16)
    corners = np.array([(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                        (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)])
    for name, size, centre, colour in geometry(blocked,with_body):
        p = np.array([rig.to_gltf(v) for v in corners*np.array(size)/2+centre])
        attrs = {'POSITION':rig.append_accessor(doc,blob,p,'VEC3'),
                 'COLOR_0':rig.append_accessor(doc,blob,np.tile([*colour,1],(len(p),1)),'VEC4')}
        indices = rig.append_accessor(doc,blob,faces.ravel(),'SCALAR',5123)
        doc['meshes'].append({'name':name,'primitives':[{'attributes':attrs,'indices':indices}]})
        doc['nodes'].append({'name':name,'mesh':len(doc['meshes'])-1})
        doc['scenes'][0]['nodes'].append(len(doc['nodes'])-1)
    rig.write_glb(path,doc,blob)
    return Path(path)


def binding(blocked=False):
    def region(*names):
        return {'nodes':list(names)}
    names = [p[0] for p in geometry(blocked)]
    controls = {'neck':{'pivot':[0,0,0], 'region':region(*names), 'rigid_nodes':names},
                'jaw':{'pivot':[0,-.13,.36], 'region':region('lower_lip','chin')},
                'eye_r':{'pivot':[.22,.16,.65], 'region':region('eye_r','iris_r','pupil_r')}}
    for name in ('mouth_l','mouth_r','brow_l','brow_r'):
        centre = next(p[2] for p in geometry() if p[0]==name)
        controls[name] = {'pivot':list(centre),'region':region(name)}
    return {'version':1, 'rig':{'version':1, 'visible_eyes':['right'], 'face_scale_m':.6, 'look_target':[0,3,.65],
            'jaw_open_degrees':35, 'mouth_topology':'open_mouth', 'controls':controls,
            'preview':{'aim':[0,0,.46],'span':1.2}},
            'semantics':{'mouth':{k:region(n) for k,n in [('upper_lip','upper_lip'),('lower_lip','lower_lip'),('interior','interior')]},
                         'eyes':{'right':{k:region(n) for k,n in [('surface','eye_r'),('iris','iris_r'),('pupil','pupil_r')]}}}}


def model_script(blocked=False):
    """The same fixture as a normal, sandbox-compatible Blender modelling recipe."""
    return 'import bpy\n\nPARTS = '+repr(geometry(blocked))+'''

def build():
    corners = [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
               (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]
    faces = [(0,2,1),(0,3,2),(4,5,6),(4,6,7),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]
    for name,size,centre,colour in PARTS:
        vertices = [tuple(v[i]*size[i]/2+centre[i] for i in range(3)) for v in corners]
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices,[],faces)
        attr = mesh.color_attributes.new(name='Color',type='FLOAT_COLOR',domain='CORNER')
        for c in attr.data:
            c.color = (*colour,1)
        bpy.context.scene.collection.objects.link(bpy.data.objects.new(name,mesh))
'''


def make(root, blocked=False):
    root = Path(root).resolve()
    if root.exists():
        raise ValueError('Use a new directory; the example never overwrites a project')
    attempt = root/'attempts'/'a01'
    (attempt/'out').mkdir(parents=True)
    source = write_source(attempt/'out'/'puppet.glb',blocked)
    # The example's actual export recipe is its script; changing it invalidates the seal.
    (attempt/'model.py').write_text(model_script(blocked),encoding='utf-8')
    requirements = {'version':1,'required':list(facial.STATES),'eyes':['right']}
    (root/'jig.json').write_text(json.dumps({'name':'facial-puppet','model':{'script':'model.py','glb':'puppet.glb','join':False,
                                         'single_mesh':False,'shots':[{'name':'front','cam_pos':[0,.46,-3],'cam_target':[0,.46,0],'ortho':1.2}]},
                                         'face_capabilities':requirements},indent=2),encoding='utf-8')
    (root/'brief.md').write_text('A schematic brown puppet with one visible green eye, a black patch and lips meeting at rest. Preserve the neutral appearance.',encoding='utf-8')
    (root/'refs').mkdir()
    shot = rig_look.shot('front',[-.5,-.3,0],[.5,.3,1],aim=[0,0,.46],span=1.2)
    rig_look.draw(render.load(source),shot,320).save(root/'refs'/'front.png')
    (attempt/facial.FILE).write_text(json.dumps(binding(blocked),indent=2),encoding='utf-8')
    proj = project.load(root)
    facial.seal(proj,attempt)
    report,pictures = facial.evaluate(proj,attempt/'model.py',source,attempt)
    (attempt/'face-report.json').write_text(json.dumps({'report':report,'pictures':pictures},indent=2),encoding='utf-8')
    return proj,attempt,report,pictures


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--blocked',action='store_true',help='add a solid front surface: speech/yell must fail')
    args = parser.parse_args()
    _,attempt,report,_ = make(args.root,args.blocked)
    print(facial.feedback(report))
    print('Evidence:',attempt/'face-report.json')
