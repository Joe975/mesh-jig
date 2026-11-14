"""Compose a frozen-source body skin with local facial controls, without retargeting.

Body owns its original joints. Facial joints are appended under one body anchor.
Only that anchor's existing influence is redistributed; no weight is discarded.
The supported body joint rest frames are the translation-only frames rig.build
exports. Rotating/moving body clips and transformed source meshes are supported.
"""
import copy
import hashlib
import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np

from . import face, face_motion, glb, rig, rig_look, skin

META = 'mesh_jig_character'
SAFE = re.compile(r'^[A-Za-z0-9_.-]+$')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(value):
    if not isinstance(value, dict) or set(value)-{'version','anchor','neck','fps','max_influences','demos'}:
        raise ValueError('character contract needs known fields only')
    c = copy.deepcopy(value)
    if type(c.get('version')) is not int or c['version'] != 1:
        raise ValueError('character version must be 1')
    if not isinstance(c.get('anchor'), str) or not c['anchor']:
        raise ValueError('anchor must name one existing body joint')
    if c.get('neck') not in ('body','additive'):
        raise ValueError('choose neck ownership explicitly: body or additive')
    c.setdefault('fps',24)
    c.setdefault('max_influences',4)
    rig.contract({'fps':c['fps'],'max_influences':c['max_influences'],'clips':{'check':{'seconds':1}}})
    demos=c.setdefault('demos',[])
    if not isinstance(demos,list):
        raise ValueError('demos must be a list')
    names=set()
    for d in demos:
        if not isinstance(d,dict) or set(d)-{'name','body','expression','gaze_target'}:
            raise ValueError('demo needs name/body and optional expression/gaze_target')
        if not isinstance(d.get('name'),str) or not SAFE.fullmatch(d['name']) or d['name'] in names:
            raise ValueError('demo names must be safe and unique')
        names.add(d['name'])
        if not isinstance(d.get('body'),str) or not d['body']:
            raise ValueError('demo must name a body clip')
        if d.get('expression') not in (None,*face.STATES[1:]):
            raise ValueError('demo expression must be a non-gaze face state')
        if 'gaze_target' in d:
            rig.vector(d['gaze_target'])
    return c


def _validate_source(source, body, config):
    # Inspect the static geometry with the same facial contract before modifying a skin.
    inspection=face.inspect(source,config)
    if inspection['issues']:
        raise ValueError('; '.join(inspection['issues']))
    original,original_blob=glb.read_glb(source)
    doc,blob=glb.read_glb(body)
    if doc.get('extensionsRequired') or len(doc.get('buffers',[]))!=1 or not doc.get('skins'):
        raise ValueError('body must be an uncompressed single-buffer skinned GLB')
    if META in doc.get('extras',{}):
        raise ValueError('body is already composed; start from the original body rig')
    if any('sparse' in a for a in doc.get('accessors',[])):
        raise ValueError('sparse body accessors are unsupported')
    old_nodes=list(glb.mesh_nodes(original))
    body_nodes=list(glb.mesh_nodes(doc))
    if len(old_nodes)!=len(body_nodes):
        raise ValueError('source/body mesh instances differ; use the body rig original frozen source')
    for key in ('materials','textures','images','samplers'):
        if original.get(key)!=doc.get(key):
            raise ValueError('source/body '+key+' differ')
    for image in original.get('images',[]):
        if 'bufferView' not in image:
            raise ValueError('composition requires embedded images')
        a=original['bufferViews'][image['bufferView']];b=doc['bufferViews'][image['bufferView']]
        if original_blob[a.get('byteOffset',0):a.get('byteOffset',0)+a['byteLength']]!=blob[b.get('byteOffset',0):b.get('byteOffset',0)+b['byteLength']]:
            raise ValueError('source/body embedded image bytes differ')
    for (si,sw),(bi,bw) in zip(old_nodes,body_nodes):
        sn,bn=original['nodes'][si],doc['nodes'][bi]
        if sn.get('name')!=bn.get('name') or not np.allclose(sw,bw,atol=1e-8,rtol=0):
            raise ValueError('source/body node names or rest transforms differ')
        sp=original['meshes'][sn['mesh']]['primitives']
        bp=doc['meshes'][bn['mesh']]['primitives']
        if len(sp)!=len(bp) or 'skin' not in bn:
            raise ValueError('body must skin every source primitive')
        for a,b in zip(sp,bp):
            if a.get('mode',4)!=4 or b.get('mode',4)!=4 or b.get('targets'):
                raise ValueError('only triangle skins without morph targets are supported')
            if a.get('material')!=b.get('material'):
                raise ValueError('source/body primitive material differs')
            for key,ai in a['attributes'].items():
                if key not in b['attributes'] or not np.array_equal(glb.accessor(original,original_blob,ai),glb.accessor(doc,blob,b['attributes'][key])):
                    raise ValueError('source/body attribute differs: '+key)
            ai=a.get('indices'); bi_=b.get('indices')
            if (ai is None)!=(bi_ is None) or ai is not None and not np.array_equal(glb.accessor(original,original_blob,ai),glb.accessor(doc,blob,bi_)):
                raise ValueError('source/body topology differs')
            if any(k.startswith(('JOINTS_','WEIGHTS_')) and k not in ('JOINTS_0','WEIGHTS_0') for k in b['attributes']):
                raise ValueError('only JOINTS_0/WEIGHTS_0 skins are supported')
    rest=skin.worlds(doc)
    joints={j for s in doc['skins'] for j in s['joints']}
    names=[doc['nodes'][j].get('name') for j in joints]
    if any(not n or n.startswith('face_') for n in names) or len(set(names))!=len(names):
        raise ValueError('body joints need unique names outside reserved face_ namespace')
    if any(n.get('name','').startswith('face_') for n in doc['nodes']):
        raise ValueError('body node collides with reserved face_ namespace')
    for j in joints:
        if 'matrix' in doc['nodes'][j] or j not in rest or not np.allclose(rest[j][:3,:3],np.eye(3),atol=1e-7,rtol=0):
            raise ValueError('body joint rest frames must be model-axis aligned, unit scale (mesh-jig rig convention)')
    for ni,own in body_nodes:
        s=doc['skins'][doc['nodes'][ni]['skin']]
        if len(set(s['joints']))!=len(s['joints']) or 'inverseBindMatrices' not in s:
            raise ValueError('body skins need unique joints and explicit inverse binds')
        ib=glb.accessor(doc,blob,s['inverseBindMatrices']).reshape(-1,4,4).transpose(0,2,1)
        if len(ib)!=len(s['joints']):
            raise ValueError('body inverse bind count differs from joints')
        for j,m in zip(s['joints'],ib):
            if not np.allclose(rest[j]@m,own,atol=1e-5,rtol=0):
                raise ValueError('body inverse binds do not recover source mesh transforms')
    animations=doc.get('animations',[])
    clips=[a.get('name') for a in animations]
    if not clips or any(not isinstance(n,str) or not n or n.startswith(('face.','combined.')) for n in clips) or len(set(clips))!=len(clips):
        raise ValueError('body clips need unique names outside face./combined. namespaces')
    for a in animations:
        for ch in a['channels']:
            if ch['target']['node'] not in joints or ch['target']['path'] not in ('rotation','translation'):
                raise ValueError('body animation may target only joint rotation/translation')
            s=a['samplers'][ch['sampler']]
            times=glb.accessor(doc,blob,s['input']).ravel()
            values=glb.accessor(doc,blob,s['output'])
            width=4 if ch['target']['path']=='rotation' else 3
            if not len(times) or values.shape!=(len(times),width) or not np.isfinite(times).all() or not np.isfinite(values).all() or times[0]<0 or times[-1]>60:
                raise ValueError('body keyframes must be finite and within 0..60 seconds')
            if ch['target']['path']=='rotation' and not np.allclose(np.linalg.norm(values,axis=1),1,atol=1e-5):
                raise ValueError('body rotations must be unit quaternions')
        skin.channels(doc,blob,a['name'],0)  # validates interpolation, ownership and time ordering
    return doc,blob,rest,inspection


def _face_values(doc, blob, body_clip, body_t, expression, face_t, gaze_target, *, sample_expression=True):
    m=doc['extras'][META]
    cfg=m['face']
    parts=skin.channels(doc,blob,body_clip,body_t) if body_clip else {}
    base=skin.with_channels(doc,parts)
    poses=face_motion.pose_at(cfg,expression,face_t) if expression else {}
    gaze_owned=set()
    if m['neck']=='body':
        poses.pop('face_neck',None)
    if gaze_target is not None:
        # Undo the body's CURRENT anchor deformation before solving model-rest gaze.
        target=rig.to_gltf(gaze_target)
        deform=base[m['anchor_node']]@np.linalg.inv(np.array(m['anchor_rest']))
        target=(np.linalg.inv(deform)@np.r_[target,1])[:3]
        target=rig_look.to_blender(target)
        if m['neck']=='additive':
            gaze=face_motion.pose_at(cfg,'look_at',target=target)
            poses['face_neck']=gaze['face_neck']
            gaze_owned.add('face_neck')
            for eye in ('face_eye_l','face_eye_r'):
                if eye in gaze:
                    poses[eye]=gaze[eye]
                    gaze_owned.add(eye)
        else:
            for eye in ('eye_l','eye_r'):
                if eye in cfg['controls']:
                    yaw,pitch=face_motion.angles(target-np.asarray(cfg['controls'][eye]['pivot']),30,25)
                    poses['face_'+eye]={'rotate':[('X',pitch),('Z',yaw)]}
                    gaze_owned.add('face_'+eye)
    values={}
    # Author all facial channels, including neutral eyes: switching clips cannot leave old state behind.
    for name,i in m['face_nodes'].items():
        delta=poses.get(name,{})
        values[i]={'rotation':rig.quaternion(delta.get('rotate',[])).tolist(),
                   'translation':(np.asarray(doc['nodes'][i]['translation'])+rig.to_gltf(delta.get('move',[0,0,0]))).tolist()}
    if sample_expression and expression:
        sampled=skin.channels(doc,blob,'face.'+expression,face_t%2)
        for name,i in m['face_nodes'].items():
            if name not in gaze_owned:
                values[i]=sampled[i]
    return parts,values


class Player:
    """Stateless per-frame body + expression + optional model-space point gaze.

    The body clip is sampled once. Gaze owns eyes and (with additive neck policy)
    face_neck; the expression owns jaw/corners/brows. No old frame state survives.
    """
    def __init__(self,path):
        self.path=Path(path)
        self.doc,self.blob=glb.read_glb(path)
        if self.doc.get('extras',{}).get(META,{}).get('version')!=1:
            raise ValueError('not a mesh-jig composed character')
        self.meta=self.doc['extras'][META]

    def worlds(self,body=None,t=0,*,expression=None,face_time=None,gaze_target=None):
        if body is not None and body not in self.meta['body_clips']:
            raise ValueError('select an original body clip for live composition, not a baked/face clip')
        if expression not in (None,*face.STATES[1:]):
            raise ValueError('expression must be a non-gaze face state; pass gaze_target for gaze')
        ft=t if face_time is None else face_time
        if not math.isfinite(t) or not math.isfinite(ft):
            raise ValueError('animation times must be finite')
        parts,values=_face_values(self.doc,self.blob,body,t,expression,ft,gaze_target)
        parts.update(values)
        return skin.with_channels(self.doc,parts)

    def mesh(self,*args,**kwargs):
        return skin.posed(self.path,world=self.worlds(*args,**kwargs))


def _append_animation(doc,blob,name,times,values,kind):
    a={'name':name,'channels':[],'samplers':[],'extras':{'mesh_jig_kind':kind,'loop':False}}
    ti=rig.append_accessor(doc,blob,times,'SCALAR')
    for node in values[0]:
        for path,shape in [('rotation','VEC4'),('translation','VEC3')]:
            out=rig.append_accessor(doc,blob,[v[node][path] for v in values],shape)
            a['channels'].append({'sampler':len(a['samplers']),'target':{'node':node,'path':path}})
            a['samplers'].append({'input':ti,'output':out,'interpolation':'LINEAR'})
    doc['animations'].append(a)
    return a


def compose(source,body,face_config,spec,out):
    source,body,out=Path(source),Path(body),Path(out)
    if out.resolve() in (source.resolve(),body.resolve()):
        raise ValueError('composition output must not overwrite either input')
    c=contract(spec)
    cfg=face.contract(face_config)
    doc,binary,rest,inspection=_validate_source(source,body,cfg)
    matches=[i for i,n in enumerate(doc['nodes']) if n.get('name')==c['anchor']]
    joint_ids={j for s in doc['skins'] for j in s['joints']}
    if len(matches)!=1 or matches[0] not in joint_ids:
        raise ValueError('anchor must resolve to exactly one skinned body joint')
    anchor=matches[0]
    original_animations=copy.deepcopy(doc['animations'])
    body_clips=[a['name'] for a in original_animations]
    if any(d['body'] not in body_clips for d in c['demos']):
        raise ValueError('demo names an absent body clip')
    original_meshes=list(glb.mesh_nodes(doc))
    face_nodes={}
    for control,data in cfg['controls'].items():
        name='face_'+control
        parent=anchor if control=='neck' else face_nodes['face_neck']
        pivot=rig.to_gltf(data['pivot'])
        parent_pivot=rest[anchor][:3,3] if control=='neck' else rig.to_gltf(cfg['controls']['neck']['pivot'])
        i=len(doc['nodes']); face_nodes[name]=i
        doc['nodes'].append({'name':name,'translation':(pivot-parent_pivot).tolist()})
        doc['nodes'][parent].setdefault('children',[]).append(i)
    doc.setdefault('extras',{})[META]={'version':1,'anchor_node':anchor,'anchor_rest':rest[anchor].tolist(),
        'neck':c['neck'],'face_nodes':face_nodes,'body_clips':body_clips,'face':cfg,
        'source_sha256':sha(source),'body_sha256':sha(body),
        'ownership':'original joints: body; appended face joints: face. Gaze overrides expression neck/eyes only.'}
    blob=bytearray(binary)
    count=0; max_count=0
    control_totals=dict.fromkeys(face_nodes,0)
    for ni,own in original_meshes:
        node=doc['nodes'][ni]
        old_skin=doc['skins'][node['skin']]
        new_skin=copy.deepcopy(old_skin)
        old_joints=old_skin['joints']
        inverse=glb.accessor(doc,binary,old_skin['inverseBindMatrices']).copy().tolist()
        new_skin['joints']=old_joints+list(face_nodes.values())
        if len(new_skin['joints'])>65535:
            raise ValueError('composed skin exceeds unsigned-short joint capacity')
        for name in face_nodes:
            world=np.eye(4); world[:3,3]=rig.to_gltf(cfg['controls'][name[5:]]['pivot'])
            inverse.append((np.linalg.inv(world)@own).T.reshape(16).tolist())
        new_skin['inverseBindMatrices']=rig.append_accessor(doc,blob,inverse,'MAT4')
        mesh=copy.deepcopy(doc['meshes'][node['mesh']])
        for prim in mesh['primitives']:
            attrs=prim['attributes']
            if not {'JOINTS_0','WEIGHTS_0'}<=attrs.keys() or doc['accessors'][attrs['JOINTS_0']]['componentType'] not in (5121,5123):
                raise ValueError('body needs unsigned integer JOINTS_0 and WEIGHTS_0')
            joints=glb.accessor(doc,binary,attrs['JOINTS_0']).astype(int)
            weights=glb.accessor(doc,binary,attrs['WEIGHTS_0']).astype(float)
            points=glb.accessor(doc,binary,attrs['POSITION'])
            if joints.shape!=weights.shape or weights.shape!=(len(points),4) or np.any(joints<0) or np.any(joints>=len(old_joints)) or not np.isfinite(weights).all() or (weights<0).any() or not np.allclose(weights.sum(1),1,atol=1e-6):
                raise ValueError('body weights must be finite, nonnegative, normalized four-slot influences')
            points=rig_look.to_blender(points@own[:3,:3].T+own[:3,3])
            facial=face_motion.make_weights(cfg,node['name'],points)
            head=1-facial['face_root']
            aw=(np.where(joints==old_joints.index(anchor),weights,0).sum(1) if anchor in old_joints else np.zeros(len(points)))
            if np.any((head>1e-7)&(aw<1e-7)):
                raise ValueError(node['name']+': facial selection has no anchor weight; repair the body binding or narrow the face region')
            rows=np.zeros((len(points),len(new_skin['joints'])))
            for slot in range(4):
                np.add.at(rows,(np.arange(len(points)),joints[:,slot]),weights[:,slot])
            if anchor in old_joints:
                rows[:,old_joints.index(anchor)]*=1-head
            for k,name in enumerate(face_nodes):
                rows[:,len(old_joints)+k]=aw*facial[name]
                control_totals[name]+=int(np.count_nonzero(rows[:,len(old_joints)+k]>1e-7))
            influence=(rows>0).sum(1)
            if (influence>c['max_influences']).any():
                raise ValueError(node['name']+': composed weights exceed influence limit; no pruning is permitted')
            order=np.argsort(-rows,axis=1,kind='stable')[:,:4]
            kept=np.take_along_axis(rows,order,axis=1)
            order=np.pad(order,((0,0),(0,4-order.shape[1])))
            kept=np.pad(kept,((0,0),(0,4-kept.shape[1])))
            attrs['JOINTS_0']=rig.append_accessor(doc,blob,order,'VEC4',5123)
            attrs['WEIGHTS_0']=rig.append_accessor(doc,blob,kept,'VEC4')
            count+=int(np.count_nonzero(aw*head>0)); max_count=max(max_count,int(influence.max()))
        node['mesh']=len(doc['meshes']);doc['meshes'].append(mesh)
        node['skin']=len(doc['skins']);doc['skins'].append(new_skin)
    if not count or any(v==0 for k,v in control_totals.items() if k!='face_neck'):
        raise ValueError('composition leaves requested facial controls without anchor-weighted vertices')
    # Standalone facial clips share no targets with the untouched body clips.
    times=np.linspace(0,2,2*c['fps']+1)
    for state in face.STATES:
        values=[]
        for t in times:
            _,v=_face_values(doc,blob,None,0,None if state=='look_at' else state,float(t),cfg['look_target'] if state=='look_at' else None,sample_expression=False)
            values.append(v)
        _append_animation(doc,blob,'face.'+state,times,values,'independent-face')['extras']['loop']=True
    for demo in c['demos']:
        body_anim=next(a for a in original_animations if a['name']==demo['body'])
        duration=max(float(glb.accessor(doc,blob,s['input']).max()) for s in body_anim['samplers'])
        if duration<=0:
            raise ValueError('baked demonstration needs a positive body duration')
        times=np.linspace(0,duration,max(1,math.ceil(duration*c['fps']))+1)
        vals=[_face_values(doc,blob,demo['body'],float(t),demo.get('expression'),float(t),demo.get('gaze_target'))[1] for t in times]
        a=_append_animation(doc,blob,'combined.'+demo['name'],times,vals,'baked-simultaneous')
        offset=len(a['samplers'])
        a['samplers']+=copy.deepcopy(body_anim['samplers'])
        for ch in body_anim['channels']:
            ch=copy.deepcopy(ch); ch['sampler']+=offset; a['channels'].append(ch)
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix='character-',suffix='.glb',dir=out.parent,delete=False) as temp:
        candidate=Path(temp.name)
    try:
        rig.write_glb(candidate,doc,blob)
        # Read the actual export back, checking preservation and neutral-body replay.
        written,wb=glb.read_glb(candidate)
        if written['animations'][:len(original_animations)]!=original_animations or bytes(wb[:len(binary)])!=bytes(binary):
            raise ValueError('original body animation or binary data changed')
        error=float(np.max(np.abs(skin.posed(body).tris-skin.posed(candidate).tris)))
        body_error=error
        for name in body_clips:
            a=next(a for a in original_animations if a['name']==name)
            end=max(float(glb.accessor(written,wb,s['input']).max()) for s in a['samplers'])
            for t in (0,end*.371,end):
                body_error=max(body_error,float(np.max(np.abs(skin.posed(body,name,t).tris-skin.posed(candidate,name,t).tris))))
        if body_error>1e-5:
            raise ValueError('neutral-face composition did not preserve body skin playback')
        report={'version':1,'source_sha256':sha(source),'body_sha256':sha(body),'contract':c,
                'face_contract':cfg,'face_inspection':inspection,'weighted_vertices':count,
                'max_influences':max_count,'rest_error_m':error,'body_replay_error_m':body_error,
                'body_clips':body_clips,'face_clips':['face.'+s for s in face.STATES],
                'baked_clips':['combined.'+d['name'] for d in c['demos']],
                'visual_approval':'pending','notes':['Composition preserves motion; it does not repair closed-mouth topology.']}
        candidate.replace(out)
        report['output_sha256']=sha(out)
        out.with_suffix('.character.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        return report
    finally:
        if candidate.exists():
            candidate.unlink()


def preview(path,out,body,expression=None,gaze_target=None,seconds=2,fps=12):
    from PIL import Image
    if not math.isfinite(seconds) or not 0<seconds<=60 or type(fps) is not int or not 1<=fps<=60:
        raise ValueError('preview seconds must be in (0,60], fps integer in 1..60')
    player=Player(path)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    model,_=rig.model_data(player.doc,player.blob)
    view=player.meta['face'].get('preview')
    neck=player.meta['face_nodes']['face_neck']
    rest_inverse=np.linalg.inv(skin.worlds(player.doc)[neck])
    images=[]
    for index,t in enumerate(np.linspace(0,seconds,max(2,math.ceil(seconds*fps)),endpoint=False)):
        world=player.worlds(body,float(t),expression=expression,gaze_target=gaze_target)
        mesh=skin.posed(path,world=world)
        full=rig_look.draw(mesh,rig_look.shot('front',model['lo'],model['hi']),320)
        panel=Image.new('RGB',(640,320),'white');panel.paste(full,(0,0))
        close=dict(view or {})
        if 'aim' in close:
            aim=(world[neck]@rest_inverse@np.r_[rig.to_gltf(close['aim']),1])[:3]
            close['aim']=rig_look.to_blender(aim).tolist()
        panel.paste(rig_look.draw(mesh,rig_look.shot('front',model['lo'],model['hi'],**close),320),(320,0))
        panel.save(out/f'{index:03d}.png');images.append(panel)
    images[0].save(out/'playback.gif',save_all=True,append_images=images[1:],duration=round(1000/fps),loop=0)
    return out/'playback.gif'


def add_commands(subs):
    p=subs.add_parser('character-compose',help='attach facial controls to a compatible existing body skin')
    for flag in ('source','body','face','contract','out'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.set_defaults(fn=lambda a: _cli_compose(a))
    p=subs.add_parser('character-preview',help='play simultaneous body/expression/gaze through the character player')
    p.add_argument('--rig',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--body',required=True);p.add_argument('--expression',choices=face.STATES[1:])
    p.add_argument('--gaze',nargs=3,type=float);p.add_argument('--seconds',type=float,default=2)
    p.add_argument('--fps',type=int,default=12)
    p.set_defaults(fn=lambda a: _cli_preview(a))


def _cli_compose(a):
    print(json.dumps(compose(a.source,a.body,json.loads(a.face.read_text(encoding='utf-8')),
                             json.loads(a.contract.read_text(encoding='utf-8')),a.out),indent=2))
    return 0


def _cli_preview(a):
    print(preview(a.rig,a.out,a.body,a.expression,a.gaze,a.seconds,a.fps))
    return 0
