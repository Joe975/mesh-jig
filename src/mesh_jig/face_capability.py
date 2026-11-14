"""Early, optional modelling requirements and artifact-bound facial smoke tests.

The automatic verdict is sampled evidence, never anatomical or visual approval.
No modelling score, renderer, or unflagged project's behavior is changed.
"""
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from . import face, face_motion, glb, render, review, rig, rig_look, skin

FILE = 'face_bindings.json'
STATES = face.STATES
DIGEST = re.compile(r'^[0-9a-f]{64}$')
BIND_TOOL = {'type': 'function', 'function': {'name': 'bind_face',
    'description': 'Seal the current attempt face_bindings.json to its final model script and fixed facial requirements. Call after final model/binding edits, before submit.',
    'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contract(value):
    if not isinstance(value, dict) or set(value)-{'version', 'required', 'optional', 'eyes', 'min_mouth_gap_m', 'min_interior_depth_m'}:
        raise ValueError('face_capabilities must be an object with known fields')
    if type(value.get('version')) is not int or value['version'] != 1:
        raise ValueError('face_capabilities.version must be 1')
    result = dict(value)
    for key in ('required', 'optional'):
        names = value.get(key, [])
        if not isinstance(names, list) or any(not isinstance(n, str) or n not in STATES for n in names) or len(set(names)) != len(names):
            raise ValueError(f'face_capabilities.{key} must list unique known states')
        result[key] = list(names)
    requested = result['required']+result['optional']
    if not requested or set(result['required']) & set(result['optional']):
        raise ValueError('face_capabilities needs distinct required/optional capabilities')
    eyes = value.get('eyes', [])
    if not isinstance(eyes, list) or any(e not in ('left', 'right') for e in eyes) or len(set(eyes)) != len(eyes):
        raise ValueError('face_capabilities.eyes must list unique left/right visible eyes')
    if ('look_at' in requested) != bool(eyes):
        raise ValueError('face_capabilities: gaze needs explicit visible eyes; omit eyes without gaze')
    result['eyes'] = list(eyes)
    for key, default in [('min_mouth_gap_m', .02), ('min_interior_depth_m', .01)]:
        x = value.get(key, default)
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not np.isfinite(x) or x <= 0:
            raise ValueError(f'face_capabilities.{key} must be finite positive metres')
        result[key] = float(x)
    return result


def requested(requirements):
    return requirements['required']+requirements['optional']


def needs(state, eyes):
    return ({'neck'} | {'eye_'+e[0] for e in eyes} if state == 'look_at' else {
        'talk': {'neck', 'jaw', 'mouth_l', 'mouth_r'}, 'yell': {'jaw', 'mouth_l', 'mouth_r', 'brow_l', 'brow_r'},
        'smile': {'mouth_l', 'mouth_r'}, 'anger': {'brow_l', 'brow_r'},
        'sadness': {'mouth_l', 'mouth_r', 'brow_l', 'brow_r'}, 'confusion': {'neck', 'brow_l', 'brow_r'}}[state])


def binding_template(c):
    """Self-contained grammar for file-only API builders with no access to repository docs."""
    controls={'neck':{'pivot':[0,0,0],'region':{'nodes':['Replace_Head']}}}
    semantics={}
    for name in sorted(set().union(*(needs(s,c['eyes']) for s in requested(c)))-{'neck'}):
        controls[name]={'pivot':[0,0,0],'region':{'nodes':['Replace_'+name]}}
    if 'look_at' in requested(c):
        semantics['eyes']={eye:{role:{'nodes':['Replace_'+eye+'_'+role]} for role in ('surface','iris','pupil')} for eye in c['eyes']}
        for eye in c['eyes']:
            controls['eye_'+eye[0]]['region']['nodes']=[r['nodes'][0] for r in semantics['eyes'][eye].values()]
    if set(requested(c))&{'talk','yell'}:
        semantics['mouth']={role:{'nodes':['Replace_'+role]} for role in ('upper_lip','lower_lip','interior')}
        controls['jaw']['region']['nodes']=['Replace_lower_lip']
    names={n for control in controls.values() for n in control['region']['nodes']}
    names.update(r['nodes'][0] for r in semantics.get('mouth',{}).values())
    controls['neck']['region']['nodes']=sorted(names)
    rig_config={'version':1,'face_scale_m':1.,'look_target':[0,3,1],
                'preview':{'aim':[0,0,1],'span':1.},'controls':controls}
    if c['eyes']:
        rig_config['visible_eyes']=list(c['eyes'])
    return {'version':1,'rig':rig_config,'semantics':semantics}


def brief(value, *, include_binding=True):
    if value is None:
        return ''
    c = contract(value)
    lines = ['FACIAL CAPABILITIES (fixed before modelling; separate from visual fidelity):',
             'Required: '+', '.join(c['required'])+'; optional: '+(', '.join(c['optional']) or 'none')+'.',
             'Match the reference in neutral, including a closed mouth at rest. Choose suitable geometry; no one topology technique is required.']
    states = requested(c)
    if 'look_at' in states:
        lines.append('Gaze: independently aimable visible '+', '.join(c['eyes'])+' eye(s), including their iris/pupil surface regions, carried by a turning neck. Do not invent an eye behind a patch.')
    if set(states) & {'talk', 'yell'}:
        lines.append(f'Talking/yelling: articulated jaw, upper/lower lips and a real opening leading to interior geometry. Open-pose lip gap at least {c["min_mouth_gap_m"]:g} m and sampled interior depth at least {c["min_interior_depth_m"]:g} m. A painted line, solid face, or open_mouth declaration is not evidence of a cavity.')
    for state in states:
        lines.append(state+': bind '+', '.join(sorted(needs(state, c['eyes'])))+'.')
    lines += ['Expressions need deformable mouth corners/brows as listed; yelling needs wide opening plus angry brows. Talking is subtle, not audio lip sync.',
              'Write face_bindings.json beside the model script: version 1, rig (face-stage contract), semantics (eyes with surface/iris/pupil regions; mouth with upper_lip/lower_lip/interior regions when speech is requested). Regions use exact exported node names and optional boxes/ellipsoids. Include rig.preview aim/span for close-ups. See docs/FACE_CAPABILITIES.md for a complete example.',
              'After final model and binding edits, seal them: coding CLI `mesh-jig face-bind <project> <attempt>`; API visual builder `bind_face` tool, then submit. Changed scripts/contracts invalidate the seal. A copied binding must be reviewed against the new geometry.',
              'Evaluation runs neutral, mouth-open, gaze extremes and requested expression close-ups. Read FACIAL FUNCTION feedback even when ordinary scores are withheld. Repair required failures while preserving neutral resemblance.',
              'Sampled motion/visibility/depth checks establish limited evidence only. Review the close-ups for actual cavity, lip/jaw/eye attachment, expression readability, intersections and neutral fidelity. Report visual uncertainty separately; numerical passes are not visual approval.']
    if include_binding:
        lines += ['BINDING GRAMMAR / STARTER (replace ALL Replace_* node names and measure pivots, scale, target and preview for this character; zero pivots below are placeholders, not landmarks):',
              json.dumps(binding_template(c),indent=2),
              'Coordinates: Blender metres, +Y forward, +Z up, left X<0. A region can select part of a shared named mesh using box:[[xmin,ymin,zmin],[xmax,ymax,zmax]], or center:[x,y,z], radius:[rx,ry,rz], inner:0.5; do not combine box and ellipsoid fields. Neck region covers the head and all facial parts. Other regions select only their actual semantic part. Optional neck ramp:[axis_index,start,end] blends from body to head. Optional jaw_open_degrees is in (0,45], default 28. No need to declare mouth_topology: it cannot satisfy the functional checks.']
    return '\n'.join(lines)


def _binding(path):
    b = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(b, dict) or type(b.get('version')) is not int or b['version'] != 1 or set(b)-{'version', 'rig', 'semantics', 'script_sha256', 'requirements_sha256', 'source_sha256'}:
        raise ValueError('face_bindings.json needs version 1 and known fields')
    face.contract(b.get('rig'))
    if not isinstance(b.get('semantics'), dict) or set(b['semantics'])-{'eyes', 'mouth'}:
        raise ValueError('face bindings need an eyes/mouth semantics object')
    for key in ('script_sha256', 'requirements_sha256', 'source_sha256'):
        if key in b and (not isinstance(b[key], str) or not DIGEST.fullmatch(b[key])):
            raise ValueError(f'{key} must be a SHA-256 hex digest')
    return b


def seal(proj, attempt):
    c = contract(proj.data.get('face_capabilities'))
    attempt = Path(attempt).resolve()
    if not attempt.is_relative_to(proj.attempts_dir.resolve()) or attempt.parent != proj.attempts_dir.resolve():
        raise ValueError('face-bind needs a direct attempt directory in this project')
    path = attempt/FILE
    script = attempt/glb.script_name(proj.spec)
    if path.is_symlink() or script.is_symlink() or path.resolve().parent != attempt or script.resolve().parent != attempt:
        raise ValueError('face bindings and script must stay inside the attempt')
    b = _binding(path)
    b.update(script_sha256=file_digest(script), requirements_sha256=digest(c))
    path.write_text(json.dumps(b, indent=2)+'\n', encoding='utf-8')
    return 'Facial bindings sealed to current script and requirements. Geometry and visual claims still need evaluation.'


def _region(region, node_ids, points):
    # Reuse the rig's checked region grammar without accepting unknown selectors.
    config = {'version':1, 'face_scale_m':1, 'look_target':[0,1,0],
              'controls':{'neck':{'pivot':[0,0,0], 'region':region}}}
    face.contract(config)
    if set(region['nodes'])-set(node_ids):
        raise ValueError('semantic region names absent nodes: '+', '.join(sorted(set(region['nodes'])-set(node_ids))))
    selected = np.zeros(points.shape[:2], dtype=bool)
    for name, indices in node_ids.items():
        indices = np.asarray(indices, dtype=int)
        selected[indices] = face_motion.selection(region, name, points[indices].reshape(-1,3)).reshape(-1,3) > .05
    if not selected.any():
        raise ValueError('semantic region selects no exported surface vertices')
    return selected


def _first_hit(tris, origin, direction):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    e1, e2 = tris[:,1]-tris[:,0], tris[:,2]-tris[:,0]
    h = np.cross(d, e2)
    det = np.einsum('ij,ij->i', e1, h)
    inv = np.divide(1., det, out=np.zeros_like(det), where=np.abs(det)>1e-12)
    s = np.asarray(origin)-tris[:,0]
    u = inv*np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = inv*(q@d)
    t = inv*np.einsum('ij,ij->i', e2, q)
    valid = (np.abs(det)>1e-12)&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(t>1e-8)
    distances = np.where(valid,t,np.inf)
    k = int(distances.argmin())
    return (float(distances[k]), k) if np.isfinite(distances[k]) else (None, None)


def mouth_evidence(points, upper, lower, interior, requirements):
    """Nine front rays through an opening derived from the actual lip regions.

    Double-sided occlusion includes ALL geometry. A tagged interior hidden behind
    a closed front skin cannot satisfy the sampled first-hit/depth checks.
    """
    u, l = points[upper].mean(axis=0), points[lower].mean(axis=0)
    gap = float(u[2]-l[2])
    centre = (u+l)/2
    lip = np.concatenate((points[upper], points[lower]))
    width = float(np.ptp(lip[:,0]))
    front = float(lip[:,1].max())
    far = float(np.ptp(points.reshape(-1,3),axis=0).max()+1)
    hits, depths = [], []
    inside = interior.any(axis=1)
    for x in (-.2,0,.2):
        for z in (-.2,0,.2):
            origin = [centre[0]+x*width, front+far, centre[2]+z*max(gap,0)]
            distance, index = _first_hit(points, origin, [0,-1,0])
            depth = distance-far if distance is not None else None
            depths.append(depth)
            hits.append(index is not None and bool(inside[index]) and depth >= requirements['min_interior_depth_m'])
    return {'gap_m':gap, 'min_gap_m':requirements['min_mouth_gap_m'],
            'interior_first_hits':int(sum(hits)), 'rays':9, 'depths_m':depths,
            'min_depth_m':requirements['min_interior_depth_m'],
            'passed':gap>=requirements['min_mouth_gap_m'] and width>1e-6 and sum(hits)==9}


def _test_script(config):
    code = face.script_text(config).rsplit('def pose(clip, t):',1)[0]
    return code+'''def pose(clip, t):
    if clip == 'mouth_open':
        return {'face_jaw': {'rotate': [('X', -CONFIG.get('jaw_open_degrees', 28))]}} if 'jaw' in CONFIG['controls'] else {}
    if clip.startswith('gaze_'):
        offset = {'gaze_left': [-3,3,0], 'gaze_right': [3,3,0], 'gaze_up': [0,3,3], 'gaze_down': [0,3,-3]}[clip]
        target = np.asarray(CONFIG['controls']['neck']['pivot'])+np.asarray(offset)*CONFIG['face_scale_m']
        return pose_at(CONFIG, 'look_at', target=target)
    return pose_at(CONFIG, clip, .25)
'''


def feedback(report):
    if not report:
        return ''
    lines = ['FACIAL FUNCTION (separate from resemblance; sampled evidence, not visual approval):']
    lines += report.get('errors', [])
    for state, r in report.get('capabilities', {}).items():
        lines.append(f'{state} [{r["requirement"]}]: {r["status"]}. '+(' '.join(r['issues']) or 'Inspect close-ups; visual approval remains pending.'))
    if report.get('mouth'):
        m=report['mouth']
        lines.append(f'Open mouth: lip gap {m["gap_m"]*1000:.2f} mm; interior is first surface on {m["interior_first_hits"]}/9 sampled rays. Require gap {m["min_gap_m"]*1000:g} mm, depth {m["min_depth_m"]*1000:g} mm.')
    lines += ['Preserve the neutral reference match. Fix bindings/geometry rather than changing requirements or labelling a closed surface open_mouth.',
              'Review neutral, open mouth, gaze extremes and expressions for attachment, intersections, cavity and readability; sampled passes cannot certify them.']
    return '\n'.join(lines)


def evaluate(proj, script, source, attempt, *, output=None):
    if proj.data.get('face_capabilities') is None:
        return None, {}
    c = contract(proj.data['face_capabilities'])
    report = {'version':1, 'requirements':c, 'errors':[], 'capabilities':{}, 'visual_review':'pending'}
    for state in requested(c):
        report['capabilities'][state]={'requirement':'required' if state in c['required'] else 'optional', 'status':'unavailable', 'issues':[]}
    pictures = {}
    try:
        attempt, source, script = Path(attempt), Path(source), Path(script)
        path = attempt/FILE
        if not path.is_file():
            raise ValueError('Missing face_bindings.json. Author semantic regions, then run face-bind / bind_face after final script edits.')
        if path.is_symlink() or path.resolve().parent != attempt.resolve():
            raise ValueError('Facial bindings must be a regular file inside the attempt')
        b = _binding(path)
        if b.get('script_sha256') != file_digest(script) or b.get('requirements_sha256') != digest(c):
            raise ValueError('Stale or unsealed facial bindings: review against current model and requirements, then call face-bind / bind_face.')
        if b.get('source_sha256') is not None and b['source_sha256'] != file_digest(source):
            raise ValueError('Facial bindings source_sha256 does not match the exported GLB')
        report.update(source_sha256=file_digest(source), script_sha256=file_digest(script), bindings_sha256=file_digest(path))
        config=face.contract(b['rig'])
        # This stage's evidence replaces declaration-based capability claims.
        config['required']=[]
        inspected=face.inspect(source,config)
        report['controls']=inspected['controls']
        model,_=rig.model_data(*glb.read_glb(source))
        neutral=rig_look.to_blender(render.load(source).tris)
        node_ids=review.node_triangles(source)
        masks={}
        semantic_errors={}
        for kind, roles in [('mouth',('upper_lip','lower_lip','interior'))]:
            if set(requested(c)) & {'talk','yell'}:
                for role in roles:
                    key=kind+'.'+role
                    try:
                        masks[key]=_region(b['semantics'].get(kind,{}).get(role),node_ids,neutral)
                    except (ValueError,TypeError,KeyError,AttributeError) as e:
                        semantic_errors[key]=f'{key}: bind an existing nonempty surface region ({e}).'
        if 'look_at' in requested(c):
            for eye in c['eyes']:
                for role in ('surface','iris','pupil'):
                    key='eyes.'+eye+'.'+role
                    try:
                        masks[key]=_region(b['semantics'].get('eyes',{}).get(eye,{}).get(role),node_ids,neutral)
                    except (ValueError,TypeError,KeyError,AttributeError) as e:
                        semantic_errors[key]=f'{key}: bind actual visible eye/iris/pupil geometry ({e}).'
        clips=[]
        if set(requested(c)) & {'talk','yell'}:
            clips.append('mouth_open')
        if 'look_at' in requested(c):
            clips += ['gaze_left','gaze_right','gaze_up','gaze_down']
        clips += [s for s in requested(c) if s != 'look_at']
        output=Path(output) if output is not None else attempt/'face-test'
        output.mkdir(parents=True, exist_ok=True)
        test_script=output/'test_rig.py'
        test_script.write_text(_test_script(config),encoding='utf-8')
        test_rig=output/'test.glb'
        rs={'fps':1, 'max_influences':4, 'clips':{s:{'seconds':1,'loop':True} for s in clips}, 'bones':{'max':9}}
        rig.build(source,test_script,rs,test_rig)
        posed={s:rig_look.to_blender(skin.posed(test_rig,s,.5).tris) for s in clips}
        doc,blob=glb.read_glb(test_rig)
        neck=next(i for i,n in enumerate(doc['nodes']) if n.get('name')=='face_neck')
        # Poses relative to the moved neck distinguish eye/local expression motion from head motion.
        local={}
        rest_world=skin.worlds(doc)[neck]
        for state, points in posed.items():
            inverse=np.linalg.inv(skin.sample(doc,blob,state,.5)[neck])
            transform=rest_world@inverse
            p=skin.posed(test_rig,state,.5).tris
            local[state]=rig_look.to_blender(p@transform[:3,:3].T+transform[:3,3])
        if all('mouth.'+r in masks for r in ('upper_lip','lower_lip','interior')):
            upper,lower,interior=[masks['mouth.'+r] for r in ('upper_lip','lower_lip','interior')]
            if np.any(upper & lower) or np.any(interior & (upper|lower)):
                semantic_errors['mouth.overlap']='Upper lip, lower lip and interior selections overlap; bind distinct functional surfaces.'
            report['mouth']=mouth_evidence(posed['mouth_open'],upper,lower,interior,c)
            report['mouth']['lower_lip_motion_m']=float(np.linalg.norm(local['mouth_open'][lower]-neutral[lower],axis=1).max())
            rest_gap=float(neutral[upper].mean(axis=0)[2]-neutral[lower].mean(axis=0)[2])
            report['mouth']['opening_increase_m']=report['mouth']['gap_m']-rest_gap
            if report['mouth']['lower_lip_motion_m']<1e-5 or report['mouth']['opening_increase_m']<1e-5:
                semantic_errors['mouth.motion']='The jaw test must move the lower lip and increase the opening; a static hole or unrelated jaw weights are insufficient.'
        for state, item in report['capabilities'].items():
            missing=needs(state,c['eyes'])-set(config['controls'])
            item['issues'] += ['Missing controls: '+', '.join(sorted(missing))+'.'] if missing else []
            if state in ('talk','yell'):
                item['issues'] += [v for k,v in semantic_errors.items() if k.startswith('mouth.')]
                if not report.get('mouth',{}).get('passed'):
                    item['issues'].append('Open-pose cavity evidence failed: separate the lips and provide visible interior behind the aperture; a solid stretched chin does not satisfy speech/yell.')
                if 'mouth.lower_lip' in masks:
                    motion=float(np.linalg.norm(local[state][masks['mouth.lower_lip']]-neutral[masks['mouth.lower_lip']],axis=1).max())
                    if motion<1e-5:
                        item['issues'].append('Lower lip has no local motion in the requested '+state+' pose.')
                if state=='yell' and all('mouth.'+r in masks for r in ('upper_lip','lower_lip','interior')):
                    report['yell_mouth']=mouth_evidence(local[state],upper,lower,interior,c)
                    if not report['yell_mouth']['passed']:
                        item['issues'].append('The actual yell pose does not expose the required sampled opening/interior.')
            if state=='look_at':
                item['issues'] += [v for k,v in semantic_errors.items() if k.startswith('eyes.')]
                for eye in c['eyes']:
                    roles=['eyes.'+eye+'.'+role for role in ('surface','iris','pupil')]
                    for i,key in enumerate(roles):
                        for other in roles[i+1:]:
                            if key in masks and other in masks and np.array_equal(masks[key],masks[other]):
                                item['issues'].append(key+' and '+other+': identical selections do not establish distinct eye/iris/pupil geometry.')
                    for role in ('surface','iris','pupil'):
                        key='eyes.'+eye+'.'+role
                        if key not in masks:
                            continue
                        mask=masks[key]
                        moves=[float(np.linalg.norm(local[s][mask]-neutral[mask],axis=1).max()) for s in clips if s.startswith('gaze_')]
                        if not moves or min(moves)<1e-5:
                            item['issues'].append(key+': no independent eye motion in every tested gaze direction; bind it to the eye, not only the neck.')
                        # Front visibility at neutral: nearest surface must include at least one sampled semantic vertex.
                        pts=neutral[mask]
                        visible=0
                        far=float(np.ptp(neutral.reshape(-1,3),axis=0).max()+1)
                        for p in pts[np.linspace(0,len(pts)-1,min(9,len(pts)),dtype=int)]:
                            d,_=_first_hit(neutral,p+[0,far,0],[0,-1,0])
                            visible += d is not None and abs(d-far)<1e-4
                        if not visible:
                            item['issues'].append(key+': no front-visible sampled surface; inspect occlusion and eye/iris/pupil placement.')
            if state != 'look_at':
                for control in needs(state,c['eyes'])-{'neck','jaw'}-missing:
                    if state=='talk':
                        continue  # Speech requires lip-corner bindings, but its default motion is jaw/head only.
                    selected=_region(config['controls'][control]['region'],node_ids,neutral)
                    motion=float(np.linalg.norm(local[state][selected]-neutral[selected],axis=1).max())
                    if motion<1e-5:
                        item['issues'].append(control+': no local deformation observed; head motion alone is not an expression.')
            item['status']='failed' if item['issues'] else 'sampled_evidence'
        view=config.get('preview')
        if not view:
            report['errors'].append('No rig.preview aim/span: whole-model pictures cannot replace authored face close-ups.')
        for name, state in [('neutral',None)]+[(s,s) for s in clips]:
            shot=rig_look.shot('front',model['lo'],model['hi'],**(view or {}))
            image=rig_look.draw(skin.posed(test_rig,state,.5) if state else render.load(source,textured=bool(doc.get('images')),multiply=True),shot,320)
            filename=output/(name+'.png')
            image.save(filename)
            pictures[name]=str(filename)
    except (ValueError,OSError,KeyError,TypeError,AttributeError) as error:
        # Missing/invalid optional bindings are feedback, not a loss of the static model evaluation.
        report['errors'].append(str(error).replace(str(Path(attempt).resolve()),'<attempt>'))
        for item in report['capabilities'].values():
            item['status']='failed'
            item['issues'].append('Functional test unavailable; repair the binding/export error above.')
    report['required_checks_passed']=not report['errors'] and all(v['status']=='sampled_evidence' for v in report['capabilities'].values() if v['requirement']=='required')
    # With optional-only requirements, unavailable tests remain observations rather than acceptance failures.
    report['required_checks_passed'] = report['required_checks_passed'] or not c['required']
    return report,pictures


def cli_bind(a):
    from . import project
    p=project.load(a.project)
    attempt=a.attempt if a.attempt.is_dir() else p.attempts_dir/a.attempt
    try:
        print(seal(p,attempt))
    except OSError as error:
        raise ValueError('Cannot seal facial bindings: '+str(error)) from error
    return 0
