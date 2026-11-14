"""Actual GLB composition and simultaneous playback; no model/API calls."""
import copy
import io
import json
from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest

from mesh_jig import character, cli, face, face_capability, glb, metrics, rig, skin
from test_face_capability import example

sys.path.insert(0,str(Path(__file__).parents[1]/'examples/face-capability'))
try:
    import compose_example
finally:
    sys.path.pop(0)


@pytest.fixture
def assets(tmp_path):
    source=example.write_source(tmp_path/'source.glb',with_body=True)
    script=tmp_path/'body.py'
    script.write_text(compose_example.BODY_SCRIPT)
    body=tmp_path/'body.glb'
    rig.build(source,script,compose_example.BODY_CONTRACT,body)
    config={'version':1,'anchor':'head','neck':'additive','fps':12,
            'demos':[{'name':'walk_talk','body':'walk','expression':'talk'},
                     {'name':'walk_gaze','body':'walk','expression':'yell','gaze_target':[.3,4,.7]}]}
    return source,body,example.binding()['rig'],config,tmp_path/'combined.glb'


def build(args):
    return character.compose(*args)


def rewrite(path,change):
    doc,binary=glb.read_glb(path)
    change(doc)
    rig.write_glb(path,doc,bytearray(binary))


def test_export_preserves_source_body_channels_and_neutral_playback(assets):
    source,body,cfg,config,out=assets
    report=build(assets)
    old,ob=glb.read_glb(body);doc,blob=glb.read_glb(out)
    assert doc['animations'][0]==old['animations'][0]
    assert bytes(blob[:len(ob)])==bytes(ob)
    assert report['max_influences']<=4
    assert report['body_replay_error_m']<1e-6 and report['rest_error_m']<1e-6
    for key in ('materials','images','textures','samplers'):
        assert doc.get(key)==old.get(key)
    np.testing.assert_allclose(skin.posed(source).tris,skin.posed(out).tris,atol=1e-6)
    for t in (0,.123,.641,1.1,2):
        np.testing.assert_allclose(skin.posed(body,'walk',t).tris,skin.posed(out,'walk',t).tris,atol=1e-6)
    body_targets={c['target']['node'] for c in doc['animations'][0]['channels']}
    for a in doc['animations'][1:8]:
        assert not body_targets & {c['target']['node'] for c in a['channels']}


def test_live_and_exported_layers_agree_and_repeated_switches_reset(assets):
    build(assets);player=character.Player(assets[-1]);doc,blob=player.doc,player.blob
    for state,t in [('yell',.5),('talk',.375),('smile',1.),('sadness',.5),(None,.25),('talk',.25)]:
        live=player.mesh('walk',t,expression=state)
        layers=[('walk',t)]+([('face.'+state,t)] if state else [])
        exported=skin.posed(assets[-1],world=skin.sample_layers(doc,blob,layers))
        np.testing.assert_allclose(live.tris,exported.tris,atol=1e-6)
    np.testing.assert_allclose(player.mesh().tris,skin.posed(assets[1]).tris,atol=1e-6)
    with pytest.raises(ValueError,match='overlapping'):
        skin.sample_layers(doc,blob,[('walk',.2),('combined.walk_talk',.2)])
    with pytest.raises(ValueError,match='overlapping'):
        skin.sample_layers(doc,blob,[('face.talk',.2),('face.smile',.2)])
    with pytest.raises(ValueError,match='original body'):
        player.worlds('combined.walk_talk',0)


def test_baked_and_live_simultaneous_poses_match_at_export_keys(assets):
    build(assets);player=character.Player(assets[-1])
    for name,state,gaze in [('walk_talk','talk',None),('walk_gaze','yell',[.3,4,.7])]:
        for t in (0,.25,.75,1.5,2):
            np.testing.assert_allclose(player.mesh('walk',t,expression=state,gaze_target=gaze).tris,
                                       skin.posed(assets[-1],'combined.'+name,t).tris,atol=2e-6)


def test_gaze_tracks_fixed_point_after_moving_rotating_body_head(assets):
    build(assets);p=character.Player(assets[-1]);target=np.array([.3,4,.7]);target_gltf=rig.to_gltf(target)
    eye=p.meta['face_nodes']['face_eye_r']
    for t in (.13,.5,1.25,1.9):
        world=p.worlds('walk',t,expression='talk',gaze_target=target)
        direction=world[eye][:3,:3]@np.array([0,0,-1])
        wanted=target_gltf-world[eye][:3,3];wanted/=np.linalg.norm(wanted)
        np.testing.assert_allclose(direction,wanted,atol=1e-6)
        # Every body joint is sampled exactly once, unchanged by face expression or gaze.
        body=skin.sample(p.doc,p.blob,'walk',t)
        for skin_spec in glb.read_glb(assets[1])[0]['skins']:
            for j in skin_spec['joints']:
                np.testing.assert_allclose(world[j],body[j],atol=1e-7)


def test_body_neck_ownership_and_open_mouth_under_body_motion(assets):
    assets[3]['neck']='body';build(assets);p=character.Player(assets[-1])
    neutral=p.worlds('walk',.5)
    posed=p.worlds('walk',.5,expression='yell',gaze_target=[.2,5,.7])
    neck=p.meta['face_nodes']['face_neck']
    np.testing.assert_allclose(posed[neck],neutral[neck],atol=1e-7)
    # Undo current body anchor deformation before measuring the actual opened mouth.
    anchor=p.meta['anchor_node']
    inverse=np.linalg.inv(neutral[anchor]@np.linalg.inv(np.array(p.meta['anchor_rest'])))
    mesh=p.mesh('walk',.5,expression='yell')
    points=character.rig_look.to_blender(mesh.tris@inverse[:3,:3].T+inverse[:3,3])
    rest=character.rig_look.to_blender(skin.posed(assets[0]).tris)
    from mesh_jig import review
    ids=review.node_triangles(assets[0]);semantics=example.binding()['semantics']['mouth']
    masks=[face_capability._region(semantics[k],ids,rest) for k in ('upper_lip','lower_lip','interior')]
    evidence=face_capability.mouth_evidence(points,*masks,face_capability.contract({'version':1,'required':['yell']}))
    assert evidence['passed'] and evidence['interior_first_hits']==9


@pytest.mark.parametrize('change',[
    {},{'version':2,'anchor':'head','neck':'body'},
    {'version':1,'anchor':'head','neck':'automatic'},
    {'version':1,'anchor':'head','neck':'body','fps':0},
    {'version':1,'anchor':'head','neck':'body','demos':[{'name':'../x','body':'walk'}]},
])
def test_bad_contract(change):
    with pytest.raises(ValueError):
        character.contract(change)


@pytest.mark.parametrize('kind',['wrong_source','bad_bind','joint_frame','namespace','overlap','scale_channel','morph','two_weight_sets','already_composed','missing_anchor'])
def test_incompatible_sources_refused_without_replacing_delivery(assets,kind):
    source,body,cfg,c,out=assets
    out.write_bytes(b'prior delivery')
    if kind=='wrong_source':
        rewrite(source,lambda d:d['nodes'][0].update(translation=[.2,0,0]))
    elif kind=='namespace':
        rewrite(body,lambda d:next(n for n in d['nodes'] if n.get('name')=='head').update(name='face_neck'))
    elif kind=='joint_frame':
        rewrite(body,lambda d:next(n for n in d['nodes'] if n.get('name')=='head').update(rotation=[0,0,.1,np.sqrt(.99)]))
    elif kind=='overlap':
        rewrite(body,lambda d:d['animations'][0]['channels'].append(copy.deepcopy(d['animations'][0]['channels'][0])))
    elif kind=='scale_channel':
        rewrite(body,lambda d:d['animations'][0]['channels'][0]['target'].update(path='scale'))
    elif kind=='already_composed':
        rewrite(body,lambda d:d.setdefault('extras',{}).update({character.META:{'version':1}}))
    elif kind=='missing_anchor':
        c['anchor']='absent'
    elif kind=='bad_bind':
        rewrite(body,lambda d:d['skins'][0].pop('inverseBindMatrices'))
    else:
        def edit(d):
            p=d['meshes'][d['nodes'][0]['mesh']]['primitives'][0]
            if kind=='morph': p['targets']=[{'POSITION':p['attributes']['POSITION']}]
            else: p['attributes']['JOINTS_1']=p['attributes']['JOINTS_0']
        rewrite(body,edit)
    with pytest.raises(ValueError):
        build(assets)
    assert out.read_bytes()==b'prior delivery'


def test_overlap_weights_are_refused_instead_of_pruned(assets):
    source,body,cfg,c,out=assets
    script=body.parent/'blend.py'
    code=compose_example.BODY_SCRIPT.replace('import math','import math\nimport numpy as np')
    code=code.replace("return name if name in ('body','leg_l','leg_r') else 'head'",
                      "return {n:np.full(len(position),.25) for n in ('head','body','root','leg_l')}")
    script.write_text(code);rig.build(source,script,compose_example.BODY_CONTRACT,body)
    cfg['controls']['mouth_l']['region']['nodes'].append('lower_lip')
    with pytest.raises(ValueError,match='influence limit'):
        build(assets)


def test_missing_body_attachment_is_not_silently_invented(assets):
    source,body,cfg,c,out=assets
    script=body.parent/'root_only.py'
    script.write_text(compose_example.BODY_SCRIPT.replace("return name if name in ('body','leg_l','leg_r') else 'head'","return 'root'"))
    rig.build(source,script,compose_example.BODY_CONTRACT,body)
    with pytest.raises(ValueError,match='no anchor weight'):
        build(assets)


def test_transformed_source_mesh_and_mixed_head_weights_preserve_body(assets):
    source,body,cfg,c,out=assets
    rewrite(source,lambda d:d['nodes'][0].update(translation=[.015,0,0],scale=[1.1,.9,1.05]))
    script=body.parent/'mixed.py'
    code=compose_example.BODY_SCRIPT.replace('import math','import math\nimport numpy as np')
    code=code.replace("return name if name in ('body','leg_l','leg_r') else 'head'",
                      "return name if name in ('body','leg_l','leg_r') else {'head':np.full(len(position),.8),'body':np.full(len(position),.2)}")
    script.write_text(code);rig.build(source,script,compose_example.BODY_CONTRACT,body)
    report=build(assets)
    assert report['max_influences']==2 and report['body_replay_error_m']<1e-6


def test_embedded_texture_uv_material_and_colours_survive_composition(assets):
    from PIL import Image
    from glbkit import textured_glb
    source,body,_,c,out=assets
    image=io.BytesIO();Image.new('RGB',(2,2),(17,123,201)).save(image,format='PNG')
    textured_glb(source,png=image.getvalue())
    rig.build(source,body.parent/'body.py',compose_example.BODY_CONTRACT,body)
    cfg={'version':1,'face_scale_m':.08,'look_target':[0,2,0],
         'controls':{'neck':{'pivot':[0,0,0],'region':{'nodes':['K']}}}}
    character.compose(source,body,cfg,c,out)
    before=skin.posed(source);after=character.Player(out).mesh('walk',.5,expression='talk')
    np.testing.assert_array_equal(before.colors,after.colors)
    np.testing.assert_array_equal(before.uvs,after.uvs)
    np.testing.assert_array_equal(before.textures[0],after.textures[0])
    assert np.isfinite(after.normals).all()
    # A mismatched embedded image is refused even when metadata still matches.
    doc,binary=glb.read_glb(body);blob=bytearray(binary)
    view=doc['bufferViews'][doc['images'][0]['bufferView']]
    blob[view['byteOffset']+20]^=1
    rig.write_glb(body,doc,blob)
    with pytest.raises(ValueError,match='image bytes'):
        character.compose(source,body,cfg,c,out)


def test_monocular_contract_survives_builder_and_separate_face_export(assets):
    c=face_capability.contract({'version':1,'required':['look_at'],'eyes':['right']})
    assert face_capability.binding_template(c)['rig']['visible_eyes']==['right']
    cfg=assets[2];cfg['required']=['look_at']
    assert not face.inspect(assets[0],cfg)['issues']
    assert not face.build(assets[0],cfg,assets[-1])['issues']


def test_cli_export_and_preview(assets):
    source,body,cfg,c,out=assets
    f=out.parent/'face.json';f.write_text(json.dumps(cfg))
    contract=out.parent/'compose.json';contract.write_text(json.dumps(c))
    ledger=out.parent/'metrics'
    assert cli.main(['--metrics-dir',str(ledger),'--metrics-stage','facial','--metrics-run','puppet',
                     'character-compose','--source',str(source),'--body',str(body),'--face',str(f),'--contract',str(contract),'--out',str(out)])==0
    row=metrics.summarize(metrics.read(ledger))['stages'][0]
    assert row['run']=='puppet' and row['stage']=='facial'
    assert row['counts']['command']['completed']==1 and row['token_coverage']=='unavailable'
    assert cli.main(['character-preview','--rig',str(out),'--body','walk','--expression','talk','--gaze','.3','4','.7','--out',str(out.parent/'preview'),'--seconds','.5','--fps','4'])==0
    assert (out.parent/'preview'/'playback.gif').is_file()


@pytest.mark.blender
def test_blender_import_baked_and_nla_layers_agree_with_cpu_after_switches(assets,blender):
    build(assets)
    out=assets[-1];doc,blob=glb.read_glb(out)
    cases=[{'clip':'combined.walk_gaze','t':.5},
           {'layers':['walk','face.yell'],'t':.5},
           {'layers':['walk','face.talk'],'t':.375},
           {'clip':'walk','t':.25},
           {'layers':[],'t':0},
           {'layers':['walk','face.smile'],'t':1.125}]
    request=out.parent/'blender-request.json'
    request.write_text(json.dumps({'glb':str(out),'cases':cases}),encoding='utf-8')
    script=Path(__file__).with_name('character_blender_probe.py')
    result=subprocess.run([blender,'--background','--factory-startup','--python',str(script),'--',str(request)],
                          capture_output=True,text=True,timeout=120)
    assert result.returncode==0 and 'CHARACTER_BLENDER_PLAYBACK_OK' in result.stdout,result.stdout+result.stderr
    for index,case in enumerate(cases):
        world=(skin.sample(doc,blob,case['clip'],case['t']) if case.get('clip') else
               skin.sample_layers(doc,blob,[(n,case['t']) for n in case['layers']]))
        expected=character.rig_look.to_blender(skin.posed(out,world=world).tris.reshape(-1,3))
        actual=np.load(out.parent/f'blender-{index:02d}.npy')
        for a,b in ((actual,expected),(expected,actual)):
            # Importer welds/reorders vertices. Bidirectional point-set comparison.
            for start in range(0,len(a),128):
                error=np.sqrt(((a[start:start+128,None]-b[None])**2).sum(2).min(1)).max()
                assert error<2e-5,(case,float(error))
