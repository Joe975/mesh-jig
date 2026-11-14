"""Early contract propagation and geometric counterexamples, with no model/API calls."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from mesh_jig import agent, brief, cli, comparison_context, evaluate, face_capability as f, gauntlet, glb, project, sheet, skin

EXAMPLE = Path(__file__).parents[1]/'examples'/'face-capability'/'make_example.py'
loader = importlib.util.spec_from_file_location('face_example',EXAMPLE)
example = importlib.util.module_from_spec(loader)
loader.loader.exec_module(example)


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    return example.make(tmp_path_factory.mktemp('facial')/'open')


@pytest.fixture
def setup(generated,tmp_path):
    p,_,_,_ = generated
    root = tmp_path/'project'
    shutil.copytree(p.root,root)
    p = project.load(root)
    return p,p.attempts_dir/'a01'


def run(p,a):
    return f.evaluate(p,a/'model.py',a/'out'/'puppet.glb',a)


def edit_binding(a, change):
    path=a/f.FILE
    b=json.loads(path.read_text())
    change(b)
    path.write_text(json.dumps(b))


@pytest.mark.parametrize('value', [None,{},[],{'version':True,'required':['talk']},
    {'version':1,'required':['speech']},{'version':1,'required':['talk','talk']},
    {'version':1,'required':['talk'],'optional':['talk']},{'version':1,'required':['look_at']},
    {'version':1,'required':['look_at'],'eyes':['left','left']},
    {'version':1,'required':['talk'],'eyes':['right']},
    {'version':1,'required':['talk'],'min_mouth_gap_m':float('nan')},
    {'version':1,'required':['talk'],'min_interior_depth_m':0},
    {'version':1,'required':['talk'],'surprise':True}])
def test_schema(value):
    with pytest.raises(ValueError):
        f.contract(value)


def test_setup_flags_and_noop_default(tmp_path):
    plain=tmp_path/'plain'
    assert cli.main(['init',str(plain)])==0
    assert 'face_capabilities' not in project.load(plain).data
    flagged=tmp_path/'flagged'
    assert cli.main(['init',str(flagged),'--height','1.2','--face-required','look_at','talk',
                     '--face-optional','yell','--face-eyes','right'])==0
    c=project.load(flagged).data['face_capabilities']
    assert c['eyes']==['right'] and c['optional']==['yell']
    invalid=tmp_path/'invalid'
    with pytest.raises(ValueError):
        sheet.init(invalid,face_capabilities={'version':1,'required':['look_at']})
    assert not invalid.exists()


def test_both_briefs_project_validation_and_comparison_context(setup):
    p,a=setup
    assert not p.problems()
    for text in (brief.text(p),gauntlet.visual_brief(p)):
        assert 'FACIAL CAPABILITIES' in text
        assert 'visible right eye' in text
        assert 'real opening' in text and 'face_bindings.json' in text
        assert '"face_scale_m"' in text and '"semantics"' in text and 'BINDING GRAMMAR' in text
    before=comparison_context.signature(p)
    p.data['face_capabilities']['required']=['smile']
    p.data['face_capabilities'].pop('eyes')
    assert before!=comparison_context.signature(p)
    p.data['face_capabilities']['required']=['not-a-state']
    assert any('face_capabilities.required' in issue for issue in p.problems())


def test_real_opening_export_and_monocular_gaze(generated):
    p,a,report,pictures=generated
    assert report['errors']==[]
    assert report['required_checks_passed'],f.feedback(report)
    assert report['visual_review']=='pending'
    assert report['mouth']['interior_first_hits']==9
    assert report['mouth']['opening_increase_m']>.05
    assert report['yell_mouth']['passed']
    assert set(pictures)=={'neutral','mouth_open','gaze_left','gaze_right','gaze_up','gaze_down',
                           'talk','smile','anger','sadness','confusion','yell'}
    assert all(Image.open(v).size==(320,320) for v in pictures.values())
    original=skin.posed(a/'out'/'puppet.glb').tris
    exported=skin.posed(a/'face-test'/'test.glb').tris
    np.testing.assert_allclose(original,exported,atol=1e-6)
    assert not np.allclose(original,skin.posed(a/'face-test'/'test.glb','yell',.5).tris)
    assert 'eye_l' not in report['controls']
    assert report['capabilities']['look_at']['status']=='sampled_evidence'


def test_closed_surface_declaration_and_weights_do_not_prove_cavity(tmp_path):
    _,a,r,_=example.make(tmp_path/'blocked',True)
    assert json.loads((a/f.FILE).read_text())['rig']['mouth_topology']=='open_mouth'
    assert r['mouth']['lower_lip_motion_m']>.05
    assert r['mouth']['interior_first_hits']==0
    assert not r['required_checks_passed']
    assert r['capabilities']['talk']['status']==r['capabilities']['yell']['status']=='failed'
    assert r['capabilities']['look_at']['status']=='sampled_evidence'


@pytest.mark.parametrize('kind',['missing','stale_script','stale_contract','wrong_source','bad_nodes','empty_region','overlap','iris_on_head','lip_on_head','duplicate_eye_regions'])
def test_bad_semantic_bindings(setup,kind):
    p,a=setup
    if kind=='missing':
        (a/f.FILE).unlink()
    elif kind=='stale_script':
        with (a/'model.py').open('a') as stream:
            stream.write('\n# changed model\n')
    elif kind=='stale_contract':
        p.data['face_capabilities']['min_mouth_gap_m']=.04
    elif kind=='wrong_source':
        edit_binding(a,lambda b:b.update(source_sha256='0'*64))
    elif kind in ('bad_nodes','empty_region','overlap'):
        def change(b):
            m=b['semantics']['mouth']
            if kind=='bad_nodes':
                m['interior']={'nodes':['nonexistent']}
            elif kind=='empty_region':
                m['interior']={'nodes':['interior'],'box':[[10,10,10],[11,11,11]]}
            else:
                m['interior']=m['upper_lip']
        edit_binding(a,change)
    elif kind=='iris_on_head':
        edit_binding(a,lambda b:b['rig']['controls']['eye_r']['region'].update(nodes=['eye_r','pupil_r']))
    elif kind=='duplicate_eye_regions':
        edit_binding(a,lambda b:b['semantics']['eyes']['right'].update(iris={'nodes':['eye_r']},pupil={'nodes':['eye_r']}))
    else:
        edit_binding(a,lambda b:b['rig']['controls']['jaw']['region'].update(nodes=['chin']))
    r,_=run(p,a)
    assert not r['required_checks_passed'],f.feedback(r)
    assert r['visual_review']=='pending'
    text=f.feedback(r)
    assert ('Stale' in text if kind.startswith('stale') else 'failed' in text)
    if kind=='iris_on_head':
        assert 'no independent eye motion' in text
    if kind=='lip_on_head':
        assert 'static hole' in text


def test_optional_is_feedback_not_required_failure(setup):
    p,a=setup
    p.data['face_capabilities']={'version':1,'required':['smile'],'optional':['look_at'],'eyes':['left','right']}
    f.seal(p,a)
    r,_=run(p,a)
    assert r['required_checks_passed']
    assert r['capabilities']['look_at']['status']=='failed'
    assert r['capabilities']['smile']['status']=='sampled_evidence'
    p.data['face_capabilities']={'version':1,'optional':['yell']}
    (a/f.FILE).unlink()
    r,_=run(p,a)
    assert r['required_checks_passed'] and r['errors']


def test_unflagged_does_not_read_bindings_or_export(setup,monkeypatch):
    p,a=setup
    del p.data['face_capabilities']
    def forbidden(*args,**kwargs):
        raise AssertionError('should not inspect face data')
    monkeypatch.setattr(f,'_binding',forbidden)
    assert f.evaluate(p,'missing.py','missing.glb',a)==(None,{})
    assert 'FACIAL CAPABILITIES' not in brief.text(p)
    assert 'FACIAL CAPABILITIES' not in gauntlet.visual_brief(p)
    files=gauntlet.Files(p,a)
    assert f.FILE not in files.names
    assert 'bind_face' not in [s['function']['name'] for s in files.schemas]


def test_sealing_tool_cli_and_copy_confinement(setup):
    p,a=setup
    files=gauntlet.Files(p,a)
    assert f.FILE in files.names
    assert 'sealed' in files.call('bind_face',{})
    with pytest.raises(agent.Rejected):
        files.call('bind_face',{'attempt':'a02'})
    with pytest.raises(ValueError,match='direct attempt'):
        f.seal(p,p.root)
    assert cli.main(['face-bind',str(p.root),'a01'])==0
    assert cli.main(['new',str(p.root),'copied','--from','a01'])==0
    assert (p.attempts_dir/'copied'/f.FILE).read_bytes()==(a/f.FILE).read_bytes()
    other=p.attempts_dir/'a02'
    other.mkdir()
    for name in ('model.py',f.FILE):
        shutil.copy2(a/name,other/name)
    (other/'model.py').write_text('changed geometry recipe')
    r,_=run(p,other)
    assert 'Stale' in f.feedback(r)
    with pytest.raises(agent.Rejected):
        files.call('write_file',{'path':'attempts/a02/face_bindings.json','text':'{}'})


def test_evaluation_preview_feedback_and_portability(setup,tmp_path):
    p,a=setup
    source=a/'out'/'puppet.glb'
    frozen=tmp_path/'frozen.glb'
    shutil.copy2(source,frozen)
    def build(script,out,spec,**kw):
        out.mkdir(exist_ok=True)
        shutil.copy2(frozen,out/glb.judged_glb(spec))
    def render(spec,built,out,**kw):
        return {'renders':{'front':str(p.references['front'])}}
    rec=evaluate.evaluate(p,a/'model.py',a,build_fn=build,render_fn=render,log=lambda s:None)
    assert rec['ok'],rec
    assert next(g for g in rec['gates'] if g['name']=='face_capabilities_sampled')['passed']
    assert 'FACIAL FUNCTION' in evaluate.feedback(rec)
    assert 'face-test' in evaluate.feedback(rec)
    stored=evaluate.portable(rec,a,p.root)
    assert all(not Path(v).is_absolute() for v in stored['face_pictures'].values())
    text=gauntlet.review_brief(p,rec,a)
    assert 'functional facial quality separately' in text
    assert 'mouth_open' in text and 'visual approval' in text
    numeric=rec['numeric']
    rec=evaluate.preview(p,a,render_fn=render)
    assert rec['face_capabilities']['required_checks_passed']
    assert all('/preview/face-test/' in Path(v).as_posix() for v in rec['face_pictures'].values())
    assert 'FACIAL FUNCTION' in evaluate.preview_text(p,a,rec)
    p.data['face_capabilities']={'version':1,'optional':['yell']}
    (a/f.FILE).unlink()
    optional=evaluate.evaluate(p,a/'model.py',a,build_fn=build,render_fn=render,log=lambda s:None)
    assert optional['ok'] and optional['face_capabilities']['errors']
    assert not any(g['name']=='face_capabilities_sampled' for g in optional['gates'])
    assert optional['numeric']==numeric
    del p.data['face_capabilities']
    legacy=evaluate.evaluate(p,a/'model.py',a,build_fn=build,render_fn=render,log=lambda s:None)
    assert legacy['numeric']==numeric and 'face_capabilities' not in legacy
    p.data['face_capabilities']=None
    invalid=evaluate.evaluate(p,a/'model.py',a,build_fn=build,render_fn=render,log=lambda s:None)
    assert not invalid['ok'] and invalid['stage']=='project'


def test_api_loop_function_feedback_survives_hidden_scores(tmp_path):
    p,_,initial,pictures=example.make(tmp_path/'seed')
    # Use a clean destination: the visual loop itself makes a01/a02.
    assert p.attempts_dir.resolve().parent==p.root.resolve()
    shutil.rmtree(p.attempts_dir)
    round_no=0
    writing=False
    reviews=0
    def tool(name,**args):
        return {'role':'assistant','tool_calls':[{'id':name,'type':'function','function':{'name':name,'arguments':json.dumps(args)}}]}
    def chat(messages,schemas):
        nonlocal round_no,writing,reviews
        text=json.dumps(agent.strip_images(messages))
        assert '0.4321' not in text
        if not schemas:
            reviews+=1
            assert 'FACIAL CAPABILITIES' in text and 'FACIAL FUNCTION' in text
            assert 'visual approval' in text
            return {'role':'assistant','content':'Preserve neutral; fix the mouth.'},{}
        if not writing:
            round_no+=1
            writing=True
            assert 'bind_face' in [s['function']['name'] for s in schemas]
            assert 'FACIAL CAPABILITIES' in text
            if round_no==2:
                assert 'FACIAL FUNCTION' in text and 'cavity regression failure' in text
            return tool('write_file',path=f'attempts/a{round_no:02d}/model.py',text='def build(): pass'),{}
        writing=False
        return tool('submit'),{}
    def fake_eval(proj,script,attempt,**kw):
        r=copy.deepcopy(initial)
        r['capabilities']['talk'].update(status='failed',issues=['cavity regression failure'])
        return {'ok':True,'name':attempt.name,'numeric':{'numeric_score':.4321},'gates':[],
                'compare':{'front':str(p.references['front'])},'face_capabilities':r,
                'face_pictures':{'neutral':str(p.references['front'])}}
    result=gauntlet.run(p,chat,iterations=2,checks='none',evaluate_fn=fake_eval,log=lambda s:None)
    assert result['done'] and result['iterations_completed']==2 and reviews==2,result


@pytest.mark.blender
def test_real_blender_export_and_functional_preview(blender,tmp_path):
    p,a,_,_=example.make(tmp_path/'blender-puppet')
    rec=evaluate.evaluate(p,a/'model.py',a,log=lambda s:None)
    assert rec['ok'],rec.get('error')
    assert rec['face_capabilities']['required_checks_passed'],f.feedback(rec.get('face_capabilities'))
    assert set(rec['face_pictures']) >= {'neutral','mouth_open','gaze_up','yell'}
