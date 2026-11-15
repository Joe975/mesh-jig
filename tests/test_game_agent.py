"""The post-rig model cannot change the frozen input, omit feet or grant itself easier gates."""
import copy
import json
from pathlib import Path

import pytest

from mesh_jig import cli, game_agent as G
from test_game import quad, quad_contract  # noqa: F401


@pytest.fixture()
def job(quad, tmp_path):
    rig = quad[0]
    root = tmp_path/'job'
    G.init(rig,root,legs=4,gaits=['walk','run'])
    policy = {'version':1,'cycle':[{'clip':c,'motion':'measured','cycles':2,'exit':'end'} for c in ['walk','run']]}
    proposal = quad_contract(rig)
    proposal.pop('criteria')
    return root,proposal,policy


def test_job_init_and_cli_reject_invalid_inputs(job,tmp_path):
    root,_,_ = job
    assert cli.parser().parse_args(['agent',str(root),'--workflow','game','--model','test']).workflow == 'game'
    assert G.load(root)[1]['legs'] == 4
    assert cli.parser().parse_args(['docs','game']).name == 'game'
    with pytest.raises(ValueError,match='new'):G.init(root/'animated.glb',root,legs=4,gaits=['walk'])
    with pytest.raises(ValueError,match='gaits'):G.init(root/'animated.glb',tmp_path/'bad',legs=4,gaits=['invented'])
    with pytest.raises(ValueError,match='inside'):G.Session(root,tmp_path/'outside',2)


@pytest.mark.parametrize('defect',['limits','missing_leg','missing_gait','rig'])
def test_model_cannot_weaken_required_capabilities_or_limits(job,tmp_path,defect):
    root,proposal,policy = job
    s=G.Session(root,root/'run',1)
    if defect=='limits':proposal['criteria']={'skate_mps_max':1000}
    if defect=='missing_leg':proposal['limbs'].pop()
    if defect=='missing_gait':proposal['graph']['layers'][0]['node']['points'][1]['clip']='walk'
    if defect=='rig':proposal['rig_sha256']='a'*64
    assert s.build(proposal,policy)['passed'] is False
    with pytest.raises(ValueError,match='budget'):s.build(proposal,policy)
    with pytest.raises(ValueError,match='passing'):s.done('a01','claim success')
    assert not (root/'run'/'character.glb').exists()


def test_playback_policy_requires_every_clip_and_declared_gait_motion(job):
    _,_,policy=job
    for change in ['missing','duplicate','stationary','path','reference','cycles']:
        p=copy.deepcopy(policy)
        if change=='missing':p['cycle'][0].pop('exit')
        if change=='duplicate':p['cycle'][1]['clip']='walk'
        if change=='stationary':p['cycle'][0]['motion']='stationary'
        if change=='path':p['cycle'][0]['output']='../../escape'
        if change=='reference':p['cycle'][0]['reference']='run'
        if change=='cycles':p['cycle'][0]['cycles']=True
        with pytest.raises(ValueError):G.playback_policy(p,['walk','run'],['walk','run'])


def test_valid_proposal_runs_independent_scenarios_and_offline_agent_delivers(job):
    root,proposal,policy=job
    def chat(messages,tools):
        if len(messages)==2:
            fn,args='build',{'character_json':json.dumps(proposal),'playback_json':json.dumps(policy)}
        else:
            answer=json.loads(messages[-1]['content'])
            assert answer['passed'],answer
            assert set(answer['checks'])=={'walk-playback','walk-flat','walk-slope','run-playback','run-flat','run-slope'}
            fn,args='done',{'best':answer['attempt'],'summary':'four-leg synthetic control'}
        return {'role':'assistant','content':None,'tool_calls':[{'id':f'call{len(messages)}','type':'function',
                'function':{'name':fn,'arguments':json.dumps(args)}}]},{}
    report=G.run(root,chat,out=root/'agent'/'offline',builds=2,log=lambda _:None)
    assert report['final']['best']=='a01'
    assert (root/'agent/offline/attempts/a01/package/character.runtime.json').is_file()
    assert report['inputs']['animated.glb']==G.terrain.digest(root/'animated.glb')
    transcript=(root/'agent/offline/transcript.jsonl').read_text().splitlines()
    assert json.loads(transcript[-1])['role']=='tool'


def test_failed_runtime_check_never_delivers(job,monkeypatch):
    root,proposal,policy=job
    monkeypatch.setattr(G.game_check,'check',lambda *a,**k:{'passed':False,'issues':['injected trace failure'],'gates':{}})
    s=G.Session(root,root/'run',1)
    assert not s.build(proposal,policy)['passed']
    with pytest.raises(ValueError,match='passing'):s.done('a01','ignore runtime')


def test_frozen_inputs_checked_before_delivery(job):
    root,_,_=job
    s=G.Session(root,root/'run',1)
    (root/'brief.md').write_text('changed')
    with pytest.raises(ValueError,match='frozen'):s.done('a01','done')


def test_invalid_json_shape_is_a_counted_attempt_and_failure_is_explicit(job):
    root,_,policy=job;s=G.Session(root,root/'invalid',1)
    assert not s.build([],policy)['passed']
    assert len(s.results)==1
    def chat(messages,tools):
        return {'role':'assistant','content':None,'tool_calls':[{'id':'stop','type':'function','function':
                {'name':'report_failure','arguments':json.dumps({'summary':'frozen gait fails slip'})}}]},{}
    report=G.run(root,chat,out=root/'agent'/'failed',builds=1,log=lambda _:None)
    assert report['final'] is None and report['stop']=='failed checks' and 'slip' in report['summary']


def test_landing_policy_requires_measured_flight_and_landing():
    c={'duration_s':2,'legs':{str(i):{'stance_phase':[[0,.2],[.7,1]]} for i in range(4)}}
    assert G.flight_window(c)==(.4,1.4)
    policy={'cycle':[{'clip':'jump','exit':'landing','motion':'stationary'}]}
    G.measured_policy(policy,{'clips':{'jump':c}})
    for l in c['legs'].values():l['stance_phase']=[[0,1]]
    with pytest.raises(ValueError,match='flight'):G.measured_policy(policy,{'clips':{'jump':c}})


def test_transport_failure_records_no_delivery(job):
    root,_,_=job
    def chat(*a):raise G.llm.LLMError('transport error')
    report=G.run(root,chat,out=root/'agent'/'transport',builds=1)
    assert report['stop']=='transport failed' and report['final'] is None
    assert (root/'agent/transport/report.json').exists()


def test_malformed_json_build_consumes_attempt(job):
    root,_,policy=job;s=G.Session(root,root/'malformed',1)
    assert not s.build('{',json.dumps(policy))['passed']
    with pytest.raises(ValueError,match='budget'):s.build('{}','{}')


def test_game_cli_refuses_other_workflow_flags_before_model_call(job,monkeypatch,capsys):
    root,_,_=job
    def unexpected(**kwargs):raise AssertionError('must reject before contacting a model')
    monkeypatch.setattr(cli.llm,'resolve_backend',unexpected)
    args=cli.parser().parse_args(['agent',str(root),'--workflow','game','--reviewer'])
    assert cli.cmd_game_agent(args)==2
    assert 'rig-only' in capsys.readouterr().err
