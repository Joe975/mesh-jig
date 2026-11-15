"""Pictures use the checked skin/terrain and expose time/stance, not idealised targets."""
import copy
import json

import numpy as np
import pytest
from PIL import Image

from mesh_jig import game_agent as G, game_check, game_look, game_scene_review as L, game_package, game_runtime, game_scenario as gs
from test_game import quad, quad_contract  # noqa: F401


@pytest.fixture(scope='module')
def review_case(quad,tmp_path_factory):
    rig=quad[0]
    package=tmp_path_factory.mktemp('look')/'package'
    assert game_package.build(rig,quad_contract(rig),package)['passed']
    runtime=json.loads((package/'character.runtime.json').read_text())
    job={'gaits':['walk','run'],'scenario_criteria':G.SCENARIO_CRITERIA}
    cases=[]
    for label,scenario in G.scenarios(runtime,job):
        trace=game_runtime.run(package,scenario)
        cases.append(dict(label=label,scenario=scenario,trace=trace,check=game_check.check(package,scenario,trace)))
    return package,cases


def test_failed_trace_frame_gets_adjacent_context():
    trace={'frames':[{'time':k/10} for k in range(20)]}
    ids=L.frame_indices(trace,{'gates':{'slip_mps':{'passed':False,'worst_time_s':1.1}}})
    assert {10,11,12}<=set(ids)
    assert 0 in ids and 19 in ids and len(ids)<=7
    assert len(L.frame_indices(trace,{'gates':{}}))==7


def test_obstacle_route_scales_with_rig_without_speeding_up_gait(review_case):
    package,cases=review_case
    runtime=json.loads((package/'character.runtime.json').read_text())
    base=next(x['scenario'] for x in cases if x['label']=='run-flat')
    before=copy.deepcopy(base)
    stress=L.stress_scenario(runtime,base)
    assert base==before
    assert base['duration']<stress['duration']<=L.MAX_STRESS_SECONDS
    speed=abs(base['root'][-1][3]-base['root'][0][3])/base['duration']
    assert abs(stress['root'][-1][3]-stress['root'][0][3])/stress['duration']==pytest.approx(speed)
    assert stress['terrain']['z'][0]<stress['root'][-1][3]
    assert stress['terrain']['z'][-1]>stress['root'][0][3]
    assert stress['criteria']==base['criteria']


def test_drawn_ground_uses_runtime_diagonal_and_patches():
    ground=gs.Ground({'terrain':{'x':[0,1],'z':[0,1],'heights':[[0,.1],[.3,.9]]}})
    ground.patch([[0,0],[.2,.2]],.15)
    mesh=game_look.ground_mesh(ground,ground.heights)
    assert mesh.colors.shape==mesh.tris.shape==mesh.normals.shape
    for triangle in mesh.tris:
        centre=triangle.mean(axis=0)
        assert centre[1]==pytest.approx(ground.snapshot()(centre[0],centre[2])[0])


def test_actual_runtime_pictures_are_deterministic_and_have_stance_metadata(review_case,tmp_path):
    package,cases=review_case
    pictures=L.capture(package,cases,tmp_path/'first')
    second=L.capture(package,cases,tmp_path/'second')
    assert len(pictures)==3
    assert [p.read_bytes() for p in pictures]==[p.read_bytes() for p in second]
    info=json.loads((tmp_path/'first/scenes.json').read_text())
    assert [x['label'] for x in info]==['run-playback','run-slope','run-obstacles']
    assert info[-1]['diagnostic_only'] and not info[0]['diagnostic_only']
    for picture,scene in zip(pictures,info):
        image=np.asarray(Image.open(picture))
        assert image.shape[0]==2*(L.SIZE+58)
        assert image.std()>5 and image.shape[1]==L.SIZE*len(scene['frames'])
        assert all('votes' in f and 'planted' in f for f in scene['frames'])
    # Posed geometry/ground differs, rather than returning the same source thumbnail.
    assert pictures[0].read_bytes()!=pictures[1].read_bytes()


def test_changed_trace_cannot_be_presented_as_checked_render(review_case,tmp_path):
    package,cases=review_case
    changed=copy.deepcopy(cases)
    next(x for x in changed if x['label']=='run-playback')['trace']['frames'][0]['joints'][0][12]+=.1
    with pytest.raises(ValueError,match='replay differs'):L.capture(package,changed,tmp_path/'changed')
