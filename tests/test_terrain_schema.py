"""Public terrain data has a bounded schema and an exact model/payload binding."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
POLICY=ROOT/'publication' if (ROOT/'publication/terrain_schema.py').exists() else ROOT/'tools'
SITE=ROOT/'publication/site' if (ROOT/'publication/site').exists() else ROOT/'site'
spec=importlib.util.spec_from_file_location('terrain_schema_test',POLICY/'terrain_schema.py')
schema=importlib.util.module_from_spec(spec);spec.loader.exec_module(schema)


@pytest.mark.parametrize('name',['horror','horse'])
def test_selected_payloads_and_registry_are_bound_and_anonymous(name):
    raw=(SITE/f'terrain/{name}.json').read_bytes();d=schema.payload(raw)
    assert len(d['runtime']['limbs']) in (4,6)
    assert b'local-work' not in raw and b'contract_sha256' not in raw
    registry=schema.registry((SITE/'terrain/characters.json').read_bytes())
    assert any(row['data']==name+'.json' and row['glb_sha256']==d['glb_sha256'] for row in registry['characters'].values())


@pytest.mark.parametrize('defect',['provenance','joint_cycle','node_cycle','missing_clip','bad_keys','nonfinite','aim','flight','reference','sync'])
def test_public_payload_refuses_private_or_unsupported_data(defect):
    d=json.loads((SITE/'terrain/horse.json').read_bytes());r=d['runtime']
    if defect=='provenance':d['source_path']='private/path'
    if defect=='joint_cycle':r['skeleton']['joints'][0]['parent']=r['skeleton']['joints'][-1]['name']
    if defect=='node_cycle':d['nodes'][0].setdefault('children',[]).append(0)
    if defect=='missing_clip':d['demo']['cycle'].pop()
    if defect=='bad_keys':next(iter(d['tracks'].values()))[0]['times']=[0,0]
    if defect=='nonfinite':r['ground']['blend_s']=float('nan')
    if defect=='aim':r['aims']=[{'private':'name'}]
    if defect=='flight':d['demo']['cycle'][0]['exit']='landing'
    if defect=='reference':next(row for row in d['demo']['cycle'] if row['motion']=='airborne')['reference']='idle'
    if defect=='sync':r['graph']['layers'][0]['node']['sync']=True
    with pytest.raises((ValueError,TypeError,KeyError)):
        schema.payload(json.dumps(d).encode())


@pytest.mark.parametrize('field,value',[('data','../private.json'),('model','https://example.com/model.glb'),('sha256','bad')])
def test_registry_has_no_external_or_escaping_paths(field,value):
    d=json.loads((SITE/'terrain/characters.json').read_bytes())
    next(iter(d['characters'].values()))[field]=value
    with pytest.raises(ValueError):schema.registry(json.dumps(d).encode())


def test_release_requires_exact_registry_payload_and_model_bytes():
    import hashlib
    spec=importlib.util.spec_from_file_location('terrain_release_test',POLICY/'check_publication.py')
    guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
    model=b'isolated hash-binding fixture';sha=hashlib.sha256(model).hexdigest()
    d=json.loads((SITE/'terrain/horror.json').read_bytes());d['glb_sha256']=sha;raw=json.dumps(d).encode()
    digest=hashlib.sha256(raw).hexdigest();path='showcase/test/model-01.glb'
    files={'site/terrain/characters.json':json.dumps({'version':1,'characters':{'test':{'data':'horror.json','model':path,'sha256':digest,'glb_sha256':sha}}}).encode(),
           'site/terrain/horror.json':raw,path:model,
           'site/terrain/horror.release.json':json.dumps({'format':'mesh-jig terrain release','version':1,'files':{'horror.json':digest},'glb_sha256':sha}).encode()}
    assert guard.terrain_release(files)==[]
    for name in files:
        altered=dict(files);altered[name]+=b'changed';assert guard.terrain_release(altered)
    extra=dict(files);extra['site/terrain/unknown.json']=raw;assert guard.terrain_release(extra)
