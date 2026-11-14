import json
from pathlib import Path
import pytest
from mesh_jig import metrics, metrics_native as native


def test_codex_totals_do_not_add_subsets():
    got=native.receipt([{'type':'turn.started'}, {'type':'turn.completed','usage':{
        'input_tokens':100,'cached_input_tokens':60,'output_tokens':50,'reasoning_output_tokens':40}}], 'codex')
    assert got['complete']
    assert got['usage']['input_tokens']==100
    assert got['usage']['output_tokens']==50
    assert got['usage']['cache_read_tokens']==60
    assert got['usage']['reasoning_tokens']==40
    assert got['duration_s'] is None


def test_codex_interrupted_is_partial():
    got=native.receipt([{'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':2}},
                        {'type':'turn.started'}],'codex')
    assert not got['complete']
    assert got['usage']['input_tokens']==1


def test_codex_missing_detail_is_unknown():
    got=native.receipt([{'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':2}}],'codex')
    assert got['usage']['cache_read_tokens'] is None
    assert got['usage']['reasoning_tokens'] is None


def test_codex_duplicate_explicit_turn():
    event={'type':'turn.completed','turn_id':'a','usage':{'input_tokens':1,'output_tokens':2}}
    got=native.receipt([event,event],'codex')
    assert got['reported_turns']==1


def test_claude_disjoint_input():
    got=native.receipt([{'type':'result','duration_ms':3000,'usage':{
        'input_tokens':1,'cache_read_input_tokens':2,'cache_creation_input_tokens':3,'output_tokens':4}}],'claude')
    assert got['usage']['input_tokens']==6
    assert got['duration_s']==3
    assert got['reported_turns'] is None


def test_claude_requires_single_result():
    with pytest.raises(ValueError): native.receipt([{'type':'result'},{'type':'result'}],'claude')


def agy(i=100,o=20):
    return dict(conversation_id='same',status='SUCCESS',duration_seconds=i,num_turns=i//10,
                usage=dict(input_tokens=i,output_tokens=o,cache_read_tokens=i*2,thinking_tokens=5))


def test_agy_unknown_cache_semantics():
    got=native.receipt([agy()],'agy')
    assert got['usage']['input_tokens'] is None
    assert got['usage']['cache_read_tokens']==200
    assert got['usage']['output_tokens']==20


def test_agy_explicit_additive_delta():
    got=native.delta(native.receipt([agy(200,40)],'agy',agy_input='excludes-cache'),
                     native.receipt([agy()],'agy',agy_input='excludes-cache'))
    assert got['usage']['input_tokens']==300
    assert got['usage']['output_tokens']==20
    assert got['duration_s']==100


def test_cumulative_wrong_session_or_decrease():
    a=native.receipt([agy()],'agy')
    b=native.receipt([dict(agy(),conversation_id='other')],'agy')
    with pytest.raises(ValueError): native.delta(a,b)
    with pytest.raises(ValueError): native.delta(native.receipt([agy(50,10)],'agy'),a)


def test_import_idempotent_and_private(tmp_path):
    source=tmp_path/'receipt.json'
    source.write_text(json.dumps(dict(agy(),response='SECRET-PROMPT',api_key='SECRET-KEY')))
    kwargs=dict(provider='agy',run='run',stage='model',invocation='first',scope='invocation')
    first=native.import_receipt(tmp_path/'metrics',source,**kwargs)
    assert first==native.import_receipt(tmp_path/'metrics',source,**kwargs)
    events=metrics.read(tmp_path/'metrics')
    assert len(events)==2
    assert 'SECRET' not in json.dumps(events)
    assert str(source) not in json.dumps(events)
    assert events[0]['operation']=='native-session'
    source.write_text(json.dumps(agy(200,40)))
    with pytest.raises(ValueError): native.import_receipt(tmp_path/'metrics',source,**kwargs)


def test_scope_must_be_explicit_and_consistent(tmp_path):
    source=tmp_path/'receipt.json'; source.write_text(json.dumps(agy()))
    kwargs=dict(provider='agy',run='run',stage='model',invocation='first')
    with pytest.raises(ValueError): native.import_receipt(tmp_path,source,scope='cumulative',**kwargs)
    with pytest.raises(ValueError): native.import_receipt(tmp_path,source,scope='invocation',before=source,**kwargs)


def test_null_duration_import(tmp_path):
    source=tmp_path/'receipt.jsonl'
    source.write_text(json.dumps({'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':2}})+'\n')
    native.import_receipt(tmp_path/'metrics',source,provider='codex',run='run',stage='model',invocation='first',scope='invocation')
    ends=[e for e in metrics.read(tmp_path/'metrics') if e['kind']=='end']
    assert ends[0]['duration_s'] is None


def test_codex_duplicate_anonymous_turn_rejected():
    event={'type':'turn.completed','usage':{'input_tokens':1,'output_tokens':2}}
    with pytest.raises(ValueError, match='ambiguous duplicate'):
        native.receipt([event, {'type':'turn.started'}, event],'codex')


def test_source_cannot_be_imported_twice_in_run(tmp_path):
    source=tmp_path/'receipt.json'; source.write_text(json.dumps(agy()))
    root=tmp_path/'metrics'
    kwargs=dict(provider='agy',run='run',stage='model',invocation='first',scope='invocation')
    native.import_receipt(root,source,**kwargs)
    with pytest.raises(ValueError, match='another invocation or stage'):
        native.import_receipt(root,source,**dict(kwargs,invocation='second'))
    with pytest.raises(ValueError, match='another invocation or stage'):
        native.import_receipt(root,source,**dict(kwargs,stage='rig'))
    assert len(metrics.read(root))==2
    native.import_receipt(root,source,**dict(kwargs,run='another-run'))
    assert len(metrics.read(root))==4
