import json
import pytest
from mesh_jig import metrics


def test_disabled_no_write(tmp_path, monkeypatch):
    monkeypatch.delenv('MESH_JIG_METRICS_DIR', raising=False)
    with metrics.request(model='x') as call:
        call.usage({'prompt_tokens': 4, 'completion_tokens': 2})
    assert not list(tmp_path.iterdir())


def test_stage_requests_and_failure(tmp_path):
    with metrics.stage(tmp_path, 'model'):
        with metrics.request(model='x') as call:
            call.usage({'prompt_tokens': 100, 'completion_tokens': 10, 'cost': 0,
                        'prompt_tokens_details': {'cached_tokens': 80},
                        'completion_tokens_details': {'reasoning_tokens': 6}})
        with pytest.raises(TimeoutError):
            with metrics.request(model='x'):
                raise TimeoutError('secret error text')
    events = metrics.read(tmp_path)
    row = metrics.summarize(events)['stages'][0]
    assert row['tokens']['input_tokens'] == {'known': 100, 'reported': 1, 'expected': 2}
    assert row['tokens']['output_tokens']['known'] == 10
    assert row['token_coverage'] == 'partial'
    assert row['counts']['request']['failed'] == 1
    assert row['cost']['known_reported_usd'] == 0
    assert 'secret error text' not in json.dumps(events)


def test_normalization_subsets_and_missing():
    result = metrics.normalize({'input_tokens': 5, 'cache_read_input_tokens': 20,
        'cache_creation_input_tokens': 10, 'output_tokens': 8,
        'output_tokens_details': {'thinking_tokens': 6}}, 'anthropic')
    assert result['input_tokens'] == 35
    assert result['output_tokens'] == 8
    assert result['reasoning_tokens'] == 6
    assert metrics.normalize({})['input_tokens'] is None
    assert metrics.normalize({'prompt_tokens': True})['input_tokens'] is None
    assert metrics.normalize({'input_tokens': 1, 'output_tokens': 2}, 'anthropic')['input_tokens'] is None


def test_empty_provider_usage_is_unavailable(tmp_path):
    with metrics.stage(tmp_path, 'model'):
        with metrics.request(model='test') as call:
            call.usage({})
    row = metrics.summarize(metrics.read(tmp_path))['stages'][0]
    assert row['token_coverage'] == 'unavailable'
    assert row['tokens']['input_tokens']['known'] is None


def test_markdown_distinguishes_runs_and_shows_cost_coverage(tmp_path):
    for run, raw in [('first', {'prompt_tokens': 3, 'completion_tokens': 2, 'cost': 0}),
                     ('second', {'prompt_tokens': 5, 'completion_tokens': 1, 'cost': 0.125}),
                     ('third', {})]:
        with metrics.stage(tmp_path, 'model', run):
            with metrics.request(model='test') as call:
                call.usage(raw)
            with metrics.request(model='test'):
                pass
    report = metrics.markdown(metrics.summarize(metrics.read(tmp_path)))
    assert '| first | model |' in report and '| second | model |' in report
    assert '$0.000000 (1/2)' in report
    assert '$0.125000 (1/2)' in report
    assert 'unknown (0/2)' in report


def test_unfinished_span(tmp_path):
    metrics.Ledger(tmp_path).emit('start', span='crashed', operation='request', metadata={})
    row = metrics.summarize(metrics.read(tmp_path))['stages'][0]
    assert row['counts']['request']['unfinished'] == 1
    assert row['tokens']['input_tokens']['known'] is None
    assert row['token_coverage'] == 'unavailable'


def test_overlap_union():
    assert metrics.union_seconds([(0,4),(2,6),(7,8)]) == 7


def test_decorator_failure_result(tmp_path):
    @metrics.measured('evaluation')
    def fail(): return {'ok': False}
    with metrics.stage(tmp_path, 'model'):
        assert fail() == {'ok':False}
    row=metrics.summarize(metrics.read(tmp_path))['stages'][0]
    assert row['counts']['evaluation']['failed']==1


def test_environment_and_private_projection(tmp_path, monkeypatch):
    monkeypatch.setenv('MESH_JIG_METRICS_DIR',str(tmp_path))
    monkeypatch.setenv('MESH_JIG_METRICS_STAGE','rig-animation')
    with metrics.request(model='test',provider='private-url'):
        pass
    result=metrics.summarize(metrics.read(tmp_path))
    assert result['stages'][0]['stage']=='rig-animation'
    assert 'private-url' not in json.dumps(result)
    assert 'unknown' in metrics.markdown(result)


def test_no_invalid_labels(tmp_path):
    with pytest.raises(ValueError): metrics.Ledger(tmp_path, stage='../secret')


def test_cli_options():
    from mesh_jig.cli import parser
    args=parser().parse_args(['--metrics-dir','local-work/metrics','--metrics-stage','model','doctor'])
    assert args.metrics_stage=='model'


def test_imported_partial_usage_is_not_complete(tmp_path):
    ledger=metrics.Ledger(tmp_path)
    ledger.emit('start',span='n',operation='native-session',metadata={'imported':True})
    ledger.emit('end',span='n',operation='native-session',status='failed',duration_s=None,
                usage={'input_tokens':10,'output_tokens':5},usage_complete=False)
    row=metrics.summarize(metrics.read(tmp_path))['stages'][0]
    assert row['token_coverage']=='partial'
    assert row['timing']['native-session']['observed_active_s'] is None
    assert row['timing']['native-session']['complete'] is False


def test_parallel_event_writers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    def worker(i):
        with metrics.stage(tmp_path,'model'):
            with metrics.request(model='test') as call:
                call.usage({'prompt_tokens':i,'completion_tokens':1})
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(worker,range(12)))
    row=metrics.summarize(metrics.read(tmp_path))['stages'][0]
    assert row['counts']['request']['completed']==12
    assert row['tokens']['input_tokens']['known']==66


def test_provider_cache_accounting_and_snapshot_replacement():
    anthropic = metrics.normalize({'input_tokens':50, 'cache_read_input_tokens':100,
        'cache_creation_input_tokens':20, 'output_tokens':10}, 'anthropic')
    openai = metrics.normalize({'prompt_tokens':170, 'completion_tokens':10,
        'prompt_tokens_details':{'cached_tokens':100}})
    assert anthropic['input_tokens'] == openai['input_tokens'] == 170
    rec = metrics.Record()
    rec.usage({'prompt_tokens':170, 'completion_tokens':4})
    rec.usage({'prompt_tokens':170, 'completion_tokens':10})
    assert rec.data['usage']['output_tokens'] == 10
