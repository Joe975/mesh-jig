"""Request instrumentation counts transport dispatches, not logical turns or budget refusals."""
import io
import json
import urllib.error
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from mesh_jig import llm, anthropic_sweep as native, metrics


@pytest.fixture
def requests(monkeypatch):
    rows = []
    @contextmanager
    def request(**metadata):
        row = dict(metadata)
        rows.append(row)
        def usage(raw, *, dialect):
            row.update(usage=raw, dialect=dialect)
        try:
            yield SimpleNamespace(usage=usage)
        except BaseException:
            row['failed'] = True
            raise
        else:
            row['failed'] = False
    monkeypatch.setattr(metrics, 'request', request)
    return rows


def test_blocking_retry_is_two_requests_and_one_known_usage(requests):
    opened = []
    def opener(*args, **kwargs):
        opened.append(1)
        if len(opened) == 1:
            raise urllib.error.HTTPError('https://example.test', 500, 'failed', {}, io.BytesIO())
        return io.BytesIO(json.dumps({'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}).encode())
    llm._post('https://example.test', {'model': 'test'}, None, 1, opener=opener, sleep=lambda _: None)
    assert [r['retry_index'] for r in requests] == [0, 1]
    assert requests[0]['failed'] and 'usage' not in requests[0]
    assert requests[1]['usage'] == {'prompt_tokens': 3, 'completion_tokens': 2}


def test_stream_watchdog_failure_and_retry_measured_on_caller(requests, monkeypatch):
    def consume(*args, **kwargs):
        if len(requests) == 1:
            raise llm.ChatStalled('idle', True)
        return {'role': 'assistant', 'content': 'ok'}, {'prompt_tokens': 4, 'completion_tokens': 2}
    monkeypatch.setattr(llm, 'consume_with_watchdog', consume)
    result = llm.chat_stream(llm.Backend('local', 'http://localhost:1234', 'test'), [], retries=1)
    assert result['text'] == 'ok'
    assert len(requests) == 2
    assert [r['retry_index'] for r in requests] == [0, 1]
    assert requests[0]['failed'] and 'usage' not in requests[0]
    assert requests[1]['dialect'] == 'openai'


def test_no_request_when_call_budget_already_exhausted(requests):
    with pytest.raises(llm.OutOfTime):
        llm.chat_stream(llm.Backend('local', 'http://localhost:1234', 'test'), [], budget_s=0)
    assert requests == []


def test_native_budget_refusal_is_not_request(requests, tmp_path):
    api = SimpleNamespace(model='claude-opus-5-5', url='https://api.anthropic.com/v1', key='test', timeout_s=1)
    budget = native.Budget(tmp_path / 'budget.json', .01)
    with pytest.raises(llm.LLMError, match='budget stop'):
        native.chat_for(api, budget, 'test')([{'role': 'user', 'content': 'x'}], [])
    assert requests == []


def test_native_failure_unknown_and_success_usage(requests):
    api = SimpleNamespace(model='claude-opus-5-5', url='https://api.anthropic.com/v1', key='test', timeout_s=1)
    def failed(*args, **kwargs):
        raise OSError('offline')
    with pytest.raises(OSError):
        native.chat_for(api, None, 'test', opener=failed)([{'role': 'user', 'content': 'x'}], [])
    assert requests[0]['failed'] and 'usage' not in requests[0]
    events = [
        {'type': 'message_start', 'message': {'model': api.model, 'usage': {'input_tokens': 5}}},
        {'type': 'message_delta', 'usage': {'output_tokens': 1}},
        {'type': 'message_delta', 'usage': {'output_tokens': 2}},
        {'type': 'message_stop'},
    ]
    def opener(*args, **kwargs):
        return io.BytesIO(b''.join(b'data: ' + json.dumps(e).encode() + b'\n\n' for e in events))
    native.chat_for(api, None, 'test', opener=opener)([{'role': 'user', 'content': 'x'}], [])
    assert len(requests) == 2 and not requests[1]['failed']
    assert requests[1]['usage'] == {'input_tokens': 5, 'output_tokens': 2}
    assert requests[1]['dialect'] == 'anthropic'


def test_real_ledger_keeps_retry_usage_partial(tmp_path):
    opened = []
    def opener(*args, **kwargs):
        opened.append(1)
        if len(opened) == 1:
            raise urllib.error.HTTPError('https://example.test', 500, 'failed', {}, io.BytesIO())
        return io.BytesIO(json.dumps({'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}).encode())
    with metrics.stage(tmp_path, 'model'):
        llm._post('https://example.test', {'model': 'test'}, None, 1, opener=opener, sleep=lambda _: None)
    events = metrics.read(tmp_path)
    starts = [e for e in events if e.get('operation') == 'request' and e['kind'] == 'start']
    ends = [e for e in events if e.get('operation') == 'request' and e['kind'] == 'end']
    assert len(starts) == len(ends) == 2
    assert sum(e['status'] == 'failed' for e in ends) == 1
    assert len([e for e in ends if 'usage' in e]) == 1
    assert next(e['usage']['input_tokens'] for e in ends if 'usage' in e) == 3
