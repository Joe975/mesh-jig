"""Opt-in forward-only stage telemetry. Raw events are private; summaries contain no prompts."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
import math
import os
from pathlib import Path
import re
import time
import uuid

_CONTEXT = ContextVar('mesh_jig_metrics', default=None)
FIELDS = ('input_tokens', 'output_tokens', 'cache_read_tokens', 'cache_write_tokens', 'reasoning_tokens')


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def normalize(raw, dialect='openai'):
    """Normalize documented inclusive input/output; details remain subsets, never added twice."""
    raw = raw or {}
    if dialect == 'openai' and isinstance(raw.get('native_usage'), dict):
        return normalize(raw['native_usage'], 'anthropic')
    if dialect == 'anthropic':
        r, w = number(raw.get('cache_read_input_tokens')), number(raw.get('cache_creation_input_tokens'))
        i = number(raw.get('input_tokens'))
        # Anthropic fields are disjoint. Missing cache fields do not certify zero.
        i = i + r + w if all(x is not None for x in (i, r, w)) else None
        o = number(raw.get('output_tokens'))
        reason = number((raw.get('output_tokens_details') or {}).get('thinking_tokens'))
    elif dialect == 'openai':
        i, o = number(raw.get('prompt_tokens')), number(raw.get('completion_tokens'))
        details = raw.get('prompt_tokens_details') or {}
        r, w = number(details.get('cached_tokens')), number(details.get('cache_write_tokens'))
        reason = number((raw.get('completion_tokens_details') or {}).get('reasoning_tokens'))
    elif dialect == 'codex':
        i, o = number(raw.get('input_tokens')), number(raw.get('output_tokens'))
        r, w = number(raw.get('cached_input_tokens')), number(raw.get('cache_write_tokens'))
        reason = number(raw.get('reasoning_output_tokens'))
    else:
        raise ValueError('unsupported usage dialect')
    values = dict(zip(FIELDS, (i, o, r, w, reason)))
    for key, total in (('cache_read_tokens', i), ('cache_write_tokens', i), ('reasoning_tokens', o)):
        if total is not None and values[key] is not None and values[key] > total:
            values[key] = None
    values['source'] = dialect
    return values


def valid_label(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.:+-]{1,120}', value):
        raise ValueError('metrics labels must be short identifiers, not paths or prose')
    return value


class Ledger:
    def __init__(self, root, run='run', stage='unspecified'):
        self.root = Path(root)
        self.run, self.stage = valid_label(run), valid_label(stage)

    def emit(self, kind, **data):
        # One atomic event per file avoids cross-process appends and shared summary races.
        event = dict(schema=1, id=uuid.uuid4().hex, run=self.run, stage=self.stage,
                     kind=kind, epoch_s=time.time(), **data)
        folder = self.root / 'events'
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (event['id'] + '.json')
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(event, allow_nan=False), encoding='utf-8')
        temp.replace(path)
        return event


def current():
    context = _CONTEXT.get()
    if context is not None:
        return context
    path = os.environ.get('MESH_JIG_METRICS_DIR')
    return Ledger(path, os.environ.get('MESH_JIG_METRICS_RUN', 'run'),
                  os.environ.get('MESH_JIG_METRICS_STAGE', 'unspecified')) if path else None


@contextmanager
def stage(root, name, run='run'):
    ledger = Ledger(root, run, name)
    token = _CONTEXT.set(ledger)
    try:
        with span('stage', ledger=ledger):
            yield ledger
    finally:
        _CONTEXT.reset(token)


class Record:
    def __init__(self):
        self.data = {}
    def usage(self, raw, dialect='openai'):
        self.data['usage'] = normalize(raw, dialect)
        # Only numeric cost is retained; no credentials, prompts or server error bodies.
        cost = number((raw or {}).get('cost'))
        if cost is not None:
            self.data['reported_cost_usd'] = cost
    def result(self, success):
        self.data['status'] = 'completed' if success else 'failed'


@contextmanager
def span(kind, ledger=None, **metadata):
    ledger = ledger or current()
    rec = Record()
    if ledger is None:
        yield rec
        return
    identifier = uuid.uuid4().hex
    started = time.monotonic()
    ledger.emit('start', span=identifier, operation=kind, metadata=metadata)
    try:
        yield rec
    except BaseException as error:
        rec.data.update(status='failed', error_type=type(error).__name__)
        raise
    finally:
        rec.data.setdefault('status', 'completed')
        ledger.emit('end', span=identifier, operation=kind, duration_s=time.monotonic()-started, **rec.data)


def request(**metadata):
    return span('request', **metadata)


def measured(kind):
    """Instrument Python tool entry points without changing results or mandatory output files."""
    def decorate(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            with span(kind) as rec:
                result = fn(*args, **kwargs)
                if isinstance(result, dict) and isinstance(result.get('ok'), bool):
                    rec.result(result['ok'])
                elif isinstance(getattr(result, 'ok', None), bool):
                    rec.result(result.ok)
                return result
        return wrapped
    return decorate


def read(root):
    events = []
    seen = set()
    for path in sorted((Path(root)/'events').glob('*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        if row.get('schema') != 1:
            raise ValueError('unsupported metrics schema')
        if row['id'] in seen:
            raise ValueError('duplicate metrics event identity')
        seen.add(row['id'])
        events.append(row)
    return events


def union_seconds(intervals):
    total, end = 0., None
    for left, right in sorted(intervals):
        total += max(0., right - max(left, end if end is not None else left))
        end = max(end if end is not None else right, right)
    return total


def summarize(events):
    rows = []
    for run, name in sorted({(e['run'], e['stage']) for e in events}):
        group = [e for e in events if (e['run'], e['stage']) == (run, name)]
        starts = {e['span']: e for e in group if e['kind'] == 'start'}
        ends = {e['span']: e for e in group if e['kind'] == 'end'}
        requests = [e for e in starts.values() if e['operation'] in ('request', 'native-session')]
        usages = [ends.get(e['span'], {}).get('usage', {}) for e in requests]
        tokens = {}
        for key in FIELDS:
            known = [u[key] for u in usages if u.get(key) is not None]
            tokens[key] = {'known': sum(known) if known else None, 'reported': len(known), 'expected': len(requests)}
        counts = {}
        timings = {}
        for kind in sorted({e['operation'] for e in starts.values()}):
            selected = [e for e in starts.values() if e['operation'] == kind]
            finished = [ends[e['span']] for e in selected if e['span'] in ends]
            counts[kind] = dict(started=len(selected), completed=sum(e['status']=='completed' for e in finished),
                                failed=sum(e['status']=='failed' for e in finished), unfinished=len(selected)-len(finished))
            intervals = [(s['epoch_s'], s['epoch_s']+ends[s['span']]['duration_s']) for s in selected if s['span'] in ends and ends[s['span']].get('duration_s') is not None and not s.get('metadata',{}).get('imported')]
            timings[kind] = {'observed_active_s': round(union_seconds(intervals), 3) if intervals else None,
                             'summed_s': round(sum(e['duration_s'] for e in finished if e.get('duration_s') is not None), 3) if any(e.get('duration_s') is not None for e in finished) else None,
                             'complete': len(finished)==len(selected) and all(e.get('duration_s') is not None for e in finished)}
        incomplete = any(e['span'] not in ends for e in starts.values())
        complete = bool(requests) and not incomplete and all(ends.get(e['span'], {}).get('usage_complete', True) for e in requests) and all(u.get('input_tokens') is not None and u.get('output_tokens') is not None for u in usages)
        costs = [ends[e['span']]['reported_cost_usd'] for e in requests if e['span'] in ends and 'reported_cost_usd' in ends[e['span']]]
        rows.append(dict(run=run, stage=name, timing=timings, counts=counts, tokens=tokens,
                         token_coverage='complete' if complete else 'partial' if any(t['reported'] for t in tokens.values()) else 'unavailable',
                         cost={'known_reported_usd': sum(costs) if costs else None, 'reported_requests': len(costs), 'expected_requests': len(requests)},
                         provenance='mesh-jig event ledger v1; observed operations only'))
    return {'schema': 1, 'stages': rows}


def markdown(summary):
    lines = ['| Run | Stage | Observed active time | Builds/evals started | HTTP requests / native sessions | Input / output tokens | Token coverage | Reported USD (coverage) |',
             '|---|---|---:|---:|---:|---:|---|---:|']
    for row in summary['stages']:
        counts, timing, tokens = row['counts'], row['timing'], row['tokens']
        value = lambda key: str(tokens[key]['known']) if tokens[key]['known'] is not None else 'unknown'
        seconds = timing.get('stage', timing.get('command', {})).get('observed_active_s')
        attempts = sum(v['started'] for k,v in counts.items() if k in ('evaluation', 'rig-build', 'sound-build'))
        cost = row['cost']
        usd = f"${cost['known_reported_usd']:.6f}" if cost['known_reported_usd'] is not None else 'unknown'
        lines.append(f"| {row['run']} | {row['stage']} | {seconds if seconds is not None else 'unknown'} s | {attempts} | {counts.get('request', {}).get('started',0)} / {counts.get('native-session', {}).get('started',0)} | {value('input_tokens')} / {value('output_tokens')} | {row['token_coverage']} | {usd} ({cost['reported_requests']}/{cost['expected_requests']}) |")
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--markdown', action='store_true')
    args = parser.parse_args(argv)
    summary = summarize(read(args.directory))
    print(markdown(summary) if args.markdown else json.dumps(summary, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
