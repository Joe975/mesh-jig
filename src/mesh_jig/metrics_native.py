"""Import an explicitly supplied, stage-scoped native CLI usage receipt (never global histories).

No model is invoked. Use one distinct invocation ID for each fresh stage invocation. A cumulative
receipt needs its immediately preceding snapshot; unknown fields remain unknown, not zero.
Verify invocation versus cumulative semantics for the installed CLI before importing; this helper
makes no claim that live provider semantics have been certified. A conversation spanning model
and rig needs a boundary receipt: never attribute its whole total to both stages. Native receipts
must not overlap separately recorded API-request usage.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
import time

from . import metrics


def load(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    try:
        value = json.loads(text)
        return value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _sum_known(rows, key):
    values = [row.get(key) for row in rows]
    return sum(values) if values and all(value is not None for value in values) else None


def _agy(raw, cache_mode):
    i = metrics.number(raw.get('input_tokens'))
    r = metrics.number(raw.get('cache_read_tokens'))
    if cache_mode == 'excludes-cache':
        i = i+r if i is not None and r is not None else None
    elif cache_mode != 'includes-cache':
        i = None
    return dict(input_tokens=i, output_tokens=metrics.number(raw.get('output_tokens')),
                cache_read_tokens=r, cache_write_tokens=None,
                reasoning_tokens=metrics.number(raw.get('thinking_tokens')), source='agy')


def receipt(events, provider, *, agy_input='unknown'):
    """Normalize only usage and completeness; arbitrary text, paths, credentials are discarded."""
    if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
        raise ValueError('receipt must contain JSON objects')
    if provider == 'codex':
        rows, complete, pending = [], True, False
        session, seen, anonymous = None, set(), set()
        for e in events:
            kind = e.get('type')
            if kind == 'thread.started':
                session = e.get('thread_id')
            elif kind == 'turn.started':
                pending = True
            elif kind == 'turn.completed':
                key = e.get('turn_id') or e.get('id')
                if key is not None:
                    if key in seen:
                        continue
                    seen.add(key)
                else:
                    fingerprint = json.dumps(e, sort_keys=True, separators=(',', ':'))
                    if fingerprint in anonymous:
                        raise ValueError('ambiguous duplicate Codex completed event without turn ID')
                    anonymous.add(fingerprint)
                rows.append(metrics.normalize(e.get('usage'), 'codex'))
                pending = False
            elif kind in ('turn.failed', 'error'):
                complete = False
                pending = False
        usage = {key: _sum_known(rows, key) for key in metrics.FIELDS}
        usage['source'] = 'codex-exec-json'
        return dict(usage=usage, complete=complete and not pending and bool(rows),
                    session=session, duration_s=None, reported_turns=len(rows))
    if provider == 'claude':
        results = [e for e in events if e.get('type') == 'result']
        if len(results) != 1:
            raise ValueError('Claude import requires exactly one final result receipt')
        e = results[0]
        duration = metrics.number(e.get('duration_ms'))
        return dict(usage=metrics.normalize(e.get('usage'), 'anthropic'), complete=not bool(e.get('is_error')),
                    session=e.get('session_id'), duration_s=duration/1000 if duration is not None else None,
                    reported_turns=metrics.number(e.get('num_turns')))
    if provider == 'agy':
        if len(events) != 1:
            raise ValueError('Antigravity import requires one JSON output snapshot')
        e = events[0]
        return dict(usage=_agy(e.get('usage') or {}, agy_input), complete=e.get('status') == 'SUCCESS',
                    session=e.get('conversation_id'), duration_s=metrics.number(e.get('duration_seconds')),
                    reported_turns=metrics.number(e.get('num_turns')))
    raise ValueError('unsupported native provider')


def delta(after, before):
    if not after.get('session') or after['session'] != before.get('session'):
        raise ValueError('cumulative snapshots require the same explicit session identity')
    out = dict(after, usage=dict(after['usage']))
    for key in metrics.FIELDS:
        a, b = after['usage'].get(key), before['usage'].get(key)
        if a is not None and b is not None and a < b:
            raise ValueError('cumulative usage counter decreased; do not guess a reset')
        out['usage'][key] = a-b if a is not None and b is not None else None
    for key in ('duration_s', 'reported_turns'):
        a, b = after.get(key), before.get(key)
        if a is not None and b is not None and a < b:
            raise ValueError('cumulative counter decreased')
        out[key] = a-b if a is not None and b is not None else None
    return out


def import_receipt(root, source, *, provider, run, stage, invocation, scope, before=None, agy_input='unknown'):
    """Idempotent import; changed evidence for an existing invocation is rejected, not double counted."""
    if scope not in ('invocation', 'cumulative'):
        raise ValueError('explicit invocation or cumulative scope required')
    if (scope == 'cumulative') != (before is not None):
        raise ValueError('cumulative scope requires before; invocation scope must not use before')
    metrics.valid_label(invocation)
    ledger = metrics.Ledger(root, run, stage)
    result = receipt(load(source), provider, agy_input=agy_input)
    if before is not None:
        result = delta(result, receipt(load(before), provider, agy_input=agy_input))
    content = Path(source).read_bytes()
    source_hash = hashlib.sha256(content).hexdigest()
    before_hash = hashlib.sha256(Path(before).read_bytes()).hexdigest() if before else None
    identity = hashlib.sha256(json.dumps([run, stage, provider, invocation]).encode()).hexdigest()
    metadata = dict(provider=provider, source_sha256=source_hash, before_sha256=before_hash,
                    scope=scope, agy_input=agy_input if provider=='agy' else None, imported=True,
                    timing_basis='provider-duration' if result['duration_s'] is not None else 'unavailable',
                    usage_unit='native-session', provenance='explicit stage-scoped CLI receipt')
    epoch = time.time()
    start = dict(schema=1, id=identity+'-start', run=run, stage=stage, kind='start', epoch_s=epoch,
                 span=identity, operation='native-session', metadata=metadata)
    end = dict(schema=1, id=identity+'-end', run=run, stage=stage, kind='end', epoch_s=epoch,
               span=identity, operation='native-session', duration_s=result['duration_s'],
               status='completed' if result['complete'] else 'failed', usage=result['usage'],
               usage_complete=result['complete'], reported_turns=result['reported_turns'])
    folder = ledger.root/'events'
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder/(row['id']+'.json') for row in (start,end)]
    # Atomic no-overwrite publication: interrupted/concurrent imports cannot expose partial JSON.
    def publish(path, row):
        temp = path.with_name(uuid.uuid4().hex+'.tmp')
        try:
            temp.write_text(json.dumps(row, allow_nan=False), encoding='utf-8')
            os.link(temp, path)
            return True
        except FileExistsError:
            return False
        finally:
            temp.unlink(missing_ok=True)
    # Atomic source claim prevents concurrent imports under different invocation IDs.
    claims = ledger.root/'native-source-claims'
    claims.mkdir(parents=True, exist_ok=True)
    claim_id = hashlib.sha256(json.dumps([run, source_hash]).encode()).hexdigest()
    claim_path = claims/(claim_id+'.json')
    claim = dict(run=run, source_sha256=source_hash, span=identity)
    if not publish(claim_path, claim):
        previous = json.loads(claim_path.read_text(encoding='utf-8'))
        if previous != claim:
            raise ValueError('source receipt already imported under another invocation or stage in this run')
    if not publish(paths[0], start):
        old = json.loads(paths[0].read_text(encoding='utf-8'))
        if old['metadata'] != metadata:
            raise ValueError('invocation already imported with different evidence or scope')
    publish(paths[1], end)
    return identity


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('source',type=Path)
    p.add_argument('--provider',choices=('codex','claude','agy'),required=True)
    p.add_argument('--run',required=True)
    p.add_argument('--stage',required=True)
    p.add_argument('--invocation',required=True)
    p.add_argument('--scope',choices=('invocation','cumulative'),required=True)
    p.add_argument('--before',type=Path)
    p.add_argument('--agy-input',choices=('unknown','includes-cache','excludes-cache'),default='unknown')
    a=p.parse_args(argv)
    print(import_receipt(a.directory,a.source,provider=a.provider,run=a.run,stage=a.stage,
                         invocation=a.invocation,scope=a.scope,before=a.before,agy_input=a.agy_input))
    return 0

if __name__=='__main__':
    raise SystemExit(main())
