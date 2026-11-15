"""Frozen-mesh rough-terrain trials, measured from exported animation.

All coordinates in the contract are Blender metres (+Z up, +Y forward).
Terrain is a finite triangulated heightfield; checks are finite samples, not
continuous collision detection or a dynamics simulation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np

from . import glb, render, rig, skin

VERSION = 1
LIMITS = {
    'penetration_mm': 'max', 'stance_gap_mm': 'max', 'slip_mm': 'max',
    'path_error_mm': 'max', 'support_fraction': 'min', 'swing_clearance_mm': 'min',
}


def finite(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive')
    return float(value)


def region(spec):
    if not isinstance(spec, dict) or set(spec) != {'node', 'box'} or not isinstance(spec['node'], str):
        raise ValueError('region needs exactly node and box (source Blender metres)')
    box = np.asarray(spec['box'], float)
    if box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] < box[0]):
        raise ValueError('region box needs finite ordered [lo, hi] three-vectors')


def contract(spec):
    try:
        _contract(spec)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError(f'invalid terrain contract: {error}') from error


def _contract(spec):
    required = {'version', 'rig', 'clip', 'terrain', 'body', 'contacts', 'route', 'sampling', 'criteria'}
    if not isinstance(spec, dict) or set(spec) != required or isinstance(spec['version'], bool) or spec['version'] != VERSION:
        raise ValueError(f'terrain v1 needs exactly {sorted(required)}')
    rig.contract(spec['rig'])
    if set(spec['rig']['clips']) != {spec['clip']}:
        raise ValueError('terrain trial needs exactly its named clip')
    c = spec['rig']['clips'][spec['clip']]
    if c.get('loop', False):
        raise ValueError('terrain traversal must be a non-loop clip')
    seconds = c['seconds']
    field = spec['terrain']
    if not isinstance(field, dict) or set(field) != {'x', 'y', 'heights'}:
        raise ValueError('terrain needs x, y and heights')
    for key in ('x', 'y'):
        axis = np.asarray(field[key], float)
        if axis.ndim != 1 or not 2 <= len(axis) <= 257 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError(f'terrain {key} must be 2..257 finite increasing coordinates')
    h = np.asarray(field['heights'], float)
    if h.shape != (len(field['y']), len(field['x'])) or not np.isfinite(h).all():
        raise ValueError('terrain heights must be finite rows by y, columns by x')
    region(spec['body'])
    sampling = spec['sampling']
    if set(sampling) != {'hz', 'surface_divisions'}:
        raise ValueError('sampling needs hz and surface_divisions')
    hz = sampling['hz']
    if isinstance(hz, bool) or not isinstance(hz, int) or not 2*spec['rig'].get('fps', 30) <= hz <= 240:
        raise ValueError('sampling hz must be an integer at least twice rig fps and at most 240')
    n = sampling['surface_divisions']
    if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= 8:
        raise ValueError('surface_divisions must be an integer 1..8')
    route = np.asarray(spec['route'], float)
    if route.ndim != 2 or route.shape[1] != 3 or len(route) < 2 or not np.isfinite(route).all():
        raise ValueError('route needs [seconds, x, y] waypoints')
    if route[0, 0] != 0 or route[-1, 0] != seconds or np.any(np.diff(route[:, 0]) <= 0):
        raise ValueError('route must increase in time from 0 to clip duration')
    if np.linalg.norm(route[-1, 1:]-route[0, 1:]) <= 1e-6:
        raise ValueError('route must traverse to a different endpoint')
    if not inside(field, route[:, 1:]).all():
        raise ValueError('route lies outside terrain')
    if not isinstance(spec['contacts'], dict) or len(spec['contacts']) < 2:
        raise ValueError('walking trial needs at least two declared sole contacts')
    for name, foot in spec['contacts'].items():
        if not isinstance(name, str) or not name or set(foot) != {'region', 'stances'}:
            raise ValueError('each named contact needs region and stances')
        region(foot['region'])
        intervals = np.asarray(foot['stances'], float)
        if intervals.ndim != 2 or intervals.shape[1] != 2 or len(intervals) < 2 or not np.isfinite(intervals).all():
            raise ValueError(f'{name}: at least two finite stance intervals required')
        if (intervals < 0).any() or (intervals > seconds).any() or (intervals[:, 1] <= intervals[:, 0]).any():
            raise ValueError(f'{name}: invalid stance times')
        if (intervals[1:, 0] <= intervals[:-1, 1]).any():
            raise ValueError(f'{name}: stances must be ordered with a swing between them')
    criteria = spec['criteria']
    if set(criteria) != {k+'_'+v for k, v in LIMITS.items()}:
        raise ValueError('criteria must explicitly state every terrain limit: '+', '.join(k+'_'+v for k,v in LIMITS.items()))
    for key, value in criteria.items():
        finite(value, key, positive=True)
    if criteria['support_fraction_min'] > 1:
        raise ValueError('support_fraction_min must be at most one')


def inside(field, xy):
    xy = np.asarray(xy)
    return ((xy[..., 0] >= field['x'][0]) & (xy[..., 0] <= field['x'][-1]) &
            (xy[..., 1] >= field['y'][0]) & (xy[..., 1] <= field['y'][-1]))


def height(field, xy):
    """Piecewise planar grid, diagonal (x0,y0)--(x1,y1). Outside is NaN."""
    xy = np.asarray(xy, float)
    xs, ys, z = np.asarray(field['x']), np.asarray(field['y']), np.asarray(field['heights'])
    i = np.clip(np.searchsorted(xs, xy[..., 0], side='right')-1, 0, len(xs)-2)
    j = np.clip(np.searchsorted(ys, xy[..., 1], side='right')-1, 0, len(ys)-2)
    u, v = (xy[..., 0]-xs[i])/(xs[i+1]-xs[i]), (xy[..., 1]-ys[j])/(ys[j+1]-ys[j])
    low = z[j, i]*(1-u)+z[j, i+1]*(u-v)+z[j+1, i+1]*v
    high = z[j, i]*(1-v)+z[j+1, i+1]*u+z[j+1, i]*(v-u)
    return np.where(inside(field, xy), np.where(u >= v, low, high), np.nan)


def barycentrics(divisions):
    return np.array([(i/divisions, j/divisions, 1-(i+j)/divisions)
                     for i in range(divisions+1) for j in range(divisions+1-i)])


def source_regions(doc, blob, spec):
    """Resolve independent source regions to stable primitive vertex indices."""
    _, data = rig.model_data(doc, blob)
    names = [row[1] for row in data]
    if len(names) != len(set(names)):
        raise ValueError('terrain source mesh node names must be unique')
    regions = {'body': spec['body'], **{f'foot:{k}': v['region'] for k, v in spec['contacts'].items()}}
    selected, entries, offset = {}, [], 0
    for ni, name, prims, _, _ in data:
        for pi, (p, points) in enumerate(prims):
            idx = glb.accessor(doc, blob, p['indices']).astype(int).ravel() if 'indices' in p else np.arange(len(points))
            entries.append((ni, pi, idx.reshape(-1, 3)+offset))
            for key, reg in regions.items():
                if reg['node'] == name:
                    lo, hi = np.asarray(reg['box'])
                    match = np.flatnonzero(((points >= lo-1e-7) & (points <= hi+1e-7)).all(axis=1))
                    selected.setdefault(key, []).extend((match+offset).tolist())
            offset += len(points)
    for key in regions:
        if not selected.get(key):
            raise ValueError(f'{key}: source region selects no vertices; inspect rig-info and fix the frozen contract')
    # Overlapping sole declarations can otherwise pretend one foot is two.
    occupied = set()
    for key, ids in selected.items():
        if key == 'body':
            continue
        if occupied.intersection(ids):
            raise ValueError('sole regions must not overlap')
        occupied.update(ids)
    return {k: np.asarray(v, int) for k, v in selected.items()}, entries


def posed_vertices(doc, blob, entries, clip, t):
    world = skin.sample(doc, blob, clip, float(t))
    return np.concatenate([rig.to_blender(skin.primitive_positions(
        doc, blob, ni, doc['meshes'][doc['nodes'][ni]['mesh']]['primitives'][pi], world)[0])
        for ni, pi, _ in entries])


def sample_times(spec):
    duration = spec['rig']['clips'][spec['clip']]['seconds']
    times = list(np.linspace(0, duration, math.ceil(duration*spec['sampling']['hz'])+1))
    times.extend(row[0] for row in spec['route'])
    for foot in spec['contacts'].values():
        times.extend(t for interval in foot['stances'] for t in interval)
        for a, b in zip(foot['stances'][:-1], foot['stances'][1:]):
            times.append((a[1]+b[0])/2)
    return np.unique(times)


def check(source, animated, spec):
    contract(spec)
    sd, sb = glb.read_glb(source)
    if sd.get('skins') or sd.get('animations'):
        raise ValueError('terrain source must be a frozen static GLB')
    base = rig.check(source, animated, spec['rig'])
    selection, entries = source_regions(sd, sb, spec)
    doc, blob = glb.read_glb(animated)
    if [n for n, _ in glb.mesh_nodes(doc)] != [n for n, _ in glb.mesh_nodes(sd)]:
        raise ValueError('terrain animation must preserve source mesh node identities')
    # Do not measure invalid topology using source indices.
    if base['issues']:
        return {'version': VERSION, 'eligible': False, 'score': None, 'issues': base['issues'], 'rig': base,
                'hashes': {'source': digest(source), 'animation': digest(animated), 'contract': spec_digest(spec),
                           'measurement': measurement_digest()}}
    faces = np.concatenate([entry[2] for entry in entries])
    bary = barycentrics(spec['sampling']['surface_divisions'])
    times = sample_times(spec)
    field, route = spec['terrain'], np.asarray(spec['route'])
    measurements = {'penetration_mm': 0., 'path_error_mm': 0., 'stance_gap_mm': 0., 'slip_mm': 0.}
    foot_tracks = {name: [] for name in spec['contacts']}
    body_track, frames, supported, outside_count = [], [], [], 0
    worst = {}
    for t in times:
        pos = posed_vertices(doc, blob, entries, spec['clip'], t)
        if not np.isfinite(pos).all():
            raise ValueError('exported pose contains nonfinite coordinates')
        surface = np.einsum('kj,tjd->tkd', bary, pos[faces]).reshape(-1, 3)
        gaps = surface[:, 2]-height(field, surface[:, :2])
        outside_count += int(np.isnan(gaps).sum())
        penetration = max(0., -float(np.nanmin(gaps))) * 1000 if np.isfinite(gaps).any() else 0.
        body = pos[selection['body']].mean(axis=0)
        target = np.array([np.interp(t, route[:, 0], route[:, k]) for k in (1, 2)])
        error = float(np.linalg.norm(body[:2]-target))*1000
        body_track.append(body.tolist())
        for key, value in [('penetration_mm', penetration), ('path_error_mm', error)]:
            if value >= measurements[key]:
                measurements[key], worst[key] = value, float(t)
        frame_feet, support = {}, False
        for name, foot in spec['contacts'].items():
            points = pos[selection['foot:'+name]]
            gap = points[:, 2]-height(field, points[:, :2])
            foot_tracks[name].append(points)
            stance = any(a-1e-9 <= t <= b+1e-9 for a, b in foot['stances'])
            valid = np.isfinite(gap).all()
            absolute = float(np.max(np.abs(gap)))*1000 if valid else None
            if stance and valid:
                if absolute >= measurements['stance_gap_mm']:
                    measurements['stance_gap_mm'], worst['stance_gap_mm'] = absolute, {'time': float(t), 'foot': name}
                support |= absolute <= spec['criteria']['stance_gap_mm_max']
            frame_feet[name] = {'stance': stance, 'gap_mm': absolute,
                                'lowest_clearance_mm': float(np.min(gap))*1000 if valid else None}
        supported.append(bool(support))
        frames.append({'time': float(t), 'penetration_mm': penetration, 'path_error_mm': error, 'feet': frame_feet})
    feet = {}
    all_lifts = []
    for name, foot in spec['contacts'].items():
        track = np.asarray(foot_tracks[name])
        slips, lifts = [], []
        for a, b in foot['stances']:
            indices = np.flatnonzero((times >= a-1e-9) & (times <= b+1e-9))
            # Maximum horizontal movement of any fixed sole vertex, not centroid cancellation.
            distances = np.linalg.norm(track[indices, :, :2]-track[indices[0], :, :2], axis=2)*1000
            slips.append(float(distances.max()))
        for first, second in zip(foot['stances'][:-1], foot['stances'][1:]):
            indices = np.flatnonzero((times > first[1]+1e-9) & (times < second[0]-1e-9))
            values = [frames[i]['feet'][name]['lowest_clearance_mm'] for i in indices]
            lifts.append(max((v for v in values if v is not None), default=0.))
        feet[name] = {'stance_slip_mm': slips, 'swing_peak_clearance_mm': lifts}
        if max(slips) >= measurements['slip_mm']:
            measurements['slip_mm'], worst['slip_mm'] = max(slips), name
        all_lifts.extend(lifts)
    # Duration-weighted support prevents tiny authored intervals from adding many favorable samples.
    measurements['support_fraction'] = float(np.sum(np.diff(times)*(np.array(supported[:-1]) & np.array(supported[1:]))) / (times[-1]-times[0]))
    measurements['swing_clearance_mm'] = min(all_lifts)
    issues, gates, terms = [], {}, {}
    if outside_count:
        issues.append(f'{outside_count} surface samples outside the finite terrain; extend the course or correct the route/motion')
    for key, direction in LIMITS.items():
        value, limit = measurements[key], spec['criteria'][key+'_'+direction]
        passed = value <= limit if direction == 'max' else value >= limit
        gates[key] = {'value': value, 'limit': limit, 'direction': direction, 'passed': bool(passed)}
        terms[key] = max(0., 1-value/(2*limit)) if direction == 'max' else min(1., value/limit)
        if not passed:
            issues.append(f'{key}: {value:.6g}, needs {direction} {limit:.6g}')
    score = float(np.mean(list(terms.values())))
    return {'version': VERSION, 'eligible': not issues, 'score': score if not issues else None,
            'diagnostic_score': score, 'score_terms': terms, 'issues': issues, 'gates': gates,
            'measurements': measurements, 'worst': worst, 'feet': feet, 'frames': frames,
            'body_track': body_track, 'outside_samples': outside_count, 'sample_count': len(times),
            'surface_samples_per_frame': len(surface), 'rig': base,
            'hashes': {'source': digest(source), 'animation': digest(animated), 'contract': spec_digest(spec),
                       'measurement': measurement_digest()}}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec_digest(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest()


def measurement_digest():
    """Distinguish local measurement edits even before a release/commit is made."""
    result = hashlib.sha256()
    for path in [Path(__file__), *(Path(module.__file__) for module in (rig, skin, glb, render))]:
        result.update(path.name.encode())
        result.update(path.read_bytes())
    result.update(np.__version__.encode())
    return result.hexdigest()


def build(source, script, spec, out, failed=None):
    """Deliver only if both the original rig and terrain contracts pass."""
    contract(spec)
    source, script, out = Path(source), Path(script), Path(out)
    paths = [source.resolve(), script.resolve(), out.resolve()]
    if failed is not None:
        paths.append(Path(failed).resolve())
    if len(paths) != len(set(paths)):
        raise ValueError('source, script, output and failed output must be distinct')
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='terrain-', dir=out.parent) as temp:
        candidate = Path(temp)/'candidate.glb'
        try:
            rig.build(source, script, spec['rig'], candidate, failed=candidate)
        except rig.ContractError:
            pass  # The full check below reports the original rig refusal too.
        report = check(source, candidate, spec)
        report.update(contract=copy.deepcopy(spec), script_sha256=digest(script))
        destination = Path(failed) if report['issues'] and failed is not None else out if not report['issues'] else None
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            candidate.replace(destination)
            destination.with_suffix('.terrain.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        return report


def info(source, spec):
    contract(spec)
    doc, blob = glb.read_glb(source)
    selections, _ = source_regions(doc, blob, spec)
    lines = ['ROUGH-TERRAIN TRIAL v1',
             'Use the ordinary rig script API: skeleton(model), weights(name, position, piece), pose(clip, t).',
             'Coordinates: Blender metres, +Z up, +Y forward. Root travel is world travel.',
             'Author the named non-loop clip. Keep geometry frozen; place feet in world space during each stance.',
             'Use the frozen terrain grid, route and stance schedule below. Do not tune the contract to an attempt.',
             'Check heel/toe contact, swing clearance and reach. Inspect oblique and side views between keys.',
             'Score is diagnostic agreement with this contract, not naturalness, balance or continuous collision proof.',
             'Selected source vertices: '+', '.join(f'{k}={len(v)}' for k, v in selections.items()),
             json.dumps(spec, indent=2)]
    return '\n'.join(lines)
