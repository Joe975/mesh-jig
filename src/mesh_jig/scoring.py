"""Opt-in aggregate policy shared by all image measures. Legacy callers keep their own defaults."""
from __future__ import annotations

import math

CHARACTER_VIEWS = ("front", "side", "rear", "right", "top")


def weights(config: dict | None) -> dict[str, float] | None:
    if config is None:
        return None
    if not isinstance(config, dict) or set(config) - {"preset", "views"}:
        raise ValueError("scoring needs preset and/or views")
    preset = config.get("preset")
    if preset not in (None, "character"):
        raise ValueError("scoring.preset must be character")
    raw = config.get("views", dict.fromkeys(CHARACTER_VIEWS, 1) if preset else None)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("scoring.views must be a nonempty mapping of view names to positive weights")
    out = {}
    for view, value in raw.items():
        if not isinstance(view, str) or not view or isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("scoring.views needs named numeric weights")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("scoring weights must be finite and positive (omit excluded views)")
        out[view] = float(value)
    return out


def aggregate(views: dict, key: str, selected: dict[str, float]) -> float | None:
    missing = sorted(set(selected) - set(views))
    if missing:
        raise ValueError(f"selected scoring views missing from measures: {missing}")
    live = [(views[v][key], w) for v, w in selected.items() if v in views and views[v].get(key) is not None]
    return round(sum(v * w for v, w in live) / sum(w for _, w in live), 4) if live else None


def weak_views(views: dict, key: str, selected: dict[str, float]) -> list[dict]:
    return [{"view": v, "value": views[v][key]} for v in sorted(selected, key=lambda v: (views.get(v, {}).get(key)
            if views.get(v, {}).get(key) is not None else -1, v)) if v in views and views[v].get(key) is not None]
