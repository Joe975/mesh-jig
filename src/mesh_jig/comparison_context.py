"""Portable provenance and compatibility rules for evaluation comparisons.

Historical records cannot be signed retroactively: their complete contract and
reference inputs were not recorded. New records store only a schema, digest,
and optional target label; the digest payload is never written to an eval file.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SCHEMA = 1
FIELD = "comparison_context"
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _reference_hashes(project) -> dict[str, str]:
    return {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in sorted(project.references.items())}


def source_fingerprint(package_root: str | Path | None = None) -> str:
    """Hash package source by relative filename and bytes, independent of checkout location.

    This is a conservative compatibility boundary: any Python-source change under
    mesh_jig separates new evaluations, even when the changed module does not affect
    the particular project's measurements.
    """
    root = Path(package_root) if package_root is not None else Path(__file__).resolve().parent
    manifest = [(path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
                for path in sorted(root.rglob("*.py"))]
    return hashlib.sha256(_canonical(manifest)).hexdigest()


def signature(project, *, package_root: str | Path | None = None) -> dict:
    """Return portable provenance for the effective evaluation inputs."""
    stated = project.stated
    payload = {
        "spec": project.spec,
        "target": project.target,
        "criteria": {key: stated.get(key) for key in ("criteria", "scoring", "review", "score")},
        "palette": project.palette,
        "brief": project.brief,
        "references": _reference_hashes(project),
        "implementation": source_fingerprint(package_root),
    }
    digest = hashlib.sha256(_canonical(payload)).hexdigest()
    target = project.target
    result = {"schema": SCHEMA, "digest": digest}
    if target and target.get("name"):
        result["target"] = str(target["name"])
    return result


def classify(record: dict) -> tuple[str, object]:
    """Classify a record as a signed context, a conservative legacy bucket, or invalid.

    Missing provenance is legacy. A present but malformed or unsupported signature
    is invalid and is never treated as legacy or compatible with another record.
    """
    if not isinstance(record, dict):
        return "invalid", None
    if FIELD in record:
        value = record.get(FIELD)
        if (isinstance(value, dict) and type(value.get("schema")) is int and value.get("schema") == SCHEMA
                and isinstance(value.get("digest"), str) and _DIGEST.fullmatch(value["digest"])):
            return "signed", value["digest"]
        return "invalid", None
    if "readiness" not in record:
        return "legacy", (None, None)
    readiness = record["readiness"]
    if not isinstance(readiness, dict):
        return "invalid", None
    target, cap = readiness.get("target"), readiness.get("max_triangles")
    if target is not None and (not isinstance(target, str) or not target):
        return "invalid", None
    if cap is not None and (type(cap) is not int or cap <= 0):
        return "invalid", None
    # Preserve each known fact independently. A missing cap does not collapse
    # different explicit target names into one unknown bucket.
    return "legacy", (target, cap)


def compatible(left: dict, right: dict) -> bool:
    a_kind, a_key = classify(left)
    b_kind, b_key = classify(right)
    return a_kind == b_kind and a_kind != "invalid" and a_key == b_key


def has_provenance(record: dict) -> bool:
    return classify(record)[0] == "signed"


__all__ = ["FIELD", "SCHEMA", "signature", "source_fingerprint", "classify", "compatible", "has_provenance"]
