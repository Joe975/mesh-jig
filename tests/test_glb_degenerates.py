"""Distinguish collapsed triangles from small valid triangles without moving the gate."""
import numpy as np
import pytest

from glbkit import write_glb
from mesh_jig import glb


def fixture(tmp_path, healthy=1, exact=1, tiny=1):
    # One primitive: its one-metre bounds define the existing relative cutoff.
    vertices = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0),
                         (0, 0, 0), (1, 0, 0), (2, 0, 0),
                         (0, 0, 0), (1e-8, 0, 0), (0, 1e-8, 0)], dtype=np.float32)
    faces = [(0, 1, 2)] * healthy + [(3, 4, 5)] * exact + [(6, 7, 8)] * tiny
    path = write_glb(tmp_path / "fixture.glb",
                     [("triangles", (vertices, np.array(faces, dtype=np.uint16)), (3, 0, 0))])
    return glb.stats(path)


def test_stats_separate_exact_and_small_positive_area_in_world_space(tmp_path):
    stats = fixture(tmp_path)
    assert stats["zero_area_triangles"] == 1
    assert stats["near_degenerate_triangles"] == 1
    assert stats["degenerate_triangles"] == 2
    assert stats["triangles"] == 3
    assert glb.check(stats, {}) == [
        "fixture.glb: 2 degenerate triangles of 3 (1 zero-area, 1 tiny positive-area)"]


@pytest.mark.parametrize("healthy,tiny,fails", [(198, 2, False), (197, 2, True)])
def test_tiny_positive_area_keeps_original_one_percent_gate(tmp_path, healthy, tiny, fails):
    stats = fixture(tmp_path, healthy=healthy, exact=0, tiny=tiny)
    assert stats["zero_area_triangles"] == 0
    assert stats["near_degenerate_triangles"] == tiny
    assert stats["degenerate_triangles"] == tiny
    issues = glb.check(stats, {})
    assert bool(issues) is fails
    if fails:
        assert issues == ["fixture.glb: 2 degenerate triangles of 199 (0 zero-area, 2 tiny positive-area)"]


def test_legacy_stats_remain_usable_without_inventing_an_area_breakdown(tmp_path):
    stats = fixture(tmp_path)
    stats.pop("zero_area_triangles", None)
    stats.pop("near_degenerate_triangles", None)
    assert glb.check(stats, {}) == ["fixture.glb: 2 degenerate triangles of 3"]
