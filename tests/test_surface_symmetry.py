"""Fast deterministic checks for the experimental surface-symmetry diagnostic."""
import numpy as np
import pytest

from mesh_jig.surface_symmetry import _BVH, _point_triangle_distance2, mirrored_surface_agreement


def rectangle(x, *, y0=0.0, y1=1.0, z0=0.0, z1=1.0):
    v = np.array([[x, y0, z0], [x, y1, z0], [x, y1, z1], [x, y0, z1]], dtype=float)
    return v[[[0, 1, 2], [0, 2, 3]]]


def paired(*, left_x=-1.0, right_x=1.0, left_width=1.0, right_width=1.0):
    left = rectangle(left_x, y1=left_width)
    right = rectangle(right_x, y1=right_width)
    return np.concatenate([left, right])


def test_mirrored_open_sheets_agree_with_area_weighted_bidirectional_reading():
    result = mirrored_surface_agreement(paired(), max_samples=128)
    assert result["status"] == "available"
    assert result["agreement"] == 1.0
    assert result["left_to_right"] == result["right_to_left"] == 1.0
    assert result["tolerance_mm"] == 1.0
    assert result["sample_count"] == 128
    assert result["left_area_m2"] == pytest.approx(1.0)


def test_surface_distance_detects_a_moved_sheet_and_an_absent_side_is_unavailable():
    moved = paired(left_x=-1.02)
    result = mirrored_surface_agreement(moved, max_samples=64)
    assert result["status"] == "available"
    assert result["agreement"] == 0.0
    assert result["left_to_right"] == 0.0
    assert result["right_to_left"] == 0.0
    assert mirrored_surface_agreement(rectangle(-1.0))["status"] == "unavailable"


def test_central_plane_sheet_cannot_hide_an_asymmetric_pair():
    asym = paired(left_x=-1.02)
    huge_centre = rectangle(0.0, y0=-100.0, y1=100.0, z0=-100.0, z1=100.0)
    base = mirrored_surface_agreement(asym, max_samples=256)
    with_centre = mirrored_surface_agreement(np.concatenate([asym, huge_centre]), max_samples=256)
    assert with_centre["agreement"] == base["agreement"] == 0.0
    assert mirrored_surface_agreement(huge_centre)["status"] == "unavailable"


def test_large_matching_surfaces_can_dominate_a_small_asymmetric_feature():
    matched = paired()
    small_left_feature = rectangle(-2.0, y1=0.01, z1=0.01)
    result = mirrored_surface_agreement(np.concatenate([matched, small_left_feature]), max_samples=4096)
    assert result["status"] == "available"
    assert result["agreement"] > 0.99
    assert "all triangle area" in result["approximation"]


def test_names_order_winding_and_vertex_splitting_are_irrelevant_to_geometry():
    faces = paired()
    base = mirrored_surface_agreement(faces, max_samples=90)
    reordered = mirrored_surface_agreement(faces[::-1, ::-1], max_samples=90)
    duplicated = mirrored_surface_agreement(np.concatenate([faces, faces]), max_samples=90)
    assert reordered["agreement"] == base["agreement"] == 1.0
    assert duplicated["agreement"] == base["agreement"]
    assert reordered["sample_count"] == base["sample_count"]


def test_partial_match_is_exactly_stable_under_face_cyclic_and_reversed_orders():
    faces = paired(left_width=1.25, right_width=1.0)
    variants = [faces, faces[::-1], faces[:, [1, 2, 0]], faces[:, ::-1], np.roll(faces, 1, axis=1)]
    readings = [mirrored_surface_agreement(v, max_samples=128) for v in variants]
    assert all(reading == readings[0] for reading in readings[1:])
    reading = readings[0]
    assert reading["left_to_right"] == 0.8125
    assert reading["right_to_left"] > reading["left_to_right"]
    assert reading["agreement"] == reading["left_to_right"]
    assert reading["agreement"] < (reading["left_to_right"] + reading["right_to_left"]) / 2


def test_triangle_crossing_x_zero_is_clipped_stably_and_compares_symmetrically():
    crossing = np.array([[[-1., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    variants = [crossing, crossing[:, [1, 2, 0]], crossing[:, ::-1]]
    readings = [mirrored_surface_agreement(v, max_samples=128) for v in variants]
    assert all(reading == readings[0] for reading in readings[1:])
    assert readings[0]["status"] == "available"
    assert readings[0]["left_area_m2"] == pytest.approx(readings[0]["right_area_m2"])
    assert readings[0]["agreement"] == 1.0


def test_one_sided_coplanar_refinement_preserves_surface_agreement():
    left = rectangle(-1.0)
    right = rectangle(1.0)
    refined = []
    for face in left:
        a, b, c = face
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        refined.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
    original = mirrored_surface_agreement(np.concatenate([left, right]), max_samples=512)
    subdivided = mirrored_surface_agreement(np.concatenate([np.asarray(refined), right]), max_samples=512)
    assert original["agreement"] == 1.0
    assert subdivided["agreement"] == 1.0


def test_degenerate_invalid_and_no_area_inputs_are_never_perfect():
    degenerate = np.zeros((2, 3, 3), dtype=float)
    invalid = paired()
    invalid[0, 0, 0] = np.nan
    assert mirrored_surface_agreement(degenerate)["status"] == "unavailable"
    assert mirrored_surface_agreement(np.zeros((0, 3, 3)))["agreement"] is None
    assert mirrored_surface_agreement(invalid)["status"] == "unavailable"
    assert mirrored_surface_agreement(paired(), max_samples=0)["status"] == "unavailable"
    assert mirrored_surface_agreement(paired(), max_samples=1)["status"] == "unavailable"
    assert mirrored_surface_agreement(paired(), tolerance_mm=None)["reason"] == "invalid_parameters"
    assert mirrored_surface_agreement(paired(), tolerance_mm="1 mm")["reason"] == "invalid_parameters"
    assert mirrored_surface_agreement(paired(), tolerance_mm=True)["reason"] == "invalid_parameters"
    assert mirrored_surface_agreement(paired(), max_samples=True)["reason"] == "invalid_parameters"
    assert mirrored_surface_agreement(paired(), leaf_size=True)["reason"] == "invalid_parameters"
    assert mirrored_surface_agreement(rectangle(-1.0))["reason"] == "missing_right_area"
    degenerate_both_sides = np.array([[[-1., 0., 0.]] * 3, [[1., 0., 0.]] * 3])
    assert mirrored_surface_agreement(degenerate_both_sides)["reason"] == "no_left_area"


def test_an_enormous_integer_coordinate_returns_unavailable_instead_of_raising():
    malformed = np.array([[[10 ** 1000, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=object)
    result = mirrored_surface_agreement(malformed)
    assert result["status"] == "unavailable"
    assert result["reason"] == "invalid_input"


def test_point_to_triangle_distance_has_analytic_face_edge_vertex_and_plane_answers():
    tri = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    assert _point_triangle_distance2(np.array([0.25, 0.25, 2.]), tri) == pytest.approx(4.0)
    assert _point_triangle_distance2(np.array([0.5, -1., 0.]), tri) == pytest.approx(1.0)
    assert _point_triangle_distance2(np.array([-1., -1., 0.]), tri) == pytest.approx(2.0)
    assert _point_triangle_distance2(np.array([0.2, 0.3, 0.]), tri) == pytest.approx(0.0)


def test_bvh_nearest_is_true_point_to_triangle_distance():
    tri = rectangle(1.0)
    tree = _BVH(tri, leaf_size=1)
    # A point within the face plane's Y/Z extent is 2 cm from it, independent of its vertices.
    assert tree.nearest2(np.array([1.02, 0.5, 0.5])) == pytest.approx(0.02 ** 2)


def test_bvh_matches_brute_force_on_deterministic_mixed_geometry():
    rng = np.random.default_rng(20261026)
    triangles = rng.normal(size=(64, 3, 3))
    triangles[:, 1:] += triangles[:, :1]
    points = rng.normal(size=(40, 3))
    tree = _BVH(triangles, leaf_size=5)
    for point in points:
        brute = min(_point_triangle_distance2(point, tri) for tri in triangles)
        assert tree.nearest2(point) == pytest.approx(brute, rel=1e-11, abs=1e-12)


def test_bvh_reports_sparse_triangle_distance_work_without_a_sample_triangle_matrix():
    triangles = np.concatenate([rectangle(-1.0), rectangle(1.0)])
    reading = mirrored_surface_agreement(triangles, max_samples=128)
    pairs_in_dense_matrix = reading["sample_count"] * len(triangles)
    assert 0 < reading["triangle_distance_tests"] < pairs_in_dense_matrix


def test_sampling_and_result_are_repeatable():
    faces = paired(left_width=1.25, right_width=1.25)
    a = mirrored_surface_agreement(faces, max_samples=257)
    b = mirrored_surface_agreement(faces, max_samples=257)
    assert a == b
    assert a["sample_count"] <= 257
