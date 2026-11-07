"""mesh_jig.articulation: how much of a part a rig will bend stands free of the rest, on figures with known answers."""
import numpy as np
import pytest

from mesh_jig import articulation

FACES = [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1),
         (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)]


def box(lo, hi):
    """A closed box as its 12 triangles, Blender metres."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                  (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], float)
    return v[np.array(FACES)]


def figure(arm_x=0.3, legs_apart=True):
    """A 1 m figure: trunk 0.4 wide over 0.5 to 1.0 m, legs to the ground, arms 0.08 thick from 0.9 down to 0.5 m
    centred at +-arm_x (the trunk's side is at 0.2, so an arm at 0.3 leaves a 6 cm gap)."""
    leg = 0.1 if legs_apart else 0.01
    return np.concatenate([
        box((-0.2, -0.1, 0.5), (0.2, 0.1, 1.0)),
        box((-leg - 0.08, -0.05, 0.0), (-leg, 0.05, 0.5)), box((leg, -0.05, 0.0), (leg + 0.08, 0.05, 0.5)),
        box((-arm_x - 0.04, -0.04, 0.5), (-arm_x + 0.04, 0.04, 0.9)),
        box((arm_x - 0.04, -0.04, 0.5), (arm_x + 0.04, 0.04, 0.9))])


def arm(x=-0.3, **kw):
    return {"name": "left-arm", "path_m": [[x, 0, 0.9], [x, 0, 0.5]], "radius_m": 0.06, **kw}


def test_an_arm_that_hangs_clear_of_the_flank_is_free_all_its_length():
    check = articulation.measure(figure(arm_x=0.3), arm())
    assert check["passed"] and check["free_m"] == pytest.approx(0.4) and check["free_fraction"] == pytest.approx(1.0)
    assert check["joined_m"] == 0 and check["missing_m"] == 0
    assert articulation.describe(check) == "free of the rest for 0.40 m of its 0.40 m (100%)"


def test_an_arm_modelled_into_the_flank_is_not_free_and_says_where():
    check = articulation.measure(figure(arm_x=0.22), arm(x=-0.22))       # sunk 2 cm into the trunk
    assert not check["passed"] and check["free_m"] == 0 and check["joined_m"] == pytest.approx(0.4)
    assert check["stops_at_m"] == [-0.22, 0.0, 0.505] and check["stops_along_m"] == 0.395
    assert check["cuts"] == "joined 0.40"
    said = articulation.describe(check)
    assert said.startswith("free of the rest for 0.00 m of its 0.40 m (0%, needs 50%): the free run from the tip "
                           "stops 0.40 m along it from its root, at (-0.220, +0.000, +0.505) m in your model's own "
                           "coordinates")
    assert said.endswith("(root to tip: joined 0.40 m)")


def test_an_arm_nearer_the_flank_than_the_gap_is_joined_and_a_smaller_gap_frees_it():
    near = figure(arm_x=0.26)                                            # a 2 cm gap
    assert articulation.measure(near, arm(x=-0.26))["free_m"] == 0
    assert articulation.measure(near, arm(x=-0.26, min_gap_m=0.01))["free_m"] == pytest.approx(0.4)


def test_an_arm_joined_at_the_top_counts_only_what_hangs_free_below():
    tris = np.concatenate([figure(arm_x=0.3), box((-0.34, -0.04, 0.7), (-0.2, 0.04, 0.9))])   # a web to 0.7 m
    check = articulation.measure(tris, arm(min_free=0.6))
    assert check["free_m"] == pytest.approx(0.2) and check["joined_m"] == pytest.approx(0.2)
    assert not check["passed"] and check["free_fraction"] == pytest.approx(0.5)


def test_a_path_that_runs_past_the_tip_is_read_from_where_the_part_ends():
    check = articulation.measure(figure(arm_x=0.3), {**arm(), "path_m": [[-0.3, 0, 0.9], [-0.3, 0, 0.3]]})
    assert check["missing_m"] == pytest.approx(0.2) and check["free_m"] == pytest.approx(0.4)
    assert check["free_fraction"] == pytest.approx(0.4 / 0.6, abs=0.01)
    assert "no surface within 0.06 m of it" in articulation.describe(dict(check, passed=False))


def test_legs_pressed_together_are_one_island_and_not_free():
    leg = {"name": "left-leg", "path_m": [[-0.14, 0, 0.5], [-0.14, 0, 0.0]], "radius_m": 0.06}
    assert articulation.measure(figure(), leg)["passed"]
    pressed = {**leg, "path_m": [[-0.05, 0, 0.5], [-0.05, 0, 0.0]]}            # a 2 cm gap between them
    other = {**pressed, "name": "right-leg", "path_m": [[0.05, 0, 0.5], [0.05, 0, 0.0]]}
    assert articulation.measure(figure(legs_apart=False), pressed, [pressed, other])["free_m"] == 0
    apart = {**other, "path_m": [[0.14, 0, 0.5], [0.14, 0, 0.0]]}
    assert articulation.measure(figure(), leg, [leg, apart])["passed"]


def test_z_is_measured_above_the_lowest_point_so_a_build_placed_lower_reads_the_same():
    lowered = figure(arm_x=0.3) - np.array([0, 0, 0.6])
    assert articulation.measure(lowered, arm())["free_m"] == pytest.approx(0.4)


def test_a_bent_path_is_walked_along_each_segment():
    # a tail along -Y from the trunk's back, then up: two segments, free all the way
    tris = np.concatenate([box((-0.2, -0.1, 0.5), (0.2, 0.1, 1.0)),
                           box((-0.03, -0.5, 0.55), (0.03, -0.1, 0.61)),
                           box((-0.03, -0.56, 0.55), (0.03, -0.5, 0.85))])
    # Z above the lowest point, the trunk's bottom at 0.5 m
    tail = {"name": "tail", "path_m": [[0, -0.1, 0.08], [0, -0.53, 0.08], [0, -0.53, 0.35]], "radius_m": 0.05,
            "min_gap_m": 0.02}
    check = articulation.measure(tris, tail)
    assert check["length_m"] == pytest.approx(0.70)
    assert check["free_fraction"] > 0.85 and check["passed"]


@pytest.mark.parametrize("item, says", [
    ({"name": "a", "path_m": [[0, 0, 0]], "radius_m": 0.1}, "two or more"),
    ({"name": "a", "path_m": [[0, 0, 0], [0, 0, 0.005]], "radius_m": 0.1}, "shorter than"),
    ({"name": "a", "path_m": [[0, 0, 0], [0, 0, 1]], "radius_m": 0}, "radius_m"),
    ({"name": "a", "path_m": [[0, 0, 0], [0, 0, 1]], "radius_m": 0.1, "min_free": 2}, "min_free"),
    ({"name": "a", "path_m": [[0, 0, 0], [0, 0, 1]], "radius_m": 0.1, "kind": "hinge"}, "not measured yet"),
    ({"name": "a", "path_m": [[0, 0, 0], [0, 0, 1]], "radius_m": 0.1, "colour": 1}, "unknown setting"),
])
def test_validate_names_what_is_wrong(item, says):
    with pytest.raises(ValueError, match=says):
        articulation.validate([item])
    with pytest.raises(ValueError, match="unique name"):
        articulation.validate([arm(), arm()])


def test_the_brief_and_the_blind_builder_hear_of_them():
    review = {"articulations": [arm()]}
    assert articulation.brief_text(review).endswith("left-arm (free for 50% of its 0.40 m)")
    assert articulation.brief_text({}) == ""
    failed = articulation.measure(figure(arm_x=0.22), arm(x=-0.22))
    assert articulation.failed_lines([failed, {"name": "landmark:x", "passed": False}]) == [
        "gate articulation:left-arm: FAILED - " + articulation.describe(failed)]


def test_an_arm_bent_at_the_elbow_is_free_round_the_bend():
    # the upper arm hangs from 0.9 to 0.7 m, the forearm reaches forward (+Y) 0.25 m from the elbow
    tris = np.concatenate([box((-0.2, -0.1, 0.5), (0.2, 0.1, 1.0)),
                           box((-0.34, -0.04, 0.66), (-0.26, 0.04, 0.9)),
                           box((-0.34, 0.04, 0.66), (-0.26, 0.29, 0.74))])
    elbow = {"name": "left-arm", "path_m": [[-0.3, 0, 0.4], [-0.3, 0, 0.2], [-0.3, 0.25, 0.2]], "radius_m": 0.06}
    check = articulation.measure(tris, elbow)
    assert check["passed"] and check["free_fraction"] > 0.9, check


def test_review_measures_articulations_on_a_glb_with_no_parts_declared(tmp_path):
    from glbkit import write_glb
    from mesh_jig import review

    def part(lo, hi):           # a box in Blender metres, written in glTF axes (x, z, -y)
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        size = (hi - lo)[[0, 2, 1]]
        centre = ((lo + hi) / 2)[[0, 2, 1]] * np.array([1, 1, -1])
        from glbkit import box as gbox
        return gbox(tuple(size), tuple(centre))
    path = write_glb(tmp_path / "f.glb", [("Trunk", part((-0.2, -0.1, 0.5), (0.2, 0.1, 1.0)), (0, 0, 0)),
                                          ("Legs", part((-0.1, -0.05, 0.0), (0.1, 0.05, 0.5)), (0, 0, 0)),
                                          ("Arm", part((-0.34, -0.04, 0.5), (-0.26, 0.04, 0.9)), (0, 0, 0))])
    out = review.measure(path, {"glb": "f.glb"}, {"articulations": [arm()]}, {}, {})
    (check,) = out["checks"]
    assert check["name"] == "articulation:left-arm" and check["passed"]
    assert review.describe(check) == "free of the rest for 0.40 m of its 0.40 m (100%)"


def test_a_part_thicker_than_its_radius_is_found_from_a_path_down_its_middle():
    # a leg 0.2 m across: a path down its centre is 0.1 m from the outline, beyond a 0.05 m radius
    tris = np.concatenate([box((-0.3, -0.15, 0.5), (0.3, 0.15, 1.0)), box((-0.25, -0.1, 0.0), (-0.05, 0.1, 0.5))])
    check = articulation.measure(tris, {"name": "leg", "path_m": [[-0.15, 0, 0.5], [-0.15, 0, 0.0]], "radius_m": 0.05,
                                        "min_gap_m": 0.02})
    assert check["missing_m"] == 0



def test_a_part_that_curls_away_from_its_path_is_free_and_not_joined():
    # a tail that leaves the trunk along its path and then curls 0.3 m up, away from it, with nothing near it: the cuts
    # across the curl read it lengthwise, an island far off the path but no wider than a free part could be
    tris = np.concatenate([box((-0.2, -0.1, 0.5), (0.2, 0.1, 1.0)),
                           box((-0.03, -0.43, 0.55), (0.03, -0.1, 0.61)),
                           box((-0.03, -0.49, 0.55), (0.03, -0.43, 0.85))])
    tail = {"name": "tail", "path_m": [[0, -0.1, 0.08], [0, -0.55, 0.08]], "radius_m": 0.06, "min_gap_m": 0.02}
    check = articulation.measure(tris, tail)
    assert check["passed"] and check["free_m"] > 0.3, check
    assert check["off_path_m"] > 0, "it says the part ran off its path"
    fused = np.concatenate([tris, box((-0.2, -0.49, 0.5), (0.2, -0.43, 0.9))])   # the curl run into a wide plate
    assert articulation.measure(fused, tail)["free_m"] < check["free_m"]


def test_a_limb_sunk_inside_the_body_is_joined_though_no_surface_is_near_it():
    # a rod running down inside a thick block, its surface 0.1 m from the block's everywhere: a nested ring in each cut
    tris = np.concatenate([box((-0.3, -0.3, 0.0), (0.3, 0.3, 1.0)), box((-0.03, -0.03, 0.2), (0.03, 0.03, 0.8))])
    rod = {"name": "rod", "path_m": [[0, 0, 0.8], [0, 0, 0.2]], "radius_m": 0.05}
    assert articulation.measure(tris, rod)["free_m"] == 0
    # the same rod with a small cap of its own round its end (a paw) is still free where it hangs clear
    free = np.concatenate([box((-0.3, -0.3, 0.8), (0.3, 0.3, 1.0)), box((-0.03, -0.03, 0.2), (0.03, 0.03, 0.8)),
                           box((-0.06, -0.06, 0.2), (0.06, 0.06, 0.3))])
    lifted = {**rod, "path_m": [[0, 0, 0.6], [0, 0, 0.0]]}          # Z above its lowest point, the cap at 0.2
    assert articulation.measure(free, lifted)["free_m"] > 0.5
