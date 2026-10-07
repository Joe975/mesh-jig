"""mesh_jig.texel_bake: the arithmetic of a texture the model script paints. No Blender."""
from __future__ import annotations

import numpy as np
import pytest

from mesh_jig import texel_bake

# one quad as two triangles, laid over the left half of the atlas; on the model it is 2 m wide and 1 m tall at z = 5
UV = np.array([[[0.0, 0.0], [0.5, 0.0], [0.5, 1.0]], [[0.0, 0.0], [0.5, 1.0], [0.0, 1.0]]])
POS = np.array([[[0, 0, 5], [2, 0, 5], [2, 1, 5]], [[0, 0, 5], [2, 1, 5], [0, 1, 5]]], dtype=float)
NRM = np.tile(np.array([0.0, 0.0, 3.0]), (2, 3, 1))                      # not unit length: the bake normalises


def test_each_texel_is_given_the_point_of_the_model_it_shows():
    covered, point, normal, tri = texel_bake.rasterise(UV, POS, NRM, 8)
    assert covered[:, :4].all() and not covered[:, 4:].any(), "the left half of the atlas, and only it"
    assert (tri[:, 4:] == -1).all() and set(np.unique(tri[:, :4])) == {0, 1}
    # the texel in column 1, row 6 has its centre at u = 1.5/8, v = 6.5/8: a quarter... of the half, so x = 0.75 m
    assert point[6, 1] == pytest.approx([0.75, 6.5 / 8, 5.0])
    assert point[0, 0] == pytest.approx([0.25, 0.5 / 8, 5.0]), "row 0 is v = 0, the bottom of the image"
    assert normal[3, 2] == pytest.approx([0, 0, 1])


def test_a_triangle_smaller_than_a_texel_still_gets_one():
    speck = np.array([[[0.80, 0.80], [0.81, 0.80], [0.80, 0.81]]])
    covered, point, _n, tri = texel_bake.rasterise(speck, POS[:1], NRM[:1], 8)
    assert covered.sum() == 1 and tri[6, 6] == 0, "the texel its centroid falls in"
    assert point[6, 6] == pytest.approx(POS[0].mean(axis=0))


def test_the_script_is_asked_once_a_mesh_about_every_texel_of_it():
    covered, point, normal, tri = texel_bake.rasterise(UV, POS, NRM, 8)
    part = np.where(tri >= 0, np.array([0, 1])[np.maximum(tri, 0)], -1)     # each triangle its own mesh
    asked = []

    def paint(name, position, facing):
        asked.append((name, len(position)))
        red = (position[:, 0] > 1.0).astype(float)                         # the right-hand metre of the quad is red
        return np.stack([red, np.zeros(len(position)), 1 - red], axis=1)

    colour = texel_bake.paint(paint, ["lower", "upper"], part, covered, point, normal)
    assert sorted(asked) == [("lower", int((part == 0).sum())), ("upper", int((part == 1).sum()))]
    assert colour[4, 3] == pytest.approx([1, 0, 0]) and colour[4, 0] == pytest.approx([0, 0, 1])
    assert (colour[:, 4:] == 0).all(), "nothing painted where no mesh is"
    whole = texel_bake.paint(lambda name, p, n: (0.2, 0.4, 0.6), ["lower", "upper"], part, covered, point, normal)
    assert whole[2, 2] == pytest.approx([0.2, 0.4, 0.6]), "one colour for a whole mesh"


@pytest.mark.parametrize("returned, said", [
    (lambda p: np.zeros((len(p), 4)), "it must return one (r, g, b) row per position"),
    (lambda p: np.zeros((3, 3)), "it must return one (r, g, b) row per position"),
    (lambda p: np.full((len(p), 3), np.nan), "not finite"),
    (lambda p: np.full((len(p), 3), 200.0), "sRGB 0..1"),
    (lambda p: "orange", "must return an array of numbers"),
])
def test_what_the_script_returns_is_held_to_a_colour_per_position(returned, said):
    covered, point, normal, tri = texel_bake.rasterise(UV, POS, NRM, 8)
    with pytest.raises(ValueError, match="texture\\('body'") as e:
        texel_bake.paint(lambda name, p, n: returned(p), ["body"], np.where(tri >= 0, 0, -1), covered, point, normal)
    assert said in str(e.value)


def test_islands_spread_into_the_gutter_and_stop():
    covered = np.zeros((8, 8), dtype=bool)
    covered[:, :2] = True
    colour = np.zeros((8, 8, 3))
    colour[:, :2] = (0.9, 0.4, 0.1)
    out, reached = texel_bake.dilate(colour, covered, steps=3)
    assert reached[:, :5].all() and not reached[:, 5:].any(), "three texels out, no further and not round the edge"
    assert out[3, 4] == pytest.approx([0.9, 0.4, 0.1])
    assert np.array_equal(out[:, :2], colour[:, :2]), "a painted texel keeps its colour"
    flat = texel_bake.pixels(out, reached).reshape(8, 8, 4)
    assert flat.dtype == np.float32 and (flat[..., 3] == 1).all()
    assert flat[0, 7, :3] == pytest.approx([0.9, 0.4, 0.1]), "an unreached texel takes the mean of the painted ones"
