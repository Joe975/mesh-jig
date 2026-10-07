"""mesh_jig.palette: reading a palette off the reference views, and moving one to the reference's own paint."""
from __future__ import annotations

import json

import numpy as np
from PIL import Image

from mesh_jig import cli, palette

LIT, SHADE, TEAL, SPOT = (230, 110, 35), (150, 70, 20), (37, 75, 80), (80, 130, 60)
DULL = {"orange": "#d07829", "teal": "#254b50", "green": "#507c38"}


def painted(path):
    """An orange half with a quarter of its pixels in painted shadow, a flat teal half and a green spot too small to
    read a paint from, on a flat ground."""
    a = np.full((200, 200, 3), 245, dtype=np.uint8)
    a[20:180, 20:100] = LIT
    a[20:180, 20:100:4] = SHADE
    a[20:180, 100:180] = TEAL
    a[90:96, 137:143] = SPOT
    Image.fromarray(a).save(path)
    return str(path)


def test_refine_moves_a_colour_to_the_lit_paint_the_reference_shows_and_keeps_one_it_cannot_read(tmp_path):
    found, cells = palette.refine({"front": painted(tmp_path / "front.png")}, DULL)
    assert list(found) == list(DULL), "the names and their order are the project's"
    assert np.abs(np.array(palette.hex_rgb(found["orange"])) - LIT).max() <= 3, "the lit side, not the mean with shadow"
    assert np.abs(np.array(palette.hex_rgb(found["teal"])) - TEAL).max() <= 2
    assert found["green"] == DULL["green"] and cells["green"] < palette.REFINE_MIN_CELLS
    assert cells["orange"] > 300 and cells["teal"] > 300
    assert palette.refine({"front": painted(tmp_path / "front.png")}, DULL) == (found, cells), "deterministic"
    assert palette.refine({}, DULL)[0] == DULL and palette.refine({"front": painted(tmp_path / "f.png")}, {}) == ({}, {})


def test_the_palette_command_refines_and_writes_only_when_asked(tmp_path, capsys):
    (tmp_path / "refs").mkdir()
    painted(tmp_path / "refs" / "front.png")
    jig = tmp_path / "jig.json"
    jig.write_text(json.dumps({"palette": DULL, "model": {"glb": "x.glb", "shots": [
        {"name": "front", "cam_pos": [0, 0, -3], "cam_target": [0, 0, 0], "ortho": 1.2}]}}))
    assert cli.main(["palette", str(tmp_path), "--refine"]) == 0
    out = capsys.readouterr().out
    assert "orange: #d07829 -> #e" in out and "would move" in out and "too few cells: kept" in out
    assert "Nothing is written" in out and json.loads(jig.read_text())["palette"] == DULL
    assert cli.main(["palette", str(tmp_path), "--refine", "--write"]) == 2, "a write names the colours it moves"
    assert cli.main(["palette", str(tmp_path), "--refine", "purple"]) == 2
    capsys.readouterr()
    assert cli.main(["palette", str(tmp_path), "--refine", "orange", "--write"]) == 0
    out = capsys.readouterr().out
    assert "not comparable" in out and "teal: #254b50 ->" in out and "not named: kept" in out
    written = json.loads(jig.read_text())["palette"]
    assert written["orange"].startswith("#e") and written["teal"] == DULL["teal"] and written["green"] == DULL["green"]
    jig.write_text(json.dumps({"model": {"glb": "x.glb"}}))
    assert cli.main(["palette", str(tmp_path), "--refine"]) == 2
