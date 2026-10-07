"""The MCP tools: plain functions first (no SDK needed), then the registered server when the `mcp` extra is
installed."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mesh_jig import build, evaluate, glb, mcp_server
from test_evaluate import fake_build, make_project, quick_render


def texts(reply) -> str:
    return "\n".join(x for x in reply if isinstance(x, str))


def pictures(reply) -> list[mcp_server.Picture]:
    return [x for x in reply if isinstance(x, mcp_server.Picture)]


@pytest.fixture
def proj(tmp_path, monkeypatch):
    """The known-answer project, with Blender replaced by a builder that writes the right GLB."""
    p = make_project(tmp_path)
    def built(script, out, spec, **kw):
        fake_build()(script, out, spec)
        (Path(out) / glb.script_name(spec)).write_text(Path(script).read_text())
        glb.write_report(out, spec)
        return ""
    monkeypatch.setattr(build, "build_variants", built)
    real = evaluate.evaluate
    monkeypatch.setattr(evaluate, "evaluate",
                        lambda *a, **kw: real(*a, **{**kw, "build_fn": fake_build(), "render_fn": quick_render}))
    look = evaluate.preview                  # a preview is drawn as an evaluation is, so it is faked the same way
    monkeypatch.setattr(evaluate, "preview", lambda *a, **kw: look(*a, **{**kw, "render_fn": quick_render}))
    return p


def test_brief_returns_the_contract_and_every_reference_view(proj):
    reply = mcp_server.brief(str(proj.root))
    assert "PROJECT ship" in texts(reply) and "MEASURED REFERENCE OUTLINES" in texts(reply)
    assert sorted(Path(p.path).stem for p in pictures(reply)) == ["hero", "rear", "side", "top"]
    assert pictures(reply)[0].jpeg()[:2] == b"\xff\xd8", "pictures go out as JPEG"


def test_evaluate_measures_writes_the_ledger_and_shows_each_view_beside_its_reference(proj):
    script = proj.root / "attempts" / "a01" / "ship.py"
    script.parent.mkdir(parents=True)
    script.write_text("def build():\n    pass\n")
    reply = mcp_server.evaluate(str(proj.root), str(script), note="first")
    assert "built, in contract, rendered and measured" in texts(reply)
    assert "standing: the first measured attempt here." in texts(reply), "where it stands, as `mesh-jig eval` says it"
    assert [Path(p.path).name for p in pictures(reply)] == [f"{v}_vs_reference.png" for v in ("hero", "top", "side", "rear")]
    assert "| a01 | ok |" in proj.ledger.read_text() and "first" in proj.ledger.read_text()
    look = mcp_server.preview(str(proj.root), str(script.parent), views=["top"])
    assert "top: silhouette overlap" in texts(look) and len(pictures(look)) == 1
    assert "not every scored view was drawn, so there is no score" in texts(look)
    whole = texts(mcp_server.preview(str(proj.root), str(script.parent)))
    numbers = texts(reply).split("mm, ")[1].split(", numeric score")[0]        # overlap, resemblance, zones
    assert "a free look, not an attempt (nothing is added to the ledger)" in whole
    assert numbers in whole, "the look gives the numbers the evaluation of the same build gave"
    assert "HOW YOUR COLOURS READ" in whole and "standing:" not in whole, "no other attempt to stand against"
    assert proj.ledger.read_text().count("| a01 |") == 1, "and adds nothing to the ledger"
    again = proj.root / "attempts" / "a02" / "ship.py"
    again.parent.mkdir()
    again.write_text(script.read_text())
    mcp_server.build(str(proj.root), str(again))
    assert "standing: level with a01" in texts(mcp_server.preview(str(proj.root), str(again.parent)))
    assert "a02" not in proj.ledger.read_text() and not (again.parent / "eval.json").exists()
    assert "built and in contract" in texts(mcp_server.build(str(proj.root), str(script)))


def test_paint_shows_the_atlas_and_reports_a_bad_plan(proj, tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"size": 16, "ops": [{"op": "rect", "box": [0, 0, 1, 1], "color": "red"}]}))
    reply = mcp_server.paint(str(proj.root), str(plan), str(tmp_path / "atlas.png"))
    assert "painted" in texts(reply) and len(pictures(reply)) == 1
    plan.write_text(json.dumps({"ops": [{"op": "blur"}]}))
    assert texts(mcp_server.paint(str(proj.root), str(plan), str(tmp_path / "b.png"))).startswith("PAINT_ERROR")


def test_a_bad_request_comes_back_as_a_message(proj, tmp_path):
    assert "mesh-jig init" in texts(mcp_server.guarded(mcp_server.brief)(str(tmp_path / "nowhere")))
    assert "needs an API key" in texts(mcp_server.guarded(mcp_server.judge)(str(proj.root), str(proj.root)))


def test_the_server_registers_every_tool_and_returns_images_as_content(proj):
    pytest.importorskip("mcp")
    server = mcp_server.serve()
    tools = asyncio.run(server.list_tools())
    listed = {t.name: t for t in getattr(tools, "tools", tools)}
    assert set(listed) == {"brief", "evaluate", "build", "preview", "paint", "judge"}
    schema = getattr(listed["evaluate"], "input_schema", None) or listed["evaluate"].inputSchema     # mcp 2.x / 1.x
    assert set(schema["required"]) == {"project", "script"}
    assert "Read a project before writing anything" in listed["brief"].description
    result = asyncio.run(server.call_tool("brief", {"project": str(proj.root)}))
    content = getattr(result, "content", None) or (result[0] if isinstance(result, tuple) else result)
    kinds = [c.type for c in content]
    assert kinds.count("image") == 4 and "text" in kinds
    missing = asyncio.run(server.call_tool("brief", {"project": str(proj.root / "nope")}))
    missing = getattr(missing, "content", None) or (missing[0] if isinstance(missing, tuple) else missing)
    assert "mesh-jig init" in missing[0].text
