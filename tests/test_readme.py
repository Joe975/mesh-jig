"""README.md and the page it sends a person on to (docs/GUIDE.md) are instructions: the commands they hand over
must parse and their links must resolve (LESSONS.md section 7: the instructions are a program)."""
import re
import shlex
from pathlib import Path

import pytest

from mesh_jig import cli

README = Path(__file__).resolve().parents[1] / "README.md"
GUIDE = README.parent / "docs" / "GUIDE.md"


def code_lines(text: str) -> list[str]:
    """Every line inside a fenced block, stripped (a block may be indented under a list item)."""
    out, inside = [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            inside = not inside
        elif inside:
            out.append(line.strip())
    return out


def commands(text: str) -> list[str]:
    return [c for c in code_lines(text) if c.startswith("mesh-jig ")]


@pytest.mark.parametrize("page, fewest", [(README, 8), (GUIDE, 15)], ids=["README.md", "docs/GUIDE.md"])
def test_every_mesh_jig_command_a_page_hands_over_parses(page, fewest):
    found = commands(page.read_text(encoding="utf-8"))
    assert len(found) >= fewest                 # the walk through is there, not an empty file
    for command in found:
        try:
            cli.parser().parse_args(shlex.split(command, comments=True)[1:])
        except SystemExit:
            pytest.fail(f"{page.name} hands over a command the CLI refuses: {command}")


def test_a_command_the_cli_does_not_have_is_caught():
    assert commands("```\nmesh-jig evaluate x\n```") == ["mesh-jig evaluate x"]
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["evaluate", "x"])


def test_nothing_tells_anyone_to_install_from_pypi():
    """mesh-jig is not on PyPI: the install is `pip install -e .` in a checkout."""
    root = README.parent
    files = [README, root / "AGENTS.md", *root.glob("docs/*.md"), *root.glob("skill/*/SKILL.md"),
             *root.glob("src/mesh_jig/*.py")]
    wrong = [f.relative_to(root).as_posix() for f in files
             if re.search(r"pip install [\"']?mesh-jig", f.read_text(encoding="utf-8"))]
    assert not wrong, f"these name a PyPI install that does not exist: {wrong}"


@pytest.mark.parametrize("page", [README, GUIDE], ids=["README.md", "docs/GUIDE.md"])
def test_every_relative_link_resolves(page):
    text = page.read_text(encoding="utf-8")
    targets = [t for t in re.findall(r"\]\(([^)]+)\)", text) if not t.startswith(("http", "#"))]
    assert targets
    missing = [t for t in targets if not (page.parent / t.split("#")[0]).exists()]
    assert not missing, f"{page.name} links to nothing: {missing}"


def test_the_readme_stays_short_and_links_to_the_guide():
    text = README.read_text(encoding="utf-8")
    assert len(text.split()) <= 1200
    assert "](docs/GUIDE.md)" in text
    assert "docs/img/" not in text
