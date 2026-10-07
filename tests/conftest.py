"""Keep every test offline and off the machine it runs on: no personal secrets file, no judge or agent configured, and no
Blender unless a test asks for the real one."""
import pytest

from mesh_jig import build, llm

JUDGE_VARS = ("MESH_JIG_JUDGE_URL", "MESH_JIG_JUDGE_MODEL", "MESH_JIG_JUDGE_KEY", "MESH_JIG_ENV_FILE",
              "MESH_JIG_AGENT_URL", "MESH_JIG_AGENT_MODEL", "MESH_JIG_AGENT_KEY",
              "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "DEFAULT_ENV_FILE", str(tmp_path / "no-such-llm.env"))
    for name in JUDGE_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def blender():
    """The real Blender, or skip."""
    exe = build.find_blender()
    if not exe:
        pytest.skip("Blender is not installed")
    return exe
