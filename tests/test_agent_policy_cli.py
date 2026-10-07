"""CLI policy reaches both the request output cap and the tool loop without model calls."""
import pytest

from mesh_jig import agent, cli, llm
from test_mcp_server import proj  # noqa: F401


def test_context_reply_reserve_caps_output_and_passes_policy_to_loop(proj, monkeypatch):
    backend = llm.Backend("openai-compatible", "http://unused/v1", "fixture", "unused")
    monkeypatch.setattr(llm, "resolve_backend", lambda **kw: backend)
    # No endpoint metadata or inference: capture exactly what the CLI would configure.
    chat_args, run_args = {}, {}
    def chat(*args, **kwargs):
        chat_args.update(kwargs)
        return object()
    def run(*args, **kwargs):
        run_args.update(kwargs)
        return agent.Report(done=True)
    monkeypatch.setattr(agent, "chat_for", chat)
    monkeypatch.setattr(agent, "run", run)
    assert cli.main(["agent", str(proj.root), "--context-tokens", "65536", "--reply-reserve", "4096",
                     "--max-tokens", "20000", "--max-previews", "3", "--allow-pending-candidates"]) == 0
    assert chat_args["max_tokens"] == 4096
    assert run_args["context_tokens"] == 65536 and run_args["reply_reserve"] == 4096
    assert run_args["max_previews"] == 3 and run_args["require_evaluation"] is False
    prompt = agent.system_prompt(proj, 5, 40, False, max_previews=3, require_evaluation=False)
    assert "at most 3 successful previews" in prompt and "pending-candidate restriction is disabled" in prompt
    assert "Evaluate even a losing preview" not in prompt


@pytest.mark.parametrize("flags", [
    ["--max-previews", "-1"], ["--context-tokens", "-1"], ["--reply-reserve", "0"],
    ["--context-tokens", "8192", "--reply-reserve", "8192"],
])
def test_invalid_policy_stops_before_resolving_a_backend(proj, monkeypatch, flags):
    def unexpected(**kwargs):
        pytest.fail("invalid policy reached backend lookup")
    monkeypatch.setattr(llm, "resolve_backend", unexpected)
    assert cli.main(["agent", str(proj.root), *flags]) == 2
