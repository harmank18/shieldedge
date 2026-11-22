import platform

from shieldedge.llama_runner import llm_threads


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv("SHIELDEDGE_LLM_THREADS", "3")
    assert llm_threads() == "3"


def test_arm_defaults_to_two_threads(monkeypatch):
    monkeypatch.delenv("SHIELDEDGE_LLM_THREADS", raising=False)
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    assert llm_threads() == "2"


def test_x86_uses_runner_default(monkeypatch):
    monkeypatch.delenv("SHIELDEDGE_LLM_THREADS", raising=False)
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    assert llm_threads() is None
