from factory.config import Settings


def test_settings_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    monkeypatch.setenv("SDF_MODEL", "gpt-4o")
    s = Settings()
    assert s.openai_api_key == "sk-x"
    assert s.model == "gpt-4o"
    assert s.has_key


def test_settings_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert Settings().has_key is False
