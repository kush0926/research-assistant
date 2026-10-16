import pytest

from app.config import settings
from app.services import llm as llm_module


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        return None

    def json(self):
        return self.body


@pytest.mark.parametrize(
    ("provider", "key_name", "expected_model", "response", "expected_url"),
    [
        ("anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-5",
         {"content": [{"text": "answer"}]}, "https://api.anthropic.com/v1/messages"),
        ("openai", "OPENAI_API_KEY", "gpt-4o-mini",
         {"choices": [{"message": {"content": "answer"}}]}, "https://api.openai.com/v1/chat/completions"),
        ("groq", "GROQ_API_KEY", "openai/gpt-oss-20b",
         {"choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}]},
         "https://api.groq.com/openai/v1/chat/completions"),
        ("gemini", "GEMINI_API_KEY", "gemini-3.8-flash",
         {"candidates": [{"content": {"parts": [{"text": "answer"}]}}]},
         "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent"),
    ],
)
def test_provider_defaults_and_request_payloads(monkeypatch, provider, key_name, expected_model, response, expected_url):
    monkeypatch.setattr(settings, "LLM_PROVIDER", provider)
    monkeypatch.setattr(settings, "LLM_MODEL", "")
    monkeypatch.setattr(settings, key_name, "test-key")
    requests = []

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs == {"timeout": 90}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def post(self, url, **kwargs):
            requests.append((url, kwargs))
            return FakeResponse(response)

    monkeypatch.setattr(llm_module.httpx, "Client", FakeClient)
    client = llm_module.LLMClient()

    assert client.model == expected_model
    assert client.complete("system", "question") == "answer"
    assert requests[0][0] == expected_url
    body = requests[0][1]["json"]
    if provider == "gemini":
        assert body["generationConfig"]["maxOutputTokens"] == 1200
        assert "systemInstruction" in body
    else:
        assert body["model"] == expected_model
    if provider in ("anthropic", "gemini"):
        assert "temperature" not in body
    if provider in ("openai", "groq"):
        assert requests[0][1]["headers"]["Authorization"] == "Bearer test-key"
    if provider == "groq":
        assert body["max_completion_tokens"] == 2224
        assert "max_tokens" not in body
        assert body["reasoning_effort"] == "low"
        assert body["include_reasoning"] is False
        assert "reasoning_format" not in body


@pytest.mark.parametrize("provider,key_name", [("openai", "OPENAI_API_KEY"), ("groq", "GROQ_API_KEY")])
def test_unknown_or_missing_provider_key_stays_offline(monkeypatch, provider, key_name):
    monkeypatch.setattr(settings, "LLM_PROVIDER", provider)
    monkeypatch.setattr(settings, key_name, "")
    client = llm_module.LLMClient()

    assert not client.available
    with pytest.raises(RuntimeError, match="LLM not configured"):
        client.complete("system", "question")


@pytest.mark.parametrize("content,finish_reason", [(None, "stop"), ("", "stop"), ("partial", "length")])
def test_groq_rejects_empty_or_truncated_output(monkeypatch, content, finish_reason):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
    transport = llm_module.httpx.MockTransport(lambda request: llm_module.httpx.Response(
        200, json={"choices": [{"message": {"content": content}, "finish_reason": finish_reason}]}))
    real_client = llm_module.httpx.Client
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
    with pytest.raises(RuntimeError, match="empty or truncated"):
        llm_module.LLMClient().complete("system", "question")


def test_groq_rate_limit_is_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-key")
    transport = llm_module.httpx.MockTransport(lambda request: llm_module.httpx.Response(
        429, headers={"retry-after": "2"}, json={"error": {"message": "Rate limit reached"}}))
    real_client = llm_module.httpx.Client
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **kw: real_client(transport=transport, **kw))
    client = llm_module.LLMClient()
    with pytest.raises(llm_module.httpx.HTTPStatusError) as error:
        client.complete("system", "question")
    assert error.value.response.status_code == 429
    assert client.complete_json("system", "question") is None
