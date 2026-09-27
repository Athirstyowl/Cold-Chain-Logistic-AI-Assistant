import httpx
import openai
import pytest
from langchain_core.messages import AIMessage

from src.llm_providers import (
    PROVIDERS,
    available_providers,
    build_llm,
    final_answer,
    friendly_error,
    message_text,
    resolve_model,
)


# ==========================================
# Provider availability (keys come from .env only)
# ==========================================
def test_provider_is_offered_only_when_its_api_key_is_set():
    env = {"GEMINI_API_KEY": "g-key", "OPENROUTER_API_KEY": "or-key"}
    ids = [p.id for p in available_providers(env)]
    assert "gemini" in ids
    assert "openrouter" in ids
    assert "openai" not in ids
    assert "deepseek" not in ids
    assert "groq" not in ids


def test_blank_or_whitespace_api_key_counts_as_missing():
    env = {"GEMINI_API_KEY": "", "GROQ_API_KEY": "   "}
    ids = [p.id for p in available_providers(env)]
    assert "gemini" not in ids
    assert "groq" not in ids


def test_ollama_is_offered_without_any_api_key():
    ids = [p.id for p in available_providers({})]
    assert ids == ["ollama"]


def test_all_requested_providers_are_registered():
    assert set(PROVIDERS) == {"gemini", "openai", "deepseek", "openrouter", "groq", "ollama"}


# ==========================================
# Model resolution
# ==========================================
def test_explicit_model_choice_wins_over_env_override():
    env = {"OPENROUTER_MODEL": "qwen/qwen3.8-flash"}
    assert resolve_model("openrouter", "openai/gpt-6-luna", env) == "openai/gpt-6-luna"


def test_env_model_override_replaces_the_default():
    env = {"GEMINI_MODEL": "gemini-3.7-flash"}
    assert resolve_model("gemini", None, env) == "gemini-3.7-flash"


def test_default_model_used_when_nothing_is_configured():
    assert resolve_model("deepseek", None, {}) == PROVIDERS["deepseek"].default_model


def test_blank_model_choice_falls_back_to_default():
    assert resolve_model("groq", "  ", {}) == PROVIDERS["groq"].default_model


def test_unknown_provider_raises_clear_error():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        resolve_model("mystery", None, {})


# ==========================================
# LLM construction (no network calls)
# ==========================================
@pytest.mark.parametrize(
    "provider_id, key_env, expected_base_url",
    [
        ("openai", "OPENAI_API_KEY", None),
        ("deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com"),
        ("openrouter", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
        ("groq", "GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    ],
)
def test_openai_compatible_providers_use_their_own_key_and_endpoint(provider_id, key_env, expected_base_url):
    llm = build_llm(provider_id, "some-model", {key_env: "secret-123"})
    assert llm.model_name == "some-model"
    assert llm.openai_api_key.get_secret_value() == "secret-123"
    assert llm.openai_api_base == expected_base_url
    assert llm.temperature == 0


@pytest.mark.parametrize("provider_id", ["openai", "deepseek", "openrouter", "groq"])
def test_output_tokens_are_capped_so_credit_reservation_stays_small(provider_id):
    # OpenRouter reserves credits for max_tokens; uncapped it asks for the model maximum (402)
    llm = build_llm(provider_id, "m", {PROVIDERS[provider_id].key_env: "k"})
    # Reasoning models spend part of the budget on hidden reasoning; 2048 truncated DeepSeek v4.1 reports
    assert llm.max_tokens == 6144


def test_gemini_is_built_with_the_gemini_key_and_model():
    llm = build_llm("gemini", "gemini-3.8-flash", {"GEMINI_API_KEY": "g-secret"})
    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert llm.model.endswith("gemini-3.8-flash")
    assert llm.google_api_key.get_secret_value() == "g-secret"


def test_ollama_uses_configured_base_url():
    llm = build_llm("ollama", "qwen2.5:7b", {"OLLAMA_BASE_URL": "http://gpu-box:11434"})
    assert type(llm).__name__ == "ChatOllama"
    assert llm.model == "qwen2.5:7b"
    assert llm.base_url == "http://gpu-box:11434"


def test_building_a_provider_without_its_key_is_refused():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_llm("openai", None, {})


# ==========================================
# Friendly error mapping
# ==========================================
def _openai_status_error(cls, status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.test"))
    return cls("raw provider message", response=response, body=None)


def test_auth_failure_becomes_invalid_key_message():
    msg = friendly_error(_openai_status_error(openai.AuthenticationError, 401), "OpenRouter")
    assert "OpenRouter" in msg
    assert "API key" in msg


def test_rate_limit_becomes_retry_message():
    msg = friendly_error(_openai_status_error(openai.RateLimitError, 429), "Groq")
    assert "rate" in msg.lower() or "quota" in msg.lower()


def test_insufficient_credits_is_reported():
    msg = friendly_error(_openai_status_error(openai.APIStatusError, 402), "OpenRouter")
    assert "credit" in msg.lower()


def test_unknown_model_is_reported():
    msg = friendly_error(_openai_status_error(openai.NotFoundError, 404), "OpenAI")
    assert "model" in msg.lower()


def test_invalid_model_id_rejected_with_400_is_reported_as_model_problem():
    response = httpx.Response(400, request=httpx.Request("POST", "https://example.test"))
    exc = openai.BadRequestError("Error code: 400 - nonexistent/model-xyz is not a valid model ID", response=response, body=None)
    msg = friendly_error(exc, "OpenRouter")
    assert "model" in msg.lower()
    assert "not recognise" in msg


def test_connection_failure_is_reported():
    msg = friendly_error(ConnectionError("Connection refused"), "Ollama (local)")
    assert "reach" in msg.lower()


def test_unrecognised_error_keeps_the_original_detail():
    msg = friendly_error(RuntimeError("something odd"), "Gemini")
    assert "something odd" in msg


# ==========================================
# Message text normalisation
# ==========================================
def test_plain_string_content_is_returned_as_is():
    assert message_text(AIMessage(content="hello")) == "hello"


def test_block_list_content_is_flattened_to_text():
    msg = AIMessage(content=[{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}])
    assert message_text(msg) == "Part one. Part two."


def test_empty_content_gives_empty_string():
    assert message_text(AIMessage(content=[])) == ""


# ==========================================
# Final answer extraction
# ==========================================
def test_text_alongside_tool_calls_is_not_the_final_answer():
    msg = AIMessage(
        content="Now let me check the weather.",
        tool_calls=[{"name": "fetch_corridor_conditions", "args": {}, "id": "call_1"}],
    )
    assert final_answer(msg) is None


def test_text_without_tool_calls_is_the_final_answer():
    msg = AIMessage(content="### 1. Executive Summary", response_metadata={"finish_reason": "stop"})
    assert final_answer(msg) == ("### 1. Executive Summary", False)


@pytest.mark.parametrize("reason_key, reason", [
    ("finish_reason", "length"),      # OpenAI-compatible (OpenAI, DeepSeek, OpenRouter, Groq)
    ("finish_reason", "MAX_TOKENS"),  # Gemini
    ("done_reason", "length"),        # Ollama
])
def test_answer_cut_off_by_the_output_limit_is_flagged(reason_key, reason):
    msg = AIMessage(content="### 1. Executive Summary ...", response_metadata={reason_key: reason})
    assert final_answer(msg) == ("### 1. Executive Summary ...", True)


def test_empty_final_message_still_counts_as_the_answer_step():
    msg = AIMessage(content="", response_metadata={"finish_reason": "length"})
    assert final_answer(msg) == ("", True)
