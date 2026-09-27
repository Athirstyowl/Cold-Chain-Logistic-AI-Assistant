import os
from dataclasses import dataclass, field
from typing import Mapping, Optional

# ==========================================
# 1. PROVIDER REGISTRY
# ==========================================
# API keys are read from the server .env only. A provider is offered in the UI
# only when its key is present. Default models can be overridden per provider
# with <PROVIDER>_MODEL (e.g. GEMINI_MODEL, OPENROUTER_MODEL).


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    label: str
    kind: str                      # "openai_compatible" | "gemini" | "ollama"
    key_env: Optional[str]         # None = no API key needed
    default_model: str
    suggested_models: tuple = field(default_factory=tuple)
    base_url: Optional[str] = None

    @property
    def model_env(self) -> str:
        return f"{self.id.upper()}_MODEL"


PROVIDERS: dict[str, ProviderSpec] = {
    "gemini": ProviderSpec(
        id="gemini",
        label="Google Gemini",
        kind="gemini",
        key_env="GEMINI_API_KEY",
        default_model="gemini-3.8-flash",
        suggested_models=("gemini-3.8-flash", "gemini-3.7-flash"),
    ),
    "openai": ProviderSpec(
        id="openai",
        label="OpenAI (ChatGPT)",
        kind="openai_compatible",
        key_env="OPENAI_API_KEY",
        default_model="gpt-6-luna",
        suggested_models=("gpt-6-luna", "gpt-6-luna-pro", "gpt-6-sol-pro"),
    ),
    "deepseek": ProviderSpec(
        id="deepseek",
        label="DeepSeek",
        kind="openai_compatible",
        key_env="DEEPSEEK_API_KEY",
        default_model="deepseek-v4-flash",
        suggested_models=("deepseek-v4-flash", "deepseek-v4-pro"),
        base_url="https://api.deepseek.com",
    ),
    "openrouter": ProviderSpec(
        id="openrouter",
        label="OpenRouter",
        kind="openai_compatible",
        key_env="OPENROUTER_API_KEY",
        default_model="google/gemini-3.8-flash",
        suggested_models=(
            "google/gemini-3.8-flash",
            "openai/gpt-6-luna",
            "anthropic/claude-opus-5.5",
            "deepseek/deepseek-v4.1-flash",
            "qwen/qwen3.8-flash",
            "meta-llama/llama-4-maverick",
        ),
        base_url="https://openrouter.ai/api/v1",
    ),
    "groq": ProviderSpec(
        id="groq",
        label="Groq",
        kind="openai_compatible",
        key_env="GROQ_API_KEY",
        default_model="llama-3.3-70b-versatile",
        suggested_models=("llama-3.3-70b-versatile", "openai/gpt-oss-120b"),
        base_url="https://api.groq.com/openai/v1",
    ),
    "ollama": ProviderSpec(
        id="ollama",
        label="Ollama (local)",
        kind="ollama",
        key_env=None,
        default_model="qwen2.5:7b",
        suggested_models=("qwen2.5:7b", "llama3.1:8b"),
    ),
}


# Room for hidden reasoning plus the report (DeepSeek v4.1 used ~1.4k reasoning tokens and truncated at 2048).
# An explicit cap also keeps OpenRouter's per-request credit reservation below the model maximum.
MAX_OUTPUT_TOKENS = 6144


def _env(env: Optional[Mapping[str, str]]) -> Mapping[str, str]:
    return os.environ if env is None else env


def _get_spec(provider_id: str) -> ProviderSpec:
    spec = PROVIDERS.get((provider_id or "").strip().lower())
    if spec is None:
        raise ValueError(f"Unknown LLM provider '{provider_id}'. Choose one of: {', '.join(PROVIDERS)}")
    return spec


def _api_key(spec: ProviderSpec, env: Mapping[str, str]) -> str:
    return (env.get(spec.key_env) or "").strip() if spec.key_env else ""


def available_providers(env: Optional[Mapping[str, str]] = None) -> list[ProviderSpec]:
    """Providers usable right now: keyless ones, plus those whose key is set in .env."""
    env = _env(env)
    return [spec for spec in PROVIDERS.values() if spec.key_env is None or _api_key(spec, env)]


def resolve_model(provider_id: str, model: Optional[str] = None, env: Optional[Mapping[str, str]] = None) -> str:
    """Explicit choice > <PROVIDER>_MODEL env override > registry default."""
    spec = _get_spec(provider_id)
    env = _env(env)
    return (model or "").strip() or (env.get(spec.model_env) or "").strip() or spec.default_model


# ==========================================
# 2. LLM FACTORY
# ==========================================
def build_llm(provider_id: str, model: Optional[str] = None, env: Optional[Mapping[str, str]] = None):
    spec = _get_spec(provider_id)
    env = _env(env)
    model_name = resolve_model(spec.id, model, env)
    api_key = _api_key(spec, env)

    if spec.key_env and not api_key:
        raise ValueError(f"{spec.key_env} must be set in .env to use {spec.label}.")

    if spec.kind == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=api_key)

    if spec.kind == "ollama":
        from langchain_ollama import ChatOllama
        base_url = (env.get("OLLAMA_BASE_URL") or "").strip() or "http://localhost:11434"
        return ChatOllama(model=model_name, temperature=0, num_predict=1024, base_url=base_url)

    from langchain_openai import ChatOpenAI
    extra = {}
    if spec.id == "openrouter":
        extra["default_headers"] = {"X-Title": "Cold-Chain Dispatch Console"}
    return ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
        base_url=spec.base_url,
        max_tokens=MAX_OUTPUT_TOKENS,
        max_retries=1,
        **extra,
    )


# ==========================================
# 3. USER-FACING HELPERS
# ==========================================
def _status_code(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def friendly_error(exc: Exception, provider_label: str) -> str:
    """Turn a provider exception into a message a dispatcher can act on."""
    status = _status_code(exc)
    detail = str(exc).lower()

    if status in (401, 403) or "api key" in detail or "unauthorized" in detail:
        return f"🔑 {provider_label} rejected the API key. Check its key in the server .env."
    if status == 402 or "insufficient credit" in detail:
        return f"💳 {provider_label} reports insufficient credits on this account."
    if status == 429 or "rate limit" in detail or "quota" in detail or "resource_exhausted" in detail:
        return f"⏳ {provider_label} rate limit or quota reached. Wait a moment and try again."
    if status == 404 or "model not found" in detail or "not a valid model" in detail:
        return f"❓ {provider_label} does not recognise the selected model. Pick another model ID."
    if isinstance(exc, (ConnectionError, TimeoutError)) or "connect" in type(exc).__name__.lower() or "connection" in detail:
        return f"🔌 Could not reach {provider_label}. Check the network (or that the local server is running)."
    return f"⚠️ {provider_label} error: {exc}"


def message_text(message) -> str:
    """Plain text of an AI message, whether content is a string or a list of blocks."""
    return message.text or ""


_TRUNCATION_REASONS = {"length", "max_tokens"}


def final_answer(message) -> Optional[tuple[str, bool]]:
    """(text, truncated) for the answer step; None while the model is still calling tools."""
    if getattr(message, "tool_calls", None):
        return None
    metadata = getattr(message, "response_metadata", None) or {}
    reason = str(metadata.get("finish_reason") or metadata.get("done_reason") or "").lower()
    return message_text(message), reason in _TRUNCATION_REASONS
