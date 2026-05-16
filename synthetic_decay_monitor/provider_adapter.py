"""Provider abstraction for LLM API backends.

Supports QuickRouter (OpenAI-compatible), OpenAI, DeepSeek, and local models.
"""
import json
import math
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import time
import certifi
import httpx


@dataclass
class ProviderConfig:
    """Configuration for an LLM API provider."""
    base_url: str
    api_key: str
    model: str
    max_tokens: int = 512
    temperature: float = 0.8
    top_p: float | None = None
    timeout_sec: int = 120
    extra_headers: dict = field(default_factory=dict)


class ProviderAdapter(ABC):
    """Abstract base for LLM API providers."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Send prompt to model, return response text."""
        ...

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def provider_name(self) -> str:
        return self.__class__.__name__


class OpenAICompatibleAdapter(ProviderAdapter):
    """Generic adapter for OpenAI-compatible chat completions API.

    Works with QuickRouter, OpenAI, DeepSeek, Together AI, Groq, etc.
    Uses certifi CA bundle for consistent SSL across platforms.
    """

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self._client = httpx.Client(
            verify=certifi.where(),
            timeout=httpx.Timeout(
                config.timeout_sec,
                connect=15.0,
                read=30.0,
                write=15.0,
                pool=10.0,
            ),
            trust_env=False,  # bypass system proxy (Clash etc.) for direct API calls
        )

    def generate(self, prompt: str, max_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }
        last_err = None
        for attempt in range(3):
            try:
                resp = self._client.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                # Fallback for reasoning models (DeepSeek-R1): use reasoning_content
                if not content:
                    content = (msg.get("reasoning_content") or "").strip()
                # Retry on empty content for reasoning models (Gemini 2.5 Pro etc.)
                if not content and attempt < 2:
                    reasoning_tokens = data.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                    if reasoning_tokens > 0:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                return content
            except (httpx.ConnectError, httpx.RemoteProtocolError,
                    httpx.ReadTimeout, httpx.WriteTimeout,
                    httpx.ConnectTimeout, httpx.PoolTimeout) as e:
                last_err = e
                if attempt < 2:
                    time.sleep(1.0 * (attempt + 1))
            except httpx.HTTPStatusError as e:
                last_err = e
                if attempt < 2 and e.response.status_code >= 500:
                    time.sleep(2.0 ** (attempt + 1))
                elif e.response.status_code == 429:
                    # Rate limit: don't retry internally, let outer loop handle with long delay
                    raise
        raise last_err

    def continuation_perplexity(self, text: str, n_tokens: int = 15) -> dict:
        """Compute perplexity of a model's continuation of the given text.

        Sends the text as a user message, asks the model to continue naturally,
        and collects logprobs on the continuation tokens.

        Returns dict with keys: perplexity, mean_logprob, logprobs_list, n_tokens.
        Returns None if the API does not support logprobs.
        """
        prompt = (
            "Continue the following text naturally and coherently. "
            "Do NOT add commentary, just continue the text as if you wrote it:\n\n"
            f"{text}\n\n"
            "Continuation:"
        )
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_tokens,
            "temperature": 0.0,  # deterministic for scoring
            "logprobs": True,
            "top_logprobs": 1,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }
        last_err = None
        for attempt in range(3):
            try:
                resp = self._client.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                logprobs_data = choice.get("logprobs")
                if logprobs_data is None or not logprobs_data.get("content"):
                    # API doesn't support logprobs — return None for fallback
                    return None
                token_logprobs = []
                for token_info in logprobs_data["content"]:
                    lp = token_info.get("logprob", 0.0)
                    token_logprobs.append(lp)
                if not token_logprobs:
                    return None
                mean_lp = sum(token_logprobs) / len(token_logprobs)
                perplexity = math.exp(-mean_lp)
                return {
                    "perplexity": round(perplexity, 4),
                    "mean_logprob": round(mean_lp, 6),
                    "logprobs_list": [round(lp, 6) for lp in token_logprobs],
                    "n_tokens": len(token_logprobs),
                }
            except (httpx.ConnectError, httpx.RemoteProtocolError,
                    httpx.ReadTimeout, httpx.WriteTimeout,
                    httpx.ConnectTimeout, httpx.PoolTimeout) as e:
                last_err = e
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
            except httpx.HTTPStatusError as e:
                last_err = e
                if e.response.status_code == 400:
                    # Likely logprobs not supported by this provider
                    return None
                if attempt < 2 and e.response.status_code >= 500:
                    time.sleep(2.0 ** (attempt + 1))
                elif e.response.status_code == 429:
                    raise
        # After all retries exhausted
        if last_err:
            print(f"    [WARN] continuation_perplexity failed: {last_err}")
        return None

    def naturalness_score(self, text: str) -> dict:
        """Rate text naturalness on 1-10 scale as a proxy for perplexity.

        Returns dict with keys: naturalness (1-10), perplexity_proxy.
        Higher naturalness = lower perplexity_proxy.
        """
        prompt = (
            "Rate the following AI-generated text on its NATURALNESS — how likely "
            "a well-trained language model would produce exactly this text. "
            "Consider: fluency, coherence, word choice typicality, lack of repetition, "
            "and overall human-likeness.\n\n"
            "Score 1-10:\n"
            "10 = indistinguishable from high-quality human writing\n"
            "7 = mostly natural with minor quirks\n"
            "5 = noticeably artificial or repetitive\n"
            "3 = severely degraded, fragmented\n"
            "1 = complete gibberish\n\n"
            "Reply with ONLY a JSON object:\n"
            '{"naturalness": <1-10>, "rationale": "<one short sentence>"}\n\n'
            f"Text to rate:\n---\n{text[:2500]}\n---\n\n"
            "JSON:"
        )
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 128,
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            **self.config.extra_headers,
        }
        for attempt in range(3):
            try:
                resp = self._client.post(
                    f"{self.config.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                content = (data["choices"][0]["message"].get("content") or "").strip()
                # Parse JSON from response
                import re as _re
                # Try direct parse
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    m = _re.search(r'\{[^{}]*"naturalness"[^{}]*\}', content)
                    if m:
                        try:
                            parsed = json.loads(m.group(0))
                        except json.JSONDecodeError:
                            parsed = {"naturalness": 5, "rationale": "parse failed"}
                    else:
                        parsed = {"naturalness": 5, "rationale": "parse failed"}
                nat = float(parsed.get("naturalness", 5))
                nat = max(1.0, min(10.0, nat))
                # Convert to "perplexity proxy": inverse relationship
                # perplexity ≈ 1000 / naturalness^2 (maps 10→10, 5→40, 1→1000)
                ppl_proxy = round(1000.0 / (nat ** 2), 2)
                return {
                    "naturalness": round(nat, 2),
                    "perplexity_proxy": ppl_proxy,
                    "rationale": parsed.get("rationale", ""),
                    "method": "naturalness_rating",
                }
            except (httpx.ConnectError, httpx.RemoteProtocolError,
                    httpx.ReadTimeout, httpx.WriteTimeout,
                    httpx.ConnectTimeout, httpx.PoolTimeout) as e:
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
            except httpx.HTTPStatusError as e:
                if attempt < 2 and e.response.status_code >= 500:
                    time.sleep(2.0 ** (attempt + 1))
                elif e.response.status_code == 429:
                    raise
        return {"naturalness": 5.0, "perplexity_proxy": 40.0,
                "rationale": "all attempts failed", "method": "naturalness_rating"}

    @property
    def provider_name(self) -> str:
        base = self.config.base_url.rstrip("/")
        if "quickrouter" in base:
            return "QuickRouter"
        if "openai" in base:
            return "OpenAI"
        if "together" in base:
            return "Together"
        if "deepseek" in base:
            return "DeepSeek"
        if "openrouter" in base:
            return "OpenRouter"
        return "OpenAI-compatible"


class LocalModelAdapter(ProviderAdapter):
    """Adapter for locally-hosted models (vLLM, llama.cpp, etc.).

    Expects an OpenAI-compatible local endpoint.
    """
    def generate(self, prompt: str, max_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        return OpenAICompatibleAdapter(self.config).generate(
            prompt, max_tokens=max_tokens, temperature=temperature
        )


def create_provider(provider: str = "quickrouter", model: str | None = None,
                    api_key: str | None = None, base_url: str | None = None,
                    **kwargs) -> ProviderAdapter:
    """Factory function for creating providers.

    Pre-configured providers:
    - quickrouter: QuickRouter API with Claude models
    - openai: OpenAI API
    - deepseek: DeepSeek API (OpenAI-compatible)
    - local: local model endpoint

    Examples:
        adapter = create_provider("quickrouter", model="claude-sonnet-4-6")
        adapter = create_provider("openai", model="gpt-4o", api_key="sk-...")
        adapter = create_provider("deepseek", model="deepseek-chat")
    """
    if provider == "quickrouter":
        config = ProviderConfig(
            base_url=base_url or "https://api.quickrouter.ai/v1",
            api_key=api_key or os.environ.get("QUICKROUTER_API_KEY", ""),
            model=model or "claude-sonnet-4-6",
            **kwargs,
        )
        return OpenAICompatibleAdapter(config)

    if provider == "openai":
        config = ProviderConfig(
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            model=model or "gpt-4o",
            **kwargs,
        )
        return OpenAICompatibleAdapter(config)

    if provider == "deepseek":
        config = ProviderConfig(
            base_url=base_url or "https://api.deepseek.com/v1",
            api_key=api_key or os.environ.get("DEEPSEEK_API_KEY", ""),
            model=model or "deepseek-chat",
            extra_headers=kwargs.pop("extra_headers", {}),
            **kwargs,
        )
        return OpenAICompatibleAdapter(config)

    if provider == "openrouter":
        config = ProviderConfig(
            base_url=base_url or "https://openrouter.ai/api/v1",
            api_key=api_key or os.environ.get("OPENROUTER_API_KEY", ""),
            model=model or "meta-llama/llama-3.1-8b-instruct",
            extra_headers=kwargs.pop("extra_headers", {}),
            **kwargs,
        )
        return OpenAICompatibleAdapter(config)

    if provider == "local":
        config = ProviderConfig(
            base_url=base_url or "http://localhost:8000/v1",
            api_key=api_key or "not-needed",
            model=model or "local-model",
            **kwargs,
        )
        return LocalModelAdapter(config)

    raise ValueError(f"Unknown provider: {provider}")
