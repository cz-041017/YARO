"""把 :class:`~yaro.config.Settings` 变成 LangChain 模型。

默认供应商是 DeepSeek（OpenAI 兼容）；换供应商 = 换 ``base_url``/``model``/``api_key``，
或以后在这里按 provider 名分派。litellm 等跨供应商抽象等真有用量再上（架构讨论结论）。
"""

from __future__ import annotations

from typing import Any

from .deepseek import ChatDeepSeek, deepseek_extra_body

DEFAULT_BASE_URL = "https://api.deepseek.com"


def build_model(
    settings: Any,
    *,
    model: str | None = None,
    thinking: bool = True,
    effort: str | None = "high",
    temperature: float = 0.0,
    timeout: float = 180.0,
    max_retries: int = 2,
) -> ChatDeepSeek:
    """按 Settings 构造一个 ``ChatDeepSeek``。

    ``model`` 不传时用 ``settings.model``（默认 ``deepseek-flash``）。
    规划/难归因等重活可 ``build_model(settings, model=settings.model_plan, effort="max")``。
    """
    model_name = model or getattr(settings, "model", "deepseek-flash")
    base_url = getattr(settings, "base_url", None) or DEFAULT_BASE_URL
    api_key = getattr(settings, "api_key", None) or "missing"

    return ChatDeepSeek(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
        extra_body=deepseek_extra_body(thinking=thinking, effort=effort),
    )


__all__ = ["build_model", "DEFAULT_BASE_URL"]
