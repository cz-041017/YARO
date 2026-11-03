"""从 LangChain 消息里提取 DeepSeek 用量（hit/miss/out 拆分）。

ChatOpenAI 基座把 usage 存进 ``response_metadata["token_usage"]``，但只有
``prompt_tokens``/``completion_tokens``，把 DeepSeek 的缓存命中/未命中拆分会丢掉。
:class:`~yaro.provider.deepseek.ChatDeepSeek` 在 ``_create_chat_result`` 里把 DeepSeek
原始字段透传进 ``response_metadata["deepseek_usage"]``，本模块据此还原成
:class:`~yaro.contracts.Usage`（hit/miss 价差 50×，计费必须分开）。
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.messages import AIMessage

from ..contracts import Usage


def usage_from_messages(messages: Sequence[Any]) -> Usage:
    """累加所有 AIMessage 的用量。

    - ``deepseek_usage`` 存在（ChatDeepSeek 真模型）→ 用 hit/miss/out 拆分；
    - 不存在（fake/测试模型）→ 回退 ``token_usage.prompt_tokens`` 全算 miss。
    """
    usage = Usage()
    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue
        du = (msg.response_metadata or {}).get("deepseek_usage") or {}
        if du:
            hit = int(du.get("prompt_cache_hit_tokens") or 0)
            miss = int(du.get("prompt_cache_miss_tokens") or 0)
            out = int(du.get("completion_tokens") or 0)
            if hit == 0 and miss == 0:  # 兜底：字段缺失时全算 miss（与旧 llm/deepseek.py 一致）
                miss = int(du.get("prompt_tokens") or 0)
        else:
            tu = (msg.response_metadata or {}).get("token_usage") or {}
            hit = 0
            miss = int(tu.get("prompt_tokens") or 0)
            out = int(tu.get("completion_tokens") or 0)
        usage.hit += hit
        usage.miss += miss
        usage.out += out
        usage.calls += 1
    return usage


__all__ = ["usage_from_messages"]
