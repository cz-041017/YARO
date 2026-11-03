"""DeepSeek 的 LangChain 模型适配器。

为什么不用 ``langchain-deepseek`` 的 ``ChatDeepSeek`` 直接上：它把思考模式的
``reasoning_content`` 存进 ``AIMessage.additional_kwargs``，但序列化回请求时**不把
它带出去**（langchain#35093，closed 未 merge）。后果：一旦在 thinking 模式下带
``tools`` 发起下一次调用，DeepSeek 侧就报 400 ``Missing reasoning_content field in
the assistant message``。

本模块用 ``langchain_openai.ChatOpenAI`` 作基座（DeepSeek 是 OpenAI 兼容协议），补
两半缺的链路：

1. **响应解析层**：``ChatOpenAI`` 把 DeepSeek 返回的 ``reasoning_content`` 直接丢掉
   （``_convert_dict_to_message`` / ``_convert_delta_to_message_chunk`` 都不提取它），
   所以先在这两处把 ``reasoning_content`` 抄进 ``additional_kwargs``。
2. **请求序列化层**：``_get_request_payload`` 里把 ``additional_kwargs["reasoning_content"]``
   回填到**每一条** assistant 消息上（thinking 模式下，含无 tool_calls 的轮次，空串也回填；
   参照 PR #35093 diff）。

另修 DeepSeek 特有 wire 约束：content 列表转字符串、图只能放 user 消息。

出网脱敏（egress）不在这里做，见 :mod:`yaro.provider` 包 docstring。
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_openai import ChatOpenAI

# 旧模型名（thinking 模式专属，已退役但仍有人用），等价于显式开 thinking。
_REASONING_MODEL_NAMES = frozenset({"deepseek-reasoner"})


def deepseek_extra_body(*, thinking: bool = True, effort: str | None = "high") -> dict[str, Any]:
    """构造 DeepSeek 请求的 ``extra_body``（OpenAI 格式的 thinking + effort）。

    - ``thinking=True`` → ``{"thinking": {"type": "enabled"}, "reasoning_effort": <effort>}``
    - ``thinking=False`` → ``{"thinking": {"type": "disabled"}}``（reasoning_effort 在非
      thinking 模式无效，故不传）

    ``effort`` 只接受 DeepSeek 认可的档位（low/high/max）；传 None 表示不指定。
    """
    body: dict[str, Any] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
    if thinking and effort is not None:
        body["reasoning_effort"] = effort
    return body


class ChatDeepSeek(ChatOpenAI):
    """``ChatOpenAI`` 的 DeepSeek 专用子类。

    只做 DeepSeek wire 层正确性；不含出网脱敏、不含缓存前缀布局（那些是图节点的职责）。
    """

    # ------------------------------------------------------------------ #
    # thinking 判定
    # ------------------------------------------------------------------ #

    def _thinking_enabled(self) -> bool:
        if self.model_name in _REASONING_MODEL_NAMES:
            return True
        # 两条合法来源：extra_body（build_model 走这里）或 model_kwargs（直接构造时）。
        for source in (self.extra_body or {}, getattr(self, "model_kwargs", {}) or {}):
            thinking = source.get("thinking")
            if isinstance(thinking, dict) and thinking.get("type") == "enabled":
                return True
        return False

    # ------------------------------------------------------------------ #
    # 响应解析层：把 reasoning_content 抄进 additional_kwargs
    # ------------------------------------------------------------------ #

    @staticmethod
    def _response_as_dict(response: Any) -> dict[str, Any]:
        if isinstance(response, dict):
            return response
        if hasattr(response, "model_dump"):
            return response.model_dump(
                exclude={"choices": {"__all__": {"message": {"parsed"}}}},
                warnings=False,
            )
        return {}

    def _create_chat_result(self, response: Any, generation_info: dict | None = None):
        result = super()._create_chat_result(response, generation_info=generation_info)
        response_dict = self._response_as_dict(response)
        choices = response_dict.get("choices") or []
        usage = response_dict.get("usage") or {}
        for i, gen in enumerate(result.generations):
            if i >= len(choices) or not isinstance(gen.message, AIMessage):
                continue
            rc = (choices[i].get("message") or {}).get("reasoning_content")
            if rc is not None:
                gen.message.additional_kwargs["reasoning_content"] = rc
            # DeepSeek 缓存命中/未命中 token 拆分（hit/miss 价差 50×，计费必须分开）。
            # ChatOpenAI 基座只存 prompt_tokens/completion_tokens，会把 hit/miss 合进
            # prompt_tokens 丢掉缓存命中信息。这里把 DeepSeek 原始字段透传进
            # response_metadata，供 yaro.provider.usage.usage_from_messages 记账。
            gen.message.response_metadata["deepseek_usage"] = {
                "prompt_cache_hit_tokens": int(usage.get("prompt_cache_hit_tokens") or 0),
                "prompt_cache_miss_tokens": int(usage.get("prompt_cache_miss_tokens") or 0),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
            }
        return result

    def _convert_chunk_to_generation_chunk(
        self, chunk: dict, default_chunk_class: type, base_generation_info: dict | None
    ):
        gen_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if gen_chunk is None or not isinstance(gen_chunk.message, AIMessageChunk):
            return gen_chunk
        choices = chunk.get("choices") or chunk.get("chunk", {}).get("choices") or []
        if choices:
            rc = (choices[0].get("delta") or {}).get("reasoning_content")
            if rc is not None:
                gen_chunk.message.additional_kwargs["reasoning_content"] = rc
        return gen_chunk

    # ------------------------------------------------------------------ #
    # 请求序列化层：reasoning_content 回传 + content 规范化 + 视觉校验
    # ------------------------------------------------------------------ #

    def _original_messages(self, input_: Any) -> list[BaseMessage] | None:
        """把输入还原成 BaseMessage 列表，供 reasoning_content 取值。"""
        try:
            prompt_value = self._convert_input(input_)
            return list(prompt_value.to_messages())
        except Exception:
            return list(input_) if isinstance(input_, list) else None

    def _validate_vision_placement(self, messages: Sequence[Any]) -> None:
        """图只能放 user 消息；其它角色带图 DeepSeek 一律 400。"""
        for msg in messages:
            if not hasattr(msg, "type"):
                continue  # 防御：异常输入由 _original_messages 兜底，这里不二次崩溃
            if _is_user_message(msg):
                continue
            if _content_has_image(getattr(msg, "content", None)):
                raise ValueError(
                    f"DeepSeek 只允许 user 消息携带图片，但 {msg.type} 消息含 image 内容。"
                    "请把图像证据放进 HumanMessage（并注意 egress 策略）。"
                )

    def _get_request_payload(
        self, input_: Any, *, stop: list[str] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        original = self._original_messages(input_)
        if original is not None:
            self._validate_vision_placement(original)
        thinking = self._thinking_enabled()

        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")

            # DeepSeek 不接受 list 形式的 content。
            if role == "tool" and isinstance(content, list):
                msg["content"] = json.dumps(content)
            elif role == "assistant" and isinstance(content, list):
                msg["content"] = "".join(
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )

            # 关键修复：thinking 模式下**每一条** assistant 消息都要回传 reasoning_content
            # （连没有 tool_calls 的轮次也要），否则续跑/多轮时 DeepSeek 报 400
            # ``Missing reasoning_content field in the assistant message``。
            if (
                thinking
                and role == "assistant"
                and "reasoning_content" not in msg
            ):
                reasoning = ""
                if original is not None and i < len(original):
                    ak = getattr(original[i], "additional_kwargs", None)
                    if isinstance(ak, dict):
                        reasoning = ak.get("reasoning_content") or ""
                msg["reasoning_content"] = reasoning

        return payload


def _is_user_message(msg: Any) -> bool:
    if msg.type == "human":
        return True
    return msg.type == "chat" and getattr(msg, "role", None) == "user"


def _content_has_image(content: Any) -> bool:
    """判断一条消息的 content 是否含图（覆盖 OpenAI image_url 与 Anthropic image 块）。"""
    if isinstance(content, list):
        return any(_content_has_image(part) for part in content)
    if isinstance(content, dict):
        if content.get("type") == "image":
            return True
        return bool(content.get("image_url")) or bool(content.get("image"))
    return False


__all__ = ["ChatDeepSeek", "deepseek_extra_body"]
