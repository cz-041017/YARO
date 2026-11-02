"""DeepSeek 传输层。

两个职责，都容易被做错：

1. **reasoning_content 回传**：请求带 ``tools`` 时，历史每一条 assistant 消息
   都必须带上它当时的 ``reasoning_content``（原样，连没有工具调用的轮次也一样），
   否则接口返回 400。所以这里存的消息字典**永远**包含该键，并在编码时断言。
2. **前缀缓存布局**：缓存命中要求完整匹配一个缓存前缀单元，价差 50×。
   所以本层绝不改写历史消息、不重排工具、不做"聪明"的规范化——只做搬运。

模型选择（2026-11-02 官方文档）：
- ``deepseek-flash``：唯一支持 Vision，1M 上下文，thinking 默认开启
- ``deepseek-v4-pro``：不支持 Vision，用于规划/难归因/review
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from ..contracts import ContractError, DegradedBackend, Usage

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查，避免运行时循环导入
    from ..egress import EgressPolicy

# 峰时（UTC 01:00-04:00、06:00-10:00，周一至周五，不含中国法定假日）价格翻倍
_PEAK_HOURS_UTC = ((1, 4), (6, 10))

_RETRYABLE = {
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "APIStatusError",
}


def is_peak(now: datetime | None = None) -> bool:
    """是否处于峰时定价。节假日例外无法在本地判断，调用方按 false 保守处理即可。"""
    now = now or datetime.now(timezone.utc)
    if now.weekday() >= 5:  # 周六周日全谷时
        return False
    return any(start <= now.hour < end for start, end in _PEAK_HOURS_UTC)


class AssistantTurn:
    """一次模型回复。``reasoning_content`` 不是可选字段，是协议要求。"""

    __slots__ = ("content", "reasoning_content", "tool_calls", "usage", "model", "finish_reason")

    def __init__(
        self,
        *,
        content: str = "",
        reasoning_content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        usage: Usage | None = None,
        model: str = "",
        finish_reason: str = "",
    ) -> None:
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls or []
        self.usage = usage or Usage()
        self.model = model
        self.finish_reason = finish_reason

    def to_message(self) -> dict[str, Any]:
        """转成可写入会话、也可回传给 API 的消息字典。键必须齐全。"""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
            "reasoning_content": self.reasoning_content,
        }
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        return message

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"AssistantTurn(model={self.model!r}, len(content)={len(self.content)}, "
            f"tool_calls={[c['function']['name'] for c in self.tool_calls]}, "
            f"usage={self.usage.to_dict()})"
        )


def to_api_messages(messages: Sequence[dict[str, Any]], *, has_tools: bool) -> list[dict[str, Any]]:
    """把会话消息编码成请求体。

    ``has_tools=True`` 时断言 assistant 消息都带 ``reasoning_content`` 键——
    漏传会在服务端报 400，且错误信息不会告诉你是漏了哪一轮。
    """
    encoded: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "assistant":
            if has_tools and "reasoning_content" not in message:
                raise ContractError(
                    f"messages[{index}] 是 assistant 但缺少 reasoning_content；"
                    "带 tools 的请求必须把历史每轮 reasoning_content 原样回传，否则接口返回 400"
                )
            item: dict[str, Any] = {"role": "assistant", "content": message.get("content") or ""}
            if has_tools or message.get("reasoning_content"):
                # 带 tools 时**每一轮**都要有这个键（空串也要带），否则 400
                item["reasoning_content"] = message.get("reasoning_content") or ""
            if message.get("tool_calls"):
                item["tool_calls"] = message["tool_calls"]
            encoded.append(item)
        elif role == "tool":
            if not message.get("tool_call_id"):
                raise ContractError(f"messages[{index}] 是 tool 但缺少 tool_call_id")
            encoded.append(
                {
                    "role": "tool",
                    "tool_call_id": message["tool_call_id"],
                    "content": message.get("content") or "",
                }
            )
        elif role in ("user", "system"):
            encoded.append({"role": role, "content": message.get("content")})
        else:
            raise ContractError(f"messages[{index}] 的 role 非法：{role!r}")
    return encoded


def assert_vision_placement(messages: Iterable[dict[str, Any]]) -> None:
    """图片只能出现在 user 消息里；system/assistant 带图会被 400 拒绝。

    在发送前检查，比等 400 便宜得多。
    """
    for index, message in enumerate(messages):
        content = message.get("content")
        if isinstance(content, list) and message.get("role") != "user":
            has_image = any(isinstance(part, dict) and part.get("type") in ("image_url", "input_image") for part in content)
            if has_image:
                raise ContractError(f"messages[{index}] role={message.get('role')} 携带图片；图片只允许放在 user 消息")


class DeepSeek:
    """DeepSeek 客户端。

    P0 只实现 ``mode='chat'``（Chat Completions）。``mode='responses'`` 是保留位：
    要走 Responses API 才能让工具返回图片、或往历史里插入 tool call 消息，P1 补。
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-flash",
        mode: str = "chat",
        max_retries: int = 4,
        timeout_s: float = 180.0,
        egress: "EgressPolicy | None" = None,
    ) -> None:
        if not api_key:
            raise DegradedBackend("DEEPSEEK_API_KEY 为空：请在 .env 或环境变量中配置")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.mode = mode
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        # 出网策略：None 表示未受约束（仅测试）；生产必须传，否则原图可能出网。
        self.egress = egress
        self._client_obj: Any = None

    # ------------------------------------------------------------------ #
    # 客户端
    # ------------------------------------------------------------------ #

    def _client(self) -> Any:
        if self._client_obj is not None:
            return self._client_obj
        try:
            from openai import AsyncOpenAI  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - 取决于安装的 extras
            raise DegradedBackend(
                "缺少 openai 依赖：pip install 'yaro[api]'（core 刻意不依赖任何 SDK）"
            ) from exc
        self._client_obj = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout_s,
            max_retries=0,  # 重试由本层控制，便于记账与退避
        )
        return self._client_obj

    # ------------------------------------------------------------------ #
    # 主调用
    # ------------------------------------------------------------------ #

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        effort: str | None = None,
        thinking: bool = True,
        max_tokens: int | None = None,
    ) -> AssistantTurn:
        """一次补全。失败重试仅覆盖 429/5xx/超时/连接错误。"""
        model = model or self.model
        tools = list(tools or [])
        if self.egress is not None:
            # 发出请求前的最后一道闸：off 模式 / 未授权公网端点在这里被拦下。
            self.egress.check_endpoint(self.base_url)
        assert_vision_placement(messages)
        encoded = to_api_messages(messages, has_tools=bool(tools))

        body: dict[str, Any] = {
            "model": model,
            "messages": encoded,
        }
        extra: dict[str, Any] = {"thinking": {"type": "enabled" if thinking else "disabled"}}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if effort and thinking:
            extra["reasoning_effort"] = effort
        if max_tokens:
            body["max_tokens"] = max_tokens

        response = await self._create(body, extra)
        return self._parse(response, model=model)

    async def _create(self, body: dict[str, Any], extra: dict[str, Any]) -> Any:
        client = self._client()
        delay = 1.0
        last_error: BaseException | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # thinking / reasoning_effort 走 extra_body，不能进 body：
                # 它们是 DeepSeek 扩展字段，SDK 的 create() 没有这些形参。
                return await client.chat.completions.create(**body, extra_body=extra)
            except BaseException as exc:  # noqa: BLE001 - 需要按类名判定可重试性
                last_error = exc
                retryable = type(exc).__name__ in _RETRYABLE and attempt < self.max_retries
                if not retryable:
                    raise
                await asyncio.sleep(delay + random.random() * 0.3)
                delay = min(delay * 2, 20.0)
        raise last_error  # pragma: no cover - 循环内必然 return 或 raise

    # ------------------------------------------------------------------ #
    # 解析
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse(response: Any, *, model: str) -> AssistantTurn:
        choice = response.choices[0]
        message = choice.message
        raw_calls = getattr(message, "tool_calls", None) or []
        tool_calls: list[dict[str, Any]] = []
        for call in raw_calls:
            function = getattr(call, "function", None)
            tool_calls.append(
                {
                    "id": getattr(call, "id", ""),
                    "type": "function",
                    "function": {
                        "name": getattr(function, "name", ""),
                        "arguments": getattr(function, "arguments", "") or "{}",
                    },
                }
            )
        usage_obj = getattr(response, "usage", None)
        hit = int(getattr(usage_obj, "prompt_cache_hit_tokens", 0) or 0)
        miss = int(getattr(usage_obj, "prompt_cache_miss_tokens", 0) or 0)
        if hit == 0 and miss == 0:  # 兜底：老字段
            miss = int(getattr(usage_obj, "prompt_tokens", 0) or 0)
        usage = Usage(
            hit=hit,
            miss=miss,
            out=int(getattr(usage_obj, "completion_tokens", 0) or 0),
            calls=1,
        )
        return AssistantTurn(
            content=getattr(message, "content", "") or "",
            reasoning_content=getattr(message, "reasoning_content", "") or "",
            tool_calls=tool_calls,
            usage=usage,
            model=model,
            finish_reason=getattr(choice, "finish_reason", "") or "",
        )
