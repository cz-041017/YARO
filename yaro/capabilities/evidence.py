"""证据渲染。

"证据"不是把工具输出贴给模型，而是转成模型**能读**的形态：

- 图（渲染后的标注图 / 热力图 / contact sheet）→ 视觉通道，detail=low 省钱
- 表（紧凑数值行，≤20 行）→ 文本通道，精确
- 标量（RMS、覆盖率、mAP）→ 文本通道，最便宜

三级策略（成本与信息量的权衡）：

- L1：contact sheet，detail=low，每轮都送（约 1024 token/图封顶）
- L2：单张原图 original，仅在指标超阈值时才拉
- L3：紧凑数值表，永远附带
"""

from __future__ import annotations

import base64
from typing import Any, Iterable, Sequence

from ..contracts import ArtifactRef, Evidence, ToolResult, estimate_tokens
from ..memory.artifacts import ArtifactStore


def estimate_messages_tokens(messages: Sequence[dict[str, Any]]) -> int:
    """估算整段请求的 token。图像按官方上限 1024/图计。"""
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    total += estimate_tokens(str(part))
                elif part.get("type") in ("image_url", "input_image"):
                    total += 1024
                else:
                    total += estimate_tokens(str(part.get("text", "")))
        else:
            total += estimate_tokens(str(content or ""))
        if message.get("reasoning_content"):
            total += estimate_tokens(str(message["reasoning_content"]))
    return total


def render_vision_parts(
    images: Iterable[dict[str, Any]],
    store: ArtifactStore,
    *,
    max_images: int = 2,
    detail: str = "low",
) -> list[dict[str, Any]]:
    """把产物图片渲染成 OpenAI 兼容的 user content parts。

    只接受 ``{"uri","sha256","media_type"}`` 形态的引用；base64 在运行时现读现编码，
    **绝不写进会话事件**（否则事件日志会爆炸，且原图会出域）。
    """
    parts: list[dict[str, Any]] = []
    used = 0
    for item in images:
        if used >= max_images:
            break
        try:
            ref = ArtifactRef.from_dict(
                {
                    "uri": item["uri"],
                    "sha256": item["sha256"],
                    "media_type": item["media_type"],
                    "size": int(item.get("size", 0)),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
        path = store.path(ref)
        if not path.exists():
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{ref.media_type};base64,{encoded}", "detail": detail},
            }
        )
        used += 1
    if parts:
        parts.insert(0, {"type": "text", "text": "以下是本轮工具产生的视觉证据（标注图/热力图/掩膜叠加），请先看图再下结论。"})
    return parts


def render_scalar_table(metrics: dict[str, float], caption: str = "指标") -> Evidence:
    return Evidence(
        kind="table",
        payload=[{"metric": key, "value": value} for key, value in sorted(metrics.items())],
        caption=caption,
    )


def summarize_result(result: ToolResult, *, max_chars: int = 4000) -> str:
    """工具结果的文本形态：统一入口，避免各处各自截断。"""
    return result.to_text(max_chars=max_chars)
