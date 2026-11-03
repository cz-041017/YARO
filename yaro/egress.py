"""出网边界（egress）与隐私过滤。

设计目标：**默认不把任何东西送出本机**，只有显式授权才放行；无论模型跑在
本地还是云端，原始图像（raw）与绝对路径都受同一套规则约束。

四种模式：

=========  =========================================================
off        完全不出网。只能用本地模型（localhost / 私网）或演示模式。
text       允许请求打到模型服务，但只允许文本；图片（raw + derived）全部剥离。
derived    额外允许**派生证据图**（harness 渲染的诊断图 / contact sheet）；
           原始输入图仍然剥离。
full       允许原始证据图（仍然会剥离绝对路径与密钥）。
=========  =========================================================

公网端点（非 localhost / 非私网）在 ``off`` 模式下直接拒绝；其余模式也需要
``allow_public=True`` 才放行——这是"显式授权"的第二道闸，防止配置漂移把
客户数据悄悄送到公网。

本模块刻意零第三方依赖，只用 ``ipaddress``/``urllib.parse``/``re``/``copy``。
"""

from __future__ import annotations

import copy
import ipaddress
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

from .contracts import ContractError

MODE_OFF = "off"
MODE_TEXT = "text"
MODE_DERIVED = "derived"
MODE_FULL = "full"

MODES: tuple[str, ...] = (MODE_OFF, MODE_TEXT, MODE_DERIVED, MODE_FULL)

MODE_LABELS: dict[str, str] = {
    MODE_OFF: "不出网（仅本地模型 / 演示）",
    MODE_TEXT: "只发文本，图片一律剥离",
    MODE_DERIVED: "允许派生证据图，原图仍剥离",
    MODE_FULL: "允许原始证据图",
}

# 敏感度标签：raw = 用户原始输入图；derived = harness 渲染出来的诊断图。
RAW = "raw"
DERIVED = "derived"

_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),  # 常见 API Key 形态
    re.compile(r"(?i)\b(?:bearer|authorization:)\s+[A-Za-z0-9._\-]{12,}"),
)

_PRIVATE_SUFFIXES = (".local", ".internal", ".lan", ".home", ".localhost")


def normalize_mode(value: str | None) -> str:
    """校验模式名；非法值直接报错，不静默降级（配置错误必须可见）。"""
    mode = (value or "").strip().lower() or MODE_OFF
    if mode not in MODES:
        raise ContractError(
            f"未知的出网模式：{value!r}；可选 {', '.join(MODES)}"
        )
    return mode


def host_of(base_url: str) -> str:
    """从 base_url 取出主机名（失败时返回空串，让调用方按公网处理）。"""
    try:
        parts = urlsplit(base_url if "://" in base_url else f"https://{base_url}")
    except ValueError:
        return ""
    return (parts.hostname or "").strip().lower()


def is_private_host(host: str) -> bool:
    """判断是否本机/私网。判断不出来时**按公网处理**（fail-closed）。"""
    if not host:
        return False
    if host in ("localhost", "host.docker.internal", "gateway.docker.internal"):
        return True
    if host.endswith(_PRIVATE_SUFFIXES):
        return True
    try:
        ip = ipaddress.ip_address(host.strip("[]"))
    except ValueError:
        return False
    return bool(ip.is_loopback or ip.is_private or ip.is_link_local)


@dataclass(frozen=True)
class EgressPolicy:
    """一次运行期间生效的出网策略（不可变；改策略要新建对象）。"""

    mode: str = MODE_OFF
    allow_public: bool = False
    # (needle, replacement)：通常是工作区/家目录的绝对路径，替换成占位符
    redactions: tuple[tuple[str, str], ...] = ()
    # 需要擦除的密钥原文（不会出现在日志/事件/请求里）
    secrets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", normalize_mode(self.mode))

    # ------------------------------------------------------------------ #
    # 判定
    # ------------------------------------------------------------------ #

    @property
    def image_mode(self) -> str:
        """当前模式下允许发送的证据图敏感度上限。"""
        return {
            MODE_OFF: "none",
            MODE_TEXT: "none",
            MODE_DERIVED: DERIVED,
            MODE_FULL: "any",
        }[self.mode]

    def allows_text(self) -> bool:
        return self.mode != MODE_OFF

    def allows_images(self, sensitivity: str = DERIVED) -> bool:
        limit = self.image_mode
        if limit == "none":
            return False
        if limit == "any":
            return True
        return sensitivity != RAW

    def check_endpoint(self, base_url: str) -> None:
        """请求真正发出前的最后一道闸。禁止时抛 ``ContractError``。"""
        host = host_of(base_url)
        private = is_private_host(host)
        if self.mode == MODE_OFF:
            raise ContractError(
                f"出网策略为 off：拒绝把任何请求发到 {host or base_url!r}。"
                "做法：①把它指到本机/内网模型（YARO_BASE_URL=http://127.0.0.1:8000/v1）；"
                "②或显式授权出网（yaro init --egress text --allow-public）。"
            )
        if not private and not self.allow_public:
            raise ContractError(
                f"目标 {host!r} 是公网端点，但未显式授权（allow_public=False）。"
                "隐私数据不出公网是默认策略：确认要用云端模型后，"
                "在 .env 里设置 YARO_EGRESS_ALLOW_PUBLIC=1。"
            )

    # ------------------------------------------------------------------ #
    # 脱敏
    # ------------------------------------------------------------------ #

    def redact(self, text: str) -> str:
        """擦掉绝对路径与密钥。幂等、不会因为替换结果再次命中。"""
        if not text:
            return text
        out = text
        for needle, replacement in self.redactions:
            if needle:
                out = out.replace(needle, replacement)
        for secret in self.secrets:
            if secret and len(secret) >= 8:
                out = out.replace(secret, "***")
        for pattern in _SECRET_PATTERNS:
            out = pattern.sub("***", out)
        return out

    def redact_any(self, value: Any) -> Any:
        """递归脱敏（dict/list/str），用于 tool_calls 参数这类嵌套结构。"""
        if isinstance(value, str):
            return self.redact(value)
        if isinstance(value, list):
            return [self.redact_any(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.redact_any(item) for item in value)
        if isinstance(value, dict):
            return {key: self.redact_any(item) for key, item in value.items()}
        return value

    def filter_messages(self, messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        """返回**新的**消息列表：剥离不允许的图片、脱敏文本。

        必须深拷贝——会话事件是账本，绝不能因为出网过滤而被就地改写。
        """
        filtered: list[dict[str, Any]] = []
        for message in messages:
            item = copy.deepcopy(dict(message))
            content = item.get("content")
            if isinstance(content, list):
                kept: list[dict[str, Any]] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                        continue  # 是否发图由 Agent 在注入前决定，这里兜底剥离
                    kept.append(self.redact_any(part))
                item["content"] = kept
            elif isinstance(content, str):
                item["content"] = self.redact(content)
            if isinstance(item.get("reasoning_content"), str):
                item["reasoning_content"] = self.redact(item["reasoning_content"])
            if "tool_calls" in item:
                item["tool_calls"] = self.redact_any(item["tool_calls"])
            if isinstance(item.get("refusal"), str):
                item["refusal"] = self.redact(item["refusal"])
            filtered.append(item)
        return filtered

    # ------------------------------------------------------------------ #
    # 展示
    # ------------------------------------------------------------------ #

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "label": MODE_LABELS[self.mode],
            "allow_public": self.allow_public,
            "images": self.image_mode,
        }


def build_policy(
    mode: str | None,
    *,
    allow_public: bool = False,
    redact_paths: Iterable[str | None] = (),
    secrets: Iterable[str | None] = (),
) -> EgressPolicy:
    """从配置构造策略；路径按长度倒序替换，避免前缀互相吞掉。"""
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in redact_paths:
        if not raw:
            continue
        path = str(raw).rstrip("\\/")
        if len(path) < 4 or path in seen:
            continue
        seen.add(path)
        pairs.append((path, "<workspace>" if len(pairs) == 0 else "<home>"))
    pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    return EgressPolicy(
        mode=normalize_mode(mode),
        allow_public=bool(allow_public),
        redactions=tuple(pairs),
        secrets=tuple(secret for secret in secrets if secret),
    )


def summarize_policy(policy: EgressPolicy, base_url: str) -> str:
    """给 CLI/GUI 的一行人话说明。"""
    if policy.mode == MODE_OFF:
        # off 时端点写什么都不重要——请求会被拦下。写"目标公网"容易让人以为已经出网了。
        return f"egress=off（{MODE_LABELS[policy.mode]}）· 不会发起任何模型请求（演示模式可用）"
    host = host_of(base_url)
    scope = "本机/内网" if is_private_host(host) else "公网"
    return f"egress={policy.mode}（{MODE_LABELS[policy.mode]}）· 目标 {host or '?'}（{scope}）"


__all__ = [
    "DERIVED",
    "MODES",
    "MODE_DERIVED",
    "MODE_FULL",
    "MODE_LABELS",
    "MODE_OFF",
    "MODE_TEXT",
    "RAW",
    "EgressPolicy",
    "build_policy",
    "host_of",
    "is_private_host",
    "normalize_mode",
    "summarize_policy",
]
