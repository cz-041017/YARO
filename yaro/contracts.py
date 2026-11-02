"""YARO 全局契约。

一切模块只依赖这里的类型。三条设计约束从这里推导：

1. 所有产物都是内容寻址的 ``ArtifactRef``，永不原地覆盖；
2. 工具输出必须携带 ``Evidence``（模型可读的观察），不是裸日志；
3. 任何分数（``metrics``）由 harness 计算，不接受模型自报。
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, Sequence

SCHEMA_VERSION = "yaro/1"

# --------------------------------------------------------------------------- #
# 基础工具函数
# --------------------------------------------------------------------------- #


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def new_id() -> str:
    return uuid.uuid4().hex


def now_ts() -> float:
    return time.time()


def estimate_tokens(text: str) -> int:
    """粗略 token 估算：中英混排按 ~3 字符/token。只用于预算与压缩触发，不用于计费。"""
    return max(1, len(text) // 3)


class ContractError(RuntimeError):
    """不变式被违反（例如漏回传 reasoning_content）。调用方不应吞掉该异常。"""


class DegradedBackend(RuntimeError):
    """后端不可用（依赖未安装 / 服务未启动）。

    上层应记录降级（``ToolResult.degraded``）并继续，而不是整轮崩溃。
    """


# --------------------------------------------------------------------------- #
# 产物
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArtifactRef:
    """内容寻址产物的引用。

    ``uri`` 形如 ``artifact://{run_id}/{step:04d}/{kind}/{sha256[:12]}.{ext}``
    """

    uri: str
    sha256: str
    media_type: str
    size: int

    @property
    def short(self) -> str:
        return self.sha256[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "sha256": self.sha256,
            "media_type": self.media_type,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRef":
        return cls(str(data["uri"]), str(data["sha256"]), str(data["media_type"]), int(data["size"]))


def object_key(sha256: str, ext: str) -> str:
    """CAS 存储键：``<sha[:2]>/<sha>.<ext>``（两级目录避免单目录过多文件）。"""
    return f"{sha256[:2]}/{sha256}.{ext.lstrip('.')}"


def artifact_uri(run_id: str, step: int, kind: str, sha256: str, ext: str) -> str:
    return f"artifact://{run_id}/{step:04d}/{kind}/{sha256[:12]}.{ext.lstrip('.')}"


# --------------------------------------------------------------------------- #
# 证据
# --------------------------------------------------------------------------- #

EvidenceKind = Literal["image", "table", "scalar", "text", "json"]


@dataclass(frozen=True)
class Evidence:
    """模型可读的观察。

    image / table / scalar 是主要形态；json 仅用于小结构化对象。
    禁止把原始日志、完整数组直接塞进 payload——那不是证据，是噪声。
    """

    kind: EvidenceKind
    payload: Any
    caption: str = ""

    def to_text(self) -> str:
        if self.kind == "scalar":
            value = self.payload
            shown = f"{value:.4g}" if isinstance(value, float) else str(value)
            return f"- {self.caption}: {shown}"
        if self.kind == "table":
            rows: Sequence[dict[str, Any]] = self.payload
            if not rows:
                return f"- {self.caption}: (empty)"
            cols = list(rows[0].keys())
            header = " | ".join(cols)
            body = "\n".join(" | ".join(str(row.get(c, "")) for c in cols) for row in list(rows)[:20])
            more = "" if len(rows) <= 20 else f"\n  ... {len(rows)} rows total"
            return f"- {self.caption}\n{header}\n{body}{more}"
        if self.kind == "image":
            ref = self.payload
            uri = ref.uri if isinstance(ref, ArtifactRef) else str(ref)
            return f"- [image] {self.caption} ({uri})"
        if self.kind == "json":
            return f"- {self.caption}: {json.dumps(self.payload, ensure_ascii=False)[:1200]}"
        return f"- {self.caption}: {self.payload}"


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]


class Tool(Protocol):
    """能力层接口。每个工具必须返回 ToolResult，不得抛裸异常（由 registry 兜底）。"""

    spec: ToolSpec

    async def run(self, **kwargs: Any) -> "ToolResult":  # pragma: no cover - protocol
        ...


def function_schema(spec: ToolSpec) -> dict[str, Any]:
    """转成 DeepSeek/OpenAI 的 function calling schema。"""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
        },
    }


@dataclass
class ToolResult:
    ok: bool
    summary: str
    evidence: list[Evidence] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    degraded: list[str] = field(default_factory=list)
    hint: str | None = None
    # 结果里是否含**用户原始输入图**（raw）。出网策略据此决定能不能把图发给模型：
    # raw 默认永不出网，只有 egress=full 才放行；derived（harness 渲染的诊断图）在
    # egress≥derived 时可用。详见 yaro/egress.py。
    sensitive: bool = False

    def to_text(self, max_chars: int = 4000) -> str:
        """给模型的紧凑文本表示。控制在 4K 字符内（论文实测 25K 是上限，我们更保守）。"""
        lines = [f"{'OK' if self.ok else 'ERROR'}: {self.summary}"]
        if self.metrics:
            pairs = ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in self.metrics.items()
            )
            lines.append(f"metrics: {pairs}")
        if self.degraded:
            lines.append("degraded: " + "; ".join(self.degraded))
        for ev in self.evidence:
            lines.append(ev.to_text())
        if self.artifacts:
            lines.append("artifacts: " + ", ".join(a.uri for a in self.artifacts))
        if self.hint:
            lines.append(f"next: {self.hint}")
        return "\n".join(lines)[:max_chars]


# --------------------------------------------------------------------------- #
# 用量与成本
# --------------------------------------------------------------------------- #

# USD / 1M tokens，谷时价；峰时 ×2（UTC 01:00-04:00、06:00-10:00 周一至周五）
PRICES_USD_PER_MTOK: dict[str, dict[str, float]] = {
    "deepseek-flash": {"hit": 0.003, "miss": 0.15, "out": 0.60},
    "deepseek-v4-pro": {"hit": 0.022, "miss": 0.66, "out": 1.98},
}


def ensure_known_model(model: str) -> str:
    """启动时就校验模型名：配置写错应该在启动时炸，而不是跑一半记账失败。

    价格表既是"我们认识的型号清单"，也是计费依据；两者必须是同一份数据。
    """
    if model not in PRICES_USD_PER_MTOK:
        raise ContractError(
            f"未知模型 {model!r}：可选 {sorted(PRICES_USD_PER_MTOK)}。"
            "请检查 YARO_MODEL / YARO_MODEL_PLAN / .env / CLI --model。"
        )
    return model


@dataclass
class Usage:
    """累计用量。``hit``/``miss`` 是输入缓存命中与否——这是 50× 价差的关键观测点。"""

    hit: int = 0
    miss: int = 0
    out: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> "Usage":
        self.hit += other.hit
        self.miss += other.miss
        self.out += other.out
        self.calls += other.calls
        return self

    @property
    def input_total(self) -> int:
        return self.hit + self.miss

    @property
    def hit_rate(self) -> float:
        return self.hit / self.input_total if self.input_total else 0.0

    def cost_usd(self, model: str, *, peak: bool = False) -> float:
        price = PRICES_USD_PER_MTOK.get(model)
        if price is None:
            # 未知型号必须**直接报错**，不能按 0 也不能猜一个价：
            # 记错成本会让 max_cost 预算失效（以前是 return 0.0，改个 YARO_MODEL 就"免费"），
            # 猜一个价又会让账目看起来正常、实际没人知道贵多少。
            raise ContractError(
                f"未知模型 {model!r}：无法计价。"
                f"可选 {sorted(PRICES_USD_PER_MTOK)}；"
                "模型名来自 YARO_MODEL / YARO_MODEL_PLAN 或 CLI --model。"
            )
        mult = 2.0 if peak else 1.0
        raw = self.hit * price["hit"] + self.miss * price["miss"] + self.out * price["out"]
        return raw / 1_000_000 * mult

    def to_dict(self) -> dict[str, int]:
        return {"hit": self.hit, "miss": self.miss, "out": self.out, "calls": self.calls}
