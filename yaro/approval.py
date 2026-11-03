"""工具审批策略（PLAN / DEFAULT / YOLO），数据驱动 + 硬地板。

参照 arXiv:2609.00006 的 R9/R11：三模式审批（plan 全部问 / default 危险问 / yolo 全放），
权限规则写成**数据**（可序列化的 dict 列表）而非命令式代码；YOLO 也保留 denylist 硬地板
—— 即便 yolo，命中 ``deny`` 的工具仍然拒绝（防御纵深，不是信任模式本身）。

与其它模块的分工：
- :mod:`yaro.provider.guard.GuardedModel` 管"出网前脱敏"（消息离开进程前）；
- 本模块管"工具执行前是否要人批"（副作用发生前）；
- :mod:`yaro.gates` 管"生成的代码是否合规"（静态扫描）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

ApprovalMode = Literal["plan", "default", "yolo"]
Decision = Literal["allow", "ask", "deny"]

_MODES: tuple[ApprovalMode, ...] = ("plan", "default", "yolo")

# 危险前缀：会写文件 / 跑代码 / 训练 / 导出 / 长时间占资源。默认模式下默认要问。
# （dataset_render 写渲染图、train_plan/deploy_plan 写计划文件 → 都是"写"，故在此列。）
DEFAULT_ASK_PREFIXES: tuple[str, ...] = (
    "run_python",
    "train_run",
    "tune_run",
    "deploy_run",
    "prelabel_run",
    "dataset_split",   # apply=True 时改布局
    "dataset_dedupe",  # apply=True 时移动文件
    "label_convert",   # 写导出文件
    "dataset_render",  # 写 runs/review 渲染图 + contact sheet
    "train_plan",      # 写 recipe.json / train.py / acceptance.md
    "deploy_plan",     # 写 export/bench 脚本 + Dockerfile
)

# 只读/纯查询前缀：默认模式下直接放行（不写任何文件）。
DEFAULT_ALLOW_PREFIXES: tuple[str, ...] = (
    "list_files",
    "image_probe",
    "calib_assess",
    "dataset_scan",
    "dataset_audit",
    "train_analyze",
    "tune_plan",
)

# 硬地板：任何模式下都不允许（当前为空——本地工具无真正"不可做"项；保留机制供未来接
# 网络写/越权操作时启用）。规则是数据，改这里或 from_dict 注入即可，不改逻辑。
DEFAULT_DENY_PREFIXES: tuple[str, ...] = ()


@dataclass(frozen=True)
class ApprovalRule:
    """一条规则：默认按前缀命中，``exact=True`` 时按精确名命中。"""

    match: str
    decision: Decision
    reason: str = ""
    exact: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.match, str) or not self.match:
            raise ValueError("规则的 match 不能为空字符串（空串会命中所有工具）")

    def hits(self, tool_name: str) -> bool:
        if self.exact:
            return tool_name == self.match
        return tool_name == self.match or tool_name.startswith(self.match)


@dataclass(frozen=True)
class ApprovalPolicy:
    """工具审批策略：deny 硬地板 → 显式规则 → 模式默认。

    优先级（按从高到低）：
    1. ``deny`` 规则**永远生效**（即使 yolo，防御纵深，不受顺序影响）；
    2. ``allow``/``ask`` 规则之间**后写的覆盖先写的**（覆盖语义）；
    3. 落到模式默认（plan 全问 / yolo 全放 / default 危险工具问）。
    """

    mode: ApprovalMode = "default"
    rules: tuple[ApprovalRule, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"未知审批模式：{self.mode!r}，可选 {_MODES}")

    def decide(self, tool_name: str, arguments: dict[str, Any] | None = None) -> Decision:
        """决定某个工具调用是否放行。纯函数，无副作用。

        ``arguments`` 目前不参与判定（保留给未来按参数条件化，如 apply=False 的切分可放行）。
        """

        # 1) deny 硬地板永远生效。
        for rule in self.rules:
            if rule.hits(tool_name) and rule.decision == "deny":
                return "deny"

        # 2) 显式 allow/ask：后写的覆盖先写的。
        decision: Decision | None = None
        for rule in self.rules:
            if rule.hits(tool_name) and rule.decision in ("allow", "ask"):
                decision = rule.decision
        if decision is not None:
            return decision

        # 3) 模式默认。
        if self.mode == "plan":
            return "ask"
        if self.mode == "yolo":
            return "allow"
        # default 模式：危险工具要问，其余放行。
        if _starts_with_any(tool_name, DEFAULT_ASK_PREFIXES):
            return "ask"
        return "allow"

    def requires_approval(self, tool_name: str, arguments: dict[str, Any] | None = None) -> bool:
        return self.decide(tool_name, arguments) == "ask"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rules": [
                {"match": r.match, "decision": r.decision, "reason": r.reason, "exact": r.exact}
                for r in self.rules
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ApprovalPolicy":
        mode = data.get("mode", "default")
        if mode not in _MODES:
            raise ValueError(f"未知审批模式：{mode!r}")
        rules: list[ApprovalRule] = []
        for item in data.get("rules") or []:
            rules.append(_rule_from_item(item))
        return cls(mode=mode, rules=tuple(rules))


def _rule_from_item(item: dict[str, Any]) -> ApprovalRule:
    decision = item.get("decision")
    if decision not in ("allow", "ask", "deny"):
        raise ValueError(f"未知规则决定：{decision!r}")
    if "match" not in item:
        raise ValueError("规则缺少 match 字段")
    match = item["match"]
    if not isinstance(match, str) or not match:
        raise ValueError("规则的 match 不能为空字符串")
    return ApprovalRule(
        match=match,
        decision=decision,
        reason=str(item.get("reason", "")),
        exact=bool(item.get("exact", False)),
    )


def _starts_with_any(name: str, prefixes: Iterable[str]) -> bool:
    return any(name == p or name.startswith(p) for p in prefixes)


def build_policy(mode: ApprovalMode = "default", *, extra_rules: Iterable[dict[str, Any]] = ()) -> ApprovalPolicy:
    """从模式 + 额外规则构造策略；额外规则覆盖模式默认。"""
    rules: list[ApprovalRule] = []
    for item in extra_rules:
        rules.append(_rule_from_item(item))
    return ApprovalPolicy(mode=mode, rules=tuple(rules))
