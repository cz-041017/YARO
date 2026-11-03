"""评估器：分数的唯一可信来源。

三条硬规矩（对应 CONTRACT.md）：

1. **分数只能由 harness 计算**，模型自报的数字一律不采信；
2. **先固定预算再比分数**——不同预算下的分数不可比（这也是 RE-Bench 的教训）;
3. **噪声之下的差异不是差异**：每个指标都要有 ``noise_floor``，否则会为了 0.02pp 的抖动
   反复修改管线，把时间烧在噪声上。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

Direction = Literal["higher", "lower"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    direction: Direction
    tolerance: float = 0.0
    noise_floor: float = 0.0
    unit: str = ""
    hard_limit: float | None = None  # 越过即不可交付（如漏检率 0.1%）

    def better(self, candidate: float, baseline: float) -> bool:
        significant = max(self.tolerance, self.noise_floor)
        delta = candidate - baseline
        if abs(delta) <= significant:
            return False
        return delta > 0 if self.direction == "higher" else delta < 0


@dataclass
class Observation:
    """一次测量的结果。``cost_usd``/``wall_s`` 让"性价比"成为可比较的一等指标。"""

    value: float
    n: int = 1
    cost_usd: float = 0.0
    wall_s: float = 0.0


@dataclass(frozen=True)
class Verdict:
    metric: str
    status: str  # improved | same | worse | blocked
    candidate: float
    baseline: float
    delta: float
    significant: bool
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.status != "blocked"

    def format(self, spec: MetricSpec) -> str:
        arrow = "+" if self.delta >= 0 else ""
        return (
            f"{self.metric}: {self.baseline:.4g}{spec.unit} → {self.candidate:.4g}{spec.unit} "
            f"({arrow}{self.delta:.4g}) {self.status}" + (f" —— {self.reason}" if self.reason else "")
        )


@dataclass
class GateDecision:
    ok: bool
    verdicts: list[Verdict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [verdict.format(METRIC_LOOKUP.get(verdict.metric, MetricSpec(verdict.metric, "higher"))) for verdict in self.verdicts]
        head = "回归闸门通过" if self.ok else "回归闸门拦截"
        return head + ("\n" + "\n".join(lines) if lines else "") + (("\n原因：" + "；".join(self.reasons)) if self.reasons else "")


METRIC_LOOKUP: dict[str, MetricSpec] = {}


def compare(
    spec: MetricSpec,
    baseline: Observation,
    candidate: Observation,
    *,
    min_n: int = 3,
) -> Verdict:
    """单指标比较。样本不足时判 blocked：宁可不给结论，也不给假结论。"""
    METRIC_LOOKUP[spec.name] = spec
    if candidate.n < min_n:
        return Verdict(
            metric=spec.name,
            status="blocked",
            candidate=candidate.value,
            baseline=baseline.value,
            delta=candidate.value - baseline.value,
            significant=False,
            reason=f"样本量不足（n={candidate.n} < {min_n}），先跑够样本再下结论",
        )
    delta = candidate.value - baseline.value
    significant = abs(delta) > max(spec.tolerance, spec.noise_floor)
    if not significant:
        return Verdict(spec.name, "same", candidate.value, baseline.value, delta, False, "差异在噪声底线内")
    improved = delta > 0 if spec.direction == "higher" else delta < 0
    return Verdict(
        spec.name,
        "improved" if improved else "worse",
        candidate.value,
        baseline.value,
        delta,
        True,
        "" if improved else "超过容差但方向变差",
    )


def regression_gate(
    specs: Sequence[MetricSpec],
    baseline: Mapping[str, Observation],
    candidate: Mapping[str, Observation],
    *,
    allow_worse: Iterable[str] = (),
    min_n: int = 3,
) -> GateDecision:
    """多指标回归判定：任一关键指标变差或越过 hard_limit 就拦截。"""
    allow = set(allow_worse)
    verdicts: list[Verdict] = []
    reasons: list[str] = []

    for spec in specs:
        METRIC_LOOKUP[spec.name] = spec
        base = baseline.get(spec.name)
        cand = candidate.get(spec.name)
        if base is None or cand is None:
            reasons.append(f"缺少指标 {spec.name} 的基线或候选值")
            verdicts.append(Verdict(spec.name, "blocked", cand.value if cand else float("nan"), base.value if base else float("nan"), float("nan"), False, "数据缺失"))
            continue
        verdict = compare(spec, base, cand, min_n=min_n)
        if spec.hard_limit is not None and cand.value > spec.hard_limit:
            verdict = Verdict(
                spec.name,
                "blocked",
                cand.value,
                base.value,
                cand.value - base.value,
                True,
                f"越过硬上限 {spec.hard_limit}{spec.unit}，不可交付",
            )
        if verdict.status == "blocked":
            reasons.append(verdict.reason or f"{spec.name} 被拦")
        elif verdict.status == "worse" and spec.name not in allow:
            reasons.append(f"{spec.name} 变差 {verdict.delta:+.4g}{spec.unit}")
        verdicts.append(verdict)

    return GateDecision(ok=not reasons, verdicts=verdicts, reasons=reasons)


def pareto_better(candidate: Mapping[str, float], baseline: Mapping[str, float]) -> bool:
    """成本-精度前沿：候选在所有维度不劣，且至少一维严格更优。"""
    if set(candidate) != set(baseline):
        raise ValueError("比较的两点必须具有相同维度")
    no_worse = all(candidate[key] <= baseline[key] for key in candidate)
    strictly_better = any(candidate[key] < baseline[key] for key in candidate)
    return no_worse and strictly_better


def cost_per_point(observation: Observation) -> float:
    """每提升 1 个点需要多少钱——把"贵不贵"变成可比数字。"""
    if observation.value <= 0:
        return float("inf")
    return observation.cost_usd / observation.value


def fixed_budget_compare(
    runs: Sequence[tuple[float, Observation]],
    *,
    budget_usd: float,
) -> Observation | None:
    """在相同预算下比较（RE-Bench 的做法）。

    ``runs`` 是 ``(累计花费, 该花费下的观察值)``；返回预算内最好的那次观察。
    跨预算比较分数没有意义，这条函数存在的目的就是让调用方无法偷懒。
    """
    eligible = [observation for cost, observation in runs if cost <= budget_usd]
    if not eligible:
        return None
    return max(eligible, key=lambda observation: observation.value)


def to_dict(decision: GateDecision) -> dict[str, Any]:
    return {
        "ok": decision.ok,
        "reasons": list(decision.reasons),
        "verdicts": [
            {
                "metric": verdict.metric,
                "status": verdict.status,
                "candidate": verdict.candidate,
                "baseline": verdict.baseline,
                "delta": verdict.delta,
            }
            for verdict in decision.verdicts
        ],
    }
