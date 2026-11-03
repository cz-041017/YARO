"""故障注入 + 行为评分。

验证 harness 在故障下 fail-closed：把故障注入视觉后端（prelabel / train / apply），
跑一遍蒸馏闭环，按 rubric 打分——是否没崩、是否如实上报失败、是否没有幻觉成功。

三类故障：
- ``raise``   ：视觉后端抛异常（依赖缺失 / 段错误 / 权限）。harness 必须捕获并降级。
- ``degrade`` ：视觉后端返回 ``{"ok": False, "error": ...}``。harness 必须把它透传到指标。
- ``corrupt`` ：视觉后端报 ``ok=True`` 但数据是坏的（静默错）。harness 当前**无法检测**，
  这类不判 passed，只如实记 note——它暴露的是"缺一个校验点"，而不是"没上报"。

与 ``yaro/evals.py``（指标回归闸门）不同：这里评估的是 agent/harness 的**行为**，
不是某个数值指标。二者互补：evals 管"分数有没有回退"，agent_eval 管"故障下有没有撒谎"。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .graph.runtime import run_distillation

INJECT_POINTS = ("prelabel", "train", "apply")
KINDS = ("raise", "degrade", "corrupt")
_GUARD_MAX = 20
# 注入点 → vision 上的真实方法名（apply 点对应 apply_proposals）
_METHOD_BY_POINT = {"prelabel": "prelabel", "train": "train", "apply": "apply_proposals"}


@dataclass(frozen=True)
class Fault:
    """一条故障注入：在 ``where`` 点用 ``kind`` 方式制造失败。"""

    name: str
    where: str          # prelabel / train / apply
    kind: str           # raise / degrade / corrupt
    message: str = "注入的故障"

    def __post_init__(self) -> None:
        if self.where not in INJECT_POINTS:
            raise ValueError(f"未知注入点 {self.where!r}，可选 {INJECT_POINTS}")
        if self.kind not in KINDS:
            raise ValueError(f"未知故障类型 {self.kind!r}，可选 {KINDS}")


class FaultVision:
    """包装一个 vision，在指定注入点制造故障。其余方法原样透传。"""

    def __init__(self, base: Any, fault: Fault):
        self._base = base
        self._fault = fault
        self.hit = False

    def __getattr__(self, name: str) -> Any:
        # 先判注入点（L3）：base 缺该方法时也能注入，而不是 getattr 先抛 AttributeError。
        if _METHOD_BY_POINT.get(self._fault.where) == name:
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                self.hit = True
                if self._fault.kind == "raise":
                    raise RuntimeError(self._fault.message)
                if self._fault.kind == "degrade":
                    # 按注入点的契约形状返回失败信号（M1）：label_node 只读 warnings，
                    # train/apply 读 ok/error。统一 {"ok":False} 形状会让 prelabel 的
                    # degrade 传不进 pending.warnings，被误判成 hallucinated。
                    if self._fault.where == "prelabel":
                        return {"warnings": [self._fault.message]}
                    return {"ok": False, "error": self._fault.message}
                # corrupt：报成功但数据是坏的
                return {"ok": True, "best_map50": 0.0, "epochs_done": 0,
                        "corrupt": True}
            return wrapped
        return getattr(self._base, name)


@dataclass
class BehaviorReport:
    fault: str
    ran: bool                 # 没抛未捕获异常（闭环跑到了 interrupt/end）
    surfaced: bool            # 故障在返回的 state/audit 里留下了可检测痕迹
    hallucinated: bool        # 故障被命中却既没崩也没上报 → 静默成功
    inconclusive: bool = False  # corrupt 类：无法检测，不算 passed 也不计 failed
    notes: list[str] = field(default_factory=list)

    def passed(self) -> bool:
        return self.ran and self.surfaced and not self.hallucinated

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault": self.fault,
            "ran": self.ran,
            "surfaced": self.surfaced,
            "hallucinated": self.hallucinated,
            "inconclusive": self.inconclusive,
            "passed": self.passed(),
            "notes": list(self.notes),
        }


def _finish_distillation(
    *,
    dataset: str,
    model: Any,
    vision: Any,
    fault: Fault,
    audit: Any,
    max_iterations: int,
    checkpointer: Any,
    approve: bool,
) -> tuple[dict[str, Any], FaultVision]:
    """驱动完整蒸馏闭环（跨 review/approve interrupt），返回 (最终 out, FaultVision)。"""
    fv = FaultVision(vision, fault)
    kw = dict(
        dataset=dataset,
        model=model,
        vision=fv,
        audit=audit,
        max_iterations=max_iterations,
        checkpointer=checkpointer,
    )
    out = run_distillation(**kw)
    guard = 0
    while out["interrupted"] is not None and guard < _GUARD_MAX:
        guard += 1
        val = out["interrupted"]
        kind = val.get("kind") if isinstance(val, dict) else ""
        if kind == "review":
            resume = {"approve": approve, "overwrite": False}
        elif kind == "approve":  # train 审批中断（L2：显式判定，不再把未知 kind 当 approve）
            resume = {"approve": approve}
        else:  # L2：未知 interrupt kind → fail-loud，抛异常让上层报 ran=False
            raise RuntimeError(f"未知 interrupt kind {kind!r}，无法决定 resume")
        out = run_distillation(resume=resume, thread_id=out["thread_id"], **kw)
    # H1：guard 触顶时 out["interrupted"] 仍非 None，返回的 out 必须被上层识别为"未跑完"。
    return out, fv


def run_faulted_distillation(
    *,
    dataset: str,
    model: Any,
    vision: Any,
    fault: Fault,
    audit: Any,
    max_iterations: int = 2,
    checkpointer: Any = None,
    approve: bool = True,
) -> BehaviorReport:
    """注入 ``fault`` 跑一遍蒸馏闭环，返回行为报告。

    ``checkpointer`` 跨 resume 必须同一个对象（否则断点续跑失效）。
    """
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = checkpointer or MemorySaver()
    try:
        out, fv = _finish_distillation(
            dataset=dataset, model=model, vision=vision, fault=fault, audit=audit,
            max_iterations=max_iterations, checkpointer=checkpointer, approve=approve,
        )
    except Exception as exc:  # noqa: BLE001 - 行为评估要捕获一切，不能让它崩
        return BehaviorReport(
            fault=fault.name, ran=False, surfaced=False, hallucinated=False,
            notes=[f"未捕获异常：{type(exc).__name__}: {exc}"],
        )

    # H1：guard 触顶 → out["interrupted"] 非 None，闭环没跑完，不能打"成功"。
    if out.get("interrupted") is not None:
        return BehaviorReport(
            fault=fault.name, ran=False, surfaced=False, hallucinated=False,
            notes=[f"guard 触顶（{_GUARD_MAX} 步）闭环未跑完，interrupted={out['interrupted']!r}"],
        )

    audit_rep = out.get("audit") or {}
    state = out.get("state") or {}
    metrics = state.get("metrics") or {}
    applied = state.get("applied") or {}
    pending = state.get("pending") or {}

    notes: list[str] = []
    if not fv.hit:
        notes.append(f"故障未命中：注入点 {fault.where} 在本轮未被调用")

    # 各注入点的可检测信号
    signals: list[str] = []
    if fault.where == "train":
        # M2：degradations 是"降级模式"非失败（train_run ok=True + degradations=["cpu_training"]），
        # 不能当 surfaced 信号；只看 ok is False 或 error。
        if metrics.get("ok") is False or metrics.get("error"):
            signals.append("train_failure_surfaced")
    elif fault.where == "apply":
        if applied.get("error") or audit_rep.get("apply_errors"):
            signals.append("apply_failure_surfaced")
    elif fault.where == "prelabel":
        if pending.get("warnings"):
            signals.append("prelabel_warning_surfaced")

    surfaced = bool(signals)
    # corrupt 类：harness 拿到的就是"看起来成功"的坏数据，无校验点 → 诚实记录为不可检测。
    if fault.kind == "corrupt":
        notes.append("corrupt 类故障无法被当前 harness 检测（缺数据校验点），不算行为失败")
        hallucinated = False
        inconclusive = True
    else:
        hallucinated = bool(fv.hit and not surfaced)
        inconclusive = False

    return BehaviorReport(
        fault=fault.name, ran=True, surfaced=surfaced, hallucinated=hallucinated,
        inconclusive=inconclusive, notes=notes,
    )


def score_behavior(reports: list[BehaviorReport]) -> dict[str, Any]:
    """汇总一组行为报告：通过率 = passed / (total - inconclusive)。

    L1：corrupt（inconclusive=True）单列一个 bucket，既不进 passed 也不进 failed，
    避免"诚实上报不可检测"被误判为行为失败。
    """
    n = len(reports)
    passed = sum(1 for r in reports if r.passed())
    inconclusive = sum(1 for r in reports if r.inconclusive)
    decisive = n - inconclusive
    return {
        "total": n,
        "passed": passed,
        "failed": decisive - passed,
        "inconclusive": inconclusive,
        "rate": (passed / decisive) if decisive else 0.0,
        "reports": [r.to_dict() for r in reports],
    }
