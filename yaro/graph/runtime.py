"""垂直切片运行时：把蒸馏子图 + 审计账本 + 审批 + checkpointer 串起来跑。

调用方持有三样**跨 resume 共享**的对象，每次调用都传同一份，否则状态/链断裂：

- ``checkpointer``：MemorySaver（测试）或 SqliteSaver（生产，回滚/断点续跑）。
- ``audit``：同一个 :class:`yaro.audit.AuditLedger`，否则 digest 链在 resume 时接不上。
- ``thread_id``：LangGraph 会话 id，resume 必须用同一个。

每次 ``run_distillation`` 执行**一步**：从头跑到下一个 ``interrupt``，或从 ``resume=``
续跑到下一个 interrupt，或跑完全程（``interrupted is None``）。

用法（离线测试）::

    checkpointer = MemorySaver()
    audit = AuditLedger.create(tmp / "events.jsonl")
    kw = dict(dataset=..., model=fake, vision=fake, checkpointer=checkpointer, audit=audit)
    out = run_distillation(**kw)
    while out["interrupted"] is not None:
        value = _resolve(out["interrupted"])      # 按 interrupted["kind"] 决定 approve
        out = run_distillation(resume=value, thread_id=out["thread_id"], **kw)
    assert out["audit"]["ok"] and out["interrupted"] is None
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from ..audit import AuditLedger
from ..contracts import new_id
from .distill import DEFAULT_SELECT_N, _DEFAULT_VISION, build_distill_graph

_UNSET = object()


def run_distillation(
    *,
    dataset: str,
    model: Any,
    vision: Any = None,
    teacher: dict[str, Any] | None = None,
    student: dict[str, Any] | None = None,
    max_iterations: int = 2,
    select_n: int = DEFAULT_SELECT_N,
    approval: Any = None,
    audit: Any = None,
    checkpointer: Any = None,
    thread_id: str | None = None,
    resume: Any = _UNSET,
    skills: Any = None,
    goal: Any = None,
    goal_store: Any = None,
    goal_id: str | None = None,
) -> dict[str, Any]:
    """执行一步蒸馏。

    返回 ``{"state", "interrupted", "thread_id", "audit", "goal_status"}``：
    - ``interrupted`` 为 None 表示跑完；否则是 ``interrupt()`` 挂起的 payload（含 ``kind``）。
    - ``audit`` 是 ``audit.verify()`` 的报告（``ok`` / ``events`` / ``problems``）。
    - ``goal_status`` 仅在传了 ``goal_store`` + ``goal_id`` 时有意义：循环以 ``done`` 收尾时
      把目标标记 ``completed``；``max_iterations``/被 interrupt 时保持原状态（``active`` 等）；
      写回失败记为 ``error: ...``（不翻转 audit.ok——那是蒸馏自身的可信度，与目标簿记分开）。
    """
    if audit is None:
        raise TypeError("run_distillation 必须显式传入 audit（AuditLedger）：resume 时 digest 链需要同一个账本")
    checkpointer = checkpointer or MemorySaver()
    thread_id = thread_id or new_id()
    config = {"configurable": {"thread_id": thread_id}}

    audit_errors: list[dict[str, Any]] = []
    graph = build_distill_graph(
        model,
        vision=vision or _DEFAULT_VISION,
        max_iterations=max_iterations,
        approval=approval,
        audit=audit,
        audit_errors=audit_errors,
        audit_session=thread_id,
        checkpointer=checkpointer,
        skills=skills,
        goal=goal,
    )

    if resume is _UNSET:
        initial: dict[str, Any] = {
            "dataset": dataset,
            "teacher": teacher or {},
            "student": student or {},
            "max_iterations": max_iterations,
            "select_n": select_n,
        }
        state = graph.invoke(initial, config)
    else:
        state = graph.invoke(Command(resume=resume), config)

    interrupted = None
    if isinstance(state, dict) and state.get("__interrupt__"):
        items = state["__interrupt__"]
        first = items[0] if isinstance(items, (list, tuple)) and items else items
        # langgraph 1.x：__interrupt__ 里的项可能是 Interrupt 对象（.value）或裸值
        interrupted = first.value if hasattr(first, "value") else first

    # 重读磁盘账本（而非内存 _events）出审计报告：F1 修复——append 写盘失败时内存
    # 与磁盘不一致，verify 必须信磁盘；audit_errors 非空则把 ok 置 False 并带上错误。
    report = AuditLedger.open(audit.path).verify()
    if audit_errors:
        report["ok"] = False
        report["write_errors"] = audit_errors

    # H1 残留修复：review 批准了但 apply_proposals 落地失败（权限/路径/签名变化）时，
    # 闭环其实仍在旧 labels 上训练——必须 fail-closed（ok=False），否则错误只沉在
    # review 审计事件的 payload 里，调用方会误判这轮蒸馏成功。
    applied = state.get("applied") if isinstance(state, dict) else None
    if isinstance(applied, dict) and applied.get("error"):
        report["ok"] = False
        report["apply_errors"] = [applied]

    # M2 修复：select_hard 失败（ok=False：无权重/无 torch/子进程崩溃）必须传导到 report.ok，
    # 否则闭环静默退化成"每轮全量 prelabel"、调用方却看到 audit.ok=True → 误报"主动学习成功"。
    # 注意"没有未标注图"的正常返回不带 ok 键，与失败（ok=False）天然区分。
    active = state.get("active") if isinstance(state, dict) else None
    if isinstance(active, dict) and active.get("ok") is False:
        report["ok"] = False
        report["active_errors"] = active.get("errors") or []

    # Goal 写回：只有"真达成"才 complete（done + 审核通过 + 训练成功），否则保持原状态待续跑。
    # H1 修复：stop_reason=="done" 也覆盖了"人类驳回 review(approve=False)/训练失败(ok=False)/
    # 模型无结论"三种未达成情形，不能一概写 completed。
    goal_status: str | None = None
    if interrupted is None and goal_store is not None:
        if not goal_id:
            goal_status = "error: 提供了 goal_store 但 goal_id 为空"
        else:
            try:
                current = goal_store.get(goal_id)
                if current is None:
                    goal_status = "missing"
                elif current.status in ("active", "paused"):
                    review = state.get("review") if isinstance(state, dict) else {}
                    metrics = state.get("metrics") if isinstance(state, dict) else {}
                    stop_reason = state.get("stop_reason") if isinstance(state, dict) else None
                    achieved = (
                        stop_reason == "done"
                        and bool(review.get("approve"))
                        and metrics.get("ok") is True
                    )
                    if achieved:
                        goal_store.update(goal_id, current.revision, action="complete")
                        goal_status = "completed"
                    else:
                        goal_status = current.status  # 驳回/失败/触顶：不动，保持待续
                else:
                    goal_status = current.status
            except Exception as exc:  # noqa: BLE001 - 目标簿记失败不翻转 audit.ok，只透出错误
                goal_status = f"error: {exc}"

    return {
        "state": state,
        "interrupted": interrupted,
        "thread_id": thread_id,
        "audit": report,
        "goal_status": goal_status,
    }
