"""distill 子图（蒸馏+主动学习闭环）的离线单测，用 fake vision + fake model 驱动。"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from langgraph.types import Command

from yaro.goal import Goal
from yaro.graph.distill import build_distill_graph


class FakeVision:
    def __init__(self) -> None:
        self.prelabel_calls = 0
        self.train_calls = 0
        self.select_hard_calls = 0
        self.last_subset: Any = None

    def prelabel(self, dataset: str, **kw: Any) -> dict[str, Any]:
        self.prelabel_calls += 1
        self.last_subset = kw.get("subset")
        return {"warnings": []}

    def review_queue(self, dataset: str) -> dict[str, Any]:
        return {"pending": ["a.txt", "b.txt", "c.txt"], "count": 3}

    def train(self, dataset: str, **kw: Any) -> dict[str, Any]:
        self.train_calls += 1
        return {
            "ok": True,
            "best_map50": 0.5 + 0.05 * self.train_calls,
            "epochs_done": 10,
            "findings": [],
        }

    def apply_proposals(self, dataset: str, **kw: Any) -> dict[str, Any]:
        return {"applied": 3, "kept": 0, "overwrite": bool(kw.get("overwrite"))}

    def select_hard(self, dataset: str, **kw: Any) -> dict[str, Any]:
        self.select_hard_calls += 1
        # L1 修复：真实契约返回**完整相对路径**（含 split），不是裸 stem/扩展名。
        return {"hard": ["images/a.jpg", "images/b.jpg"], "count": 2, "scores": {}, "ok": True, "errors": []}


class FakeModel:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0

    def invoke(self, messages: Any) -> str:
        r = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return r


class CapturingModel:
    """记录 analyze 节点收到的消息：prompts 取 user 消息（数据），last_messages 存全量做结构断言。"""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls = 0
        self.prompts: list[str] = []
        self.last_messages: list[Any] = []

    def invoke(self, messages: Any) -> str:
        self.last_messages.append(messages)
        if isinstance(messages, list) and messages:
            user = next((m for m in messages if isinstance(m, dict) and m.get("role") == "user"), None)
            self.prompts.append(str(user.get("content", "")) if user else str(messages))
        else:
            self.prompts.append(str(messages))
        r = self._replies[min(self.calls, len(self._replies) - 1)]
        self.calls += 1
        return r


class FakeSkills:
    def render_index(self, max_chars: int = 1500) -> str:
        return "- dataset-audit: 体检数据集，找脏数据\n- image-probe: 查看单张图元信息"


async def _run_to_end(
    graph: Any,
    initial: dict[str, Any],
    resume: Any,
    config: dict[str, Any],
) -> dict[str, Any]:
    """跑到结束；每次挂在 interrupt 上（结果含 __interrupt__）就用 resume 续跑。"""
    state: Any = initial
    while True:
        result = await graph.ainvoke(state, config=config)
        if "__interrupt__" in result:
            val = resume(interrupt_value(result)) if callable(resume) else resume
            state = Command(resume=val)
        else:
            return result


def interrupt_value(result: dict[str, Any]) -> Any:
    """从 __interrupt__ 里取出 payload：Interrupt 对象的 .value（langgraph 1.x 契约）。"""
    items = result.get("__interrupt__") or []
    if not items:
        return None
    first = items[0]
    return first.value if hasattr(first, "value") else first


def test_single_round_stops_on_stop_reply() -> None:
    vision = FakeVision()
    model = FakeModel(["停止：mAP 已达标"])
    g = build_distill_graph(model, vision=vision, max_iterations=3)

    final = asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": True},
            {"configurable": {"thread_id": "t1"}},
        )
    )

    assert vision.prelabel_calls == 1
    assert vision.train_calls == 1
    assert final["stop_reason"] == "done"
    assert final["iteration"] == 1
    assert len(final["history"]) == 1
    assert final["history"][0]["approved"] is True


def test_loop_continues_until_stop() -> None:
    vision = FakeVision()
    model = FakeModel(["继续：lr 降一半", "继续：再加一批标注", "停止"])
    g = build_distill_graph(model, vision=vision, max_iterations=3)

    final = asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": True},
            {"configurable": {"thread_id": "t2"}},
        )
    )

    assert vision.prelabel_calls == 3
    assert vision.train_calls == 3
    assert final["iteration"] == 3
    assert len(final["history"]) == 3


def test_review_reject_stops() -> None:
    vision = FakeVision()
    model = FakeModel(["停止"])  # 无论如何都会停，但 reject 会让 continue_flag=False
    g = build_distill_graph(model, vision=vision, max_iterations=3)

    final = asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": False},
            {"configurable": {"thread_id": "t3"}},
        )
    )

    assert final["stop_reason"] == "done"
    assert final["iteration"] == 1
    assert final["history"][0]["approved"] is False


def test_max_iterations_hard_stop() -> None:
    vision = FakeVision()
    model = FakeModel(["继续"] * 10)  # 永远继续，但 max_iterations 卡住
    g = build_distill_graph(model, vision=vision, max_iterations=2)

    final = asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 2},
            {"approve": True},
            {"configurable": {"thread_id": "t4"}},
        )
    )

    assert vision.prelabel_calls == 2  # 不会超过 2 轮
    assert final["iteration"] == 2
    assert final["stop_reason"] == "max_iterations"


def test_string_no_rejects_not_approves() -> None:
    """H2 回归：人类回填字符串 "no" 必须判为拒绝，不能因非空字符串被当作通过。"""
    vision = FakeVision()
    model = FakeModel(["继续"])  # 若 approve 被误判 True 会循环；这里应因 reject 而停
    g = build_distill_graph(model, vision=vision, max_iterations=3)

    final = asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": "no"},
            {"configurable": {"thread_id": "t5"}},
        )
    )

    assert final["stop_reason"] == "done"
    assert final["iteration"] == 1
    assert final["history"][0]["approved"] is False


def test_string_yes_approves() -> None:
    vision = FakeVision()
    model = FakeModel(["停止"])  # 通过但模型说停 → done
    g = build_distill_graph(model, vision=vision, max_iterations=3)

    final = asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": "yes"},
            {"configurable": {"thread_id": "t6"}},
        )
    )

    assert final["history"][0]["approved"] is True


def test_english_continue_loops() -> None:
    """M2 回归：英文 continue 也要能继续（不只是中文"继续"）。"""
    vision = FakeVision()
    model = FakeModel(["continue: lower lr", "停止"])
    g = build_distill_graph(model, vision=vision, max_iterations=3)

    final = asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": True},
            {"configurable": {"thread_id": "t7"}},
        )
    )

    assert final["iteration"] == 2
    assert vision.train_calls == 2


def test_stop_keyword_wins_over_continue() -> None:
    """M2 回归：一句话里同时有"停止"和"继续"时，停止优先（保守）。"""
    vision = FakeVision()
    model = FakeModel(["可以停止，也可以继续"])  # 含"停止" → 停
    g = build_distill_graph(model, vision=vision, max_iterations=3)

    final = asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": True},
            {"configurable": {"thread_id": "t8"}},
        )
    )

    assert final["stop_reason"] == "done"
    assert final["iteration"] == 1


def test_prelabel_failure_degrades_not_crashes() -> None:
    """M1 回归：prelabel 抛异常应降级为 0 提案，而不是让整个闭环崩溃。"""

    class RaisingVision(FakeVision):
        def prelabel(self, dataset: str, **kw: Any) -> dict[str, Any]:
            raise RuntimeError("no backend")

    vision = RaisingVision()
    model = FakeModel(["停止"])
    g = build_distill_graph(model, vision=vision, max_iterations=3)

    final = asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": True},
            {"configurable": {"thread_id": "t9"}},
        )
    )

    assert final["pending"]["count"] == 0
    assert any("预标注失败" in w for w in final["pending"]["warnings"])


def test_train_failure_degrades_not_crashes() -> None:
    """M1 回归：train 抛异常应降级为 ok=False，而不是让整个闭环崩溃。"""

    class RaisingVision(FakeVision):
        def train(self, dataset: str, **kw: Any) -> dict[str, Any]:
            raise RuntimeError("cuda missing")

    vision = RaisingVision()
    model = FakeModel(["停止"])
    g = build_distill_graph(model, vision=vision, max_iterations=3)

    final = asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": True},
            {"configurable": {"thread_id": "t10"}},
        )
    )

    assert final["metrics"]["ok"] is False
    assert "cuda missing" in final["metrics"]["error"]


def test_select_hard_feeds_next_round_subset() -> None:
    """主动学习：analyze 说继续后，select_hard 挑难例，下一轮 label 只标这些图。"""
    vision = FakeVision()
    model = FakeModel(["继续", "停止"])
    g = build_distill_graph(model, vision=vision, max_iterations=3)

    final = asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": True},
            {"configurable": {"thread_id": "t11"}},
        )
    )

    assert vision.select_hard_calls == 1  # 只跑了一轮"继续"，挑了一次难例
    assert vision.prelabel_calls == 2      # 两轮 prelabel（第一轮全量，第二轮 subset）
    # 第二轮 prelabel 收到的 subset 必须是 select_hard 挑出的难例（完整相对路径）
    assert vision.last_subset == ["images/a.jpg", "images/b.jpg"]
    assert final["hard"] == ["images/a.jpg", "images/b.jpg"]
    assert final["active"]["count"] == 2


def test_select_hard_failure_degrades_not_crashes() -> None:
    """select_hard 抛异常应降级为空回灌，不拖垮闭环。"""

    class RaisingVision(FakeVision):
        def select_hard(self, dataset: str, **kw: Any) -> dict[str, Any]:
            raise RuntimeError("no torch")

    vision = RaisingVision()
    model = FakeModel(["继续", "停止"])
    g = build_distill_graph(model, vision=vision, max_iterations=3)

    final = asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": True},
            {"configurable": {"thread_id": "t12"}},
        )
    )

    assert final["stop_reason"] == "done"
    assert final["hard"] == []
    assert final["active"]["ok"] is False
    assert any("no torch" in e for e in final["active"]["errors"])


def test_goal_and_skills_injected_into_analyze_prompt() -> None:
    """skills + goal 应注入 analyze 的提示词：目标文本 + 技能元数据目录（不含正文）。"""
    vision = FakeVision()
    model = CapturingModel(["停止"])
    skills = FakeSkills()
    goal = Goal(goal_id="g1", objective="把 mAP50 提到 0.9 以上")
    g = build_distill_graph(model, vision=vision, max_iterations=3, skills=skills, goal=goal)

    asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": True},
            {"configurable": {"thread_id": "t13"}},
        )
    )

    prompt = model.prompts[0]
    assert "目标：把 mAP50 提到 0.9 以上" in prompt
    assert "dataset-audit" in prompt
    assert "image-probe" in prompt

    # M1 修复（f2ffc731）：指令放 system、不可信数据（objective/skills/指标）放 user，
    # 两条消息分离——不可信 objective 绝不能和指令同角色。
    msgs = model.last_messages[0]
    assert isinstance(msgs, list) and len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "目标：" not in str(msgs[0]["content"])  # system 不含不可信 objective
    assert "目标：把 mAP50 提到 0.9 以上" in str(msgs[1]["content"])


def test_analyze_prompt_without_goal_or_skills_is_plain() -> None:
    """不传 skills/goal 时，提示词不应含"目标："/"可用技能"（向后兼容，不脏 prompt）。"""
    vision = FakeVision()
    model = CapturingModel(["停止"])
    g = build_distill_graph(model, vision=vision, max_iterations=3)

    asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": True},
            {"configurable": {"thread_id": "t14"}},
        )
    )

    prompt = model.prompts[0]
    assert "目标：" not in prompt
    assert "可用技能" not in prompt


def test_goal_string_objective_accepted() -> None:
    """goal 也接受纯字符串（objective 直接从 str 取，不要求 Goal 对象）。"""
    vision = FakeVision()
    model = CapturingModel(["停止"])
    g = build_distill_graph(model, vision=vision, max_iterations=3, goal="  压缩模型到 5MB  ")

    asyncio.run(
        _run_to_end(
            g,
            {"dataset": "demo/dataset", "teacher": {}, "student": {}, "max_iterations": 3},
            {"approve": True},
            {"configurable": {"thread_id": "t15"}},
        )
    )

    assert "目标：压缩模型到 5MB" in model.prompts[0]
