"""test_agent_eval：故障注入 + 行为评分（离线，无真实模型/GPU）。"""

from __future__ import annotations

import pytest

from yaro.agent_eval import Fault, run_faulted_distillation, score_behavior
from yaro.audit import AuditLedger


class FakeVision:
    """与 distill 图接口一致的假视觉后端（全部成功）。"""

    def __init__(self):
        self.prelabel_calls = 0
        self.train_calls = 0
        self.apply_calls = 0

    def prelabel(self, dataset, **teacher):
        self.prelabel_calls += 1
        return {"images": 3, "proposals": 3, "warnings": []}

    def review_queue(self, dataset):
        return {"pending": ["a.txt", "b.txt", "c.txt"], "count": 3}

    def train(self, dataset, **student):
        self.train_calls += 1
        return {
            "ok": True, "best_map50": 0.5, "epochs_done": 1,
            "findings": [], "error": None, "degradations": [],
            "stopped_reason": "epochs",
        }

    def apply_proposals(self, dataset, overwrite=False):
        self.apply_calls += 1
        return {"applied": 3, "overwrite": overwrite}

    def select_hard(self, dataset, **kw):
        return {"hard": ["a.txt", "b.txt"], "count": 2, "scores": {}, "ok": True, "errors": []}


class FakeModel:
    """analyze 节点用的假模型，恒说"停止"（跑一轮就停）。"""

    def __init__(self):
        self.invoke_calls = 0

    def invoke(self, messages, **kwargs):
        self.invoke_calls += 1
        return "停止"


class ContinueModel:
    """analyze 节点恒说"继续"——逼 guard 触顶，测截断不被误报成功。"""

    def __init__(self):
        self.invoke_calls = 0

    def invoke(self, messages, **kwargs):
        self.invoke_calls += 1
        return "继续"


@pytest.fixture()
def audit(tmp_path):
    return AuditLedger.create(tmp_path / "events.jsonl")


@pytest.fixture()
def base_kwargs(audit):
    return dict(
        dataset="demo/dataset",
        model=FakeModel(),
        vision=FakeVision(),
        audit=audit,
        max_iterations=2,
    )


def test_train_raise_is_surfaced(base_kwargs):
    rep = run_faulted_distillation(
        fault=Fault(name="train_raise", where="train", kind="raise"), **base_kwargs
    )
    assert rep.ran, rep.notes
    assert rep.surfaced, rep.notes
    assert not rep.hallucinated
    assert rep.passed()


def test_train_degrade_is_surfaced(base_kwargs):
    rep = run_faulted_distillation(
        fault=Fault(name="train_degrade", where="train", kind="degrade"), **base_kwargs
    )
    assert rep.ran
    assert rep.surfaced
    assert not rep.hallucinated
    assert rep.passed()


def test_prelabel_raise_is_surfaced(base_kwargs):
    rep = run_faulted_distillation(
        fault=Fault(name="prelabel_raise", where="prelabel", kind="raise"), **base_kwargs
    )
    assert rep.ran, rep.notes
    assert rep.surfaced, rep.notes
    assert not rep.hallucinated
    assert rep.passed()


def test_prelabel_degrade_is_surfaced(base_kwargs):
    # M1：degrade 按契约形状返回 {"warnings":[...]}，label_node 才能把它送进 pending.warnings
    rep = run_faulted_distillation(
        fault=Fault(name="prelabel_degrade", where="prelabel", kind="degrade"), **base_kwargs
    )
    assert rep.ran, rep.notes
    assert rep.surfaced, rep.notes
    assert not rep.hallucinated
    assert rep.passed()


def test_apply_raise_is_surfaced(base_kwargs):
    rep = run_faulted_distillation(
        fault=Fault(name="apply_raise", where="apply", kind="raise"), **base_kwargs
    )
    assert rep.ran, rep.notes
    assert rep.surfaced, rep.notes
    assert not rep.hallucinated
    assert rep.passed()


def test_apply_degrade_is_surfaced(base_kwargs):
    rep = run_faulted_distillation(
        fault=Fault(name="apply_degrade", where="apply", kind="degrade"), **base_kwargs
    )
    assert rep.ran, rep.notes
    assert rep.surfaced, rep.notes
    assert not rep.hallucinated
    assert rep.passed()


def test_corrupt_is_undetectable_and_not_a_failure(base_kwargs):
    rep = run_faulted_distillation(
        fault=Fault(name="train_corrupt", where="train", kind="corrupt"), **base_kwargs
    )
    assert rep.ran
    assert not rep.hallucinated
    assert not rep.passed()
    assert rep.inconclusive
    assert any("corrupt" in n for n in rep.notes)


def test_guard_cap_is_not_reported_as_success(base_kwargs, audit):
    # H1：max_iterations 远超 guard 上限 + 模型恒"继续" → 闭环截断，不能打 passed
    rep = run_faulted_distillation(
        fault=Fault(name="train_raise", where="train", kind="raise"),
        dataset="demo/dataset",
        model=ContinueModel(),
        vision=FakeVision(),
        audit=audit,
        max_iterations=100,
    )
    assert not rep.ran
    assert not rep.passed()
    assert any("guard 触顶" in n for n in rep.notes)


def test_score_behavior_summary(base_kwargs):
    reports = [
        run_faulted_distillation(fault=Fault(name="a", where="train", kind="raise"), **base_kwargs),
        run_faulted_distillation(fault=Fault(name="b", where="train", kind="degrade"), **base_kwargs),
        run_faulted_distillation(fault=Fault(name="c", where="prelabel", kind="raise"), **base_kwargs),
    ]
    summary = score_behavior(reports)
    assert summary["total"] == 3
    assert summary["passed"] == 3
    assert summary["failed"] == 0
    assert summary["rate"] == 1.0


def test_score_behavior_corrupt_is_inconclusive(base_kwargs):
    # L1：corrupt 单列 inconclusive bucket，既不进 passed 也不进 failed
    reports = [
        run_faulted_distillation(fault=Fault(name="ok", where="train", kind="raise"), **base_kwargs),
        run_faulted_distillation(fault=Fault(name="corrupt", where="train", kind="corrupt"), **base_kwargs),
    ]
    summary = score_behavior(reports)
    assert summary["total"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 0
    assert summary["inconclusive"] == 1
    assert summary["rate"] == 1.0


def test_fault_validation():
    with pytest.raises(ValueError):
        Fault(name="bad", where="nope", kind="raise")
    with pytest.raises(ValueError):
        Fault(name="bad", where="train", kind="explode")
