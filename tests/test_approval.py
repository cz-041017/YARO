"""审批策略（PLAN / DEFAULT / YOLO）的离线单测。"""

from __future__ import annotations

import pytest

from yaro.approval import ApprovalPolicy, ApprovalRule, build_policy


def test_plan_asks_everything() -> None:
    p = ApprovalPolicy(mode="plan")
    assert p.decide("list_files") == "ask"
    assert p.decide("dataset_scan") == "ask"
    assert p.decide("run_python") == "ask"


def test_default_asks_destructive_allows_readonly() -> None:
    p = ApprovalPolicy(mode="default")
    assert p.decide("list_files") == "allow"
    assert p.decide("dataset_scan") == "allow"
    assert p.decide("dataset_audit") == "allow"
    assert p.decide("run_python") == "ask"
    assert p.decide("train_run") == "ask"
    assert p.decide("deploy_run") == "ask"


def test_yolo_allows_everything_but_deny_floor() -> None:
    p = ApprovalPolicy(mode="yolo")
    assert p.decide("run_python") == "allow"
    assert p.decide("train_run") == "allow"
    # 硬地板：即使 yolo，deny 规则也生效。
    p2 = ApprovalPolicy(mode="yolo", rules=(ApprovalRule("run_python", "deny", "floor"),))
    assert p2.decide("run_python") == "deny"


def test_explicit_rules_override_mode() -> None:
    # 默认模式下，显式 allow 覆盖"危险工具要问"。
    p = build_policy(
        "default",
        extra_rules=[{"match": "run_python", "decision": "allow", "reason": "whitelisted"}],
    )
    assert p.decide("run_python") == "allow"
    # plan 模式下显式 allow 也生效。
    p2 = build_policy("plan", extra_rules=[{"match": "list_files", "decision": "allow"}])
    assert p2.decide("list_files") == "allow"


def test_prefix_match() -> None:
    p = build_policy("default", extra_rules=[{"match": "dataset_", "decision": "ask"}])
    assert p.decide("dataset_scan") == "ask"
    assert p.decide("dataset_render") == "ask"
    assert p.decide("tune_plan") == "allow"  # 不被 dataset_ 前缀误伤


def test_deny_always_wins_over_mode_and_allow() -> None:
    p = ApprovalPolicy(
        mode="yolo",
        rules=(
            ApprovalRule("run_python", "allow", "x"),
            ApprovalRule("run_python", "deny", "hard floor"),
        ),
    )
    # deny 在 allow 之后写入，但 decide 先扫 deny，所以 deny 胜出。
    assert p.decide("run_python") == "deny"


def test_requires_approval() -> None:
    p = ApprovalPolicy(mode="default")
    assert p.requires_approval("train_run")
    assert not p.requires_approval("list_files")


def test_to_from_dict_roundtrip() -> None:
    p = build_policy("default", extra_rules=[{"match": "run_python", "decision": "allow", "reason": "r"}])
    p2 = ApprovalPolicy.from_dict(p.to_dict())
    assert p2.mode == "default"
    assert p2.decide("run_python") == "allow"
    assert p2.decide("train_run") == "ask"


def test_invalid_mode_rejected() -> None:
    with pytest.raises(ValueError):
        ApprovalPolicy(mode="bogus")  # type: ignore[arg-type]


def test_invalid_rule_decision_rejected() -> None:
    with pytest.raises(ValueError):
        build_policy("default", extra_rules=[{"match": "x", "decision": "maybe"}])


def test_from_dict_invalid_decision_rejected() -> None:
    with pytest.raises(ValueError):
        ApprovalPolicy.from_dict({"mode": "default", "rules": [{"match": "x", "decision": "nope"}]})


def test_later_allow_overrides_earlier_ask() -> None:
    """H1 回归：allow/ask 之间后写的覆盖先写的。"""
    p = ApprovalPolicy(
        mode="default",
        rules=(ApprovalRule("run_python", "ask", "默认"), ApprovalRule("run_python", "allow", "白名单")),
    )
    assert p.decide("run_python") == "allow"


def test_later_ask_overrides_earlier_allow() -> None:
    p = ApprovalPolicy(
        mode="default",
        rules=(ApprovalRule("run_python", "allow", "先放"), ApprovalRule("run_python", "ask", "后收紧")),
    )
    assert p.decide("run_python") == "ask"


def test_deny_always_wins_regardless_of_order() -> None:
    """deny 硬地板永远生效，即使写在 allow 之前（或之后）。"""
    p = ApprovalPolicy(
        mode="default",
        rules=(ApprovalRule("run_python", "deny", "硬地板"), ApprovalRule("run_python", "allow", "白名单")),
    )
    assert p.decide("run_python") == "deny"


def test_empty_match_rejected() -> None:
    """M1 回归：空 match 会命中所有工具，必须拒绝。"""
    with pytest.raises(ValueError):
        ApprovalRule("", "deny", "")
    with pytest.raises(ValueError):
        build_policy("default", extra_rules=[{"match": "", "decision": "deny"}])
    with pytest.raises(ValueError):
        ApprovalPolicy.from_dict({"mode": "default", "rules": [{"match": "", "decision": "deny"}]})


def test_missing_match_rejected() -> None:
    """L2 回归：缺 match 字段应报清晰错误，而不是 KeyError。"""
    with pytest.raises(ValueError):
        ApprovalPolicy.from_dict({"mode": "default", "rules": [{"decision": "deny"}]})


def test_exact_match() -> None:
    """M2 回归：exact=True 只命中精确名，不命中前缀。"""
    p = ApprovalPolicy(
        mode="default",
        rules=(ApprovalRule("run_python", "deny", "只禁 run_python", exact=True),),
    )
    assert p.decide("run_python") == "deny"
    assert p.decide("run_python_extra") == "ask"  # 前缀不命中，落 default 危险前缀


def test_default_classifies_write_tools_as_ask() -> None:
    """H2 回归：dataset_render / train_plan / deploy_plan 都写文件，默认必须 ask。"""
    p = ApprovalPolicy(mode="default")
    for name in ("dataset_render", "train_plan", "deploy_plan", "run_python", "train_run"):
        assert p.decide(name) == "ask", f"{name} 应该 ask"


def test_default_classifies_readonly_tools_as_allow() -> None:
    p = ApprovalPolicy(mode="default")
    for name in ("list_files", "image_probe", "calib_assess", "dataset_scan", "dataset_audit", "train_analyze", "tune_plan"):
        assert p.decide(name) == "allow", f"{name} 应该 allow"


def test_real_18_tools_all_classified() -> None:
    """盲点回归：真实 18 个工具名都必须落到 ask 或 allow，无一例外。"""
    p = ApprovalPolicy(mode="default")
    real = [
        "list_files", "image_probe", "calib_assess", "run_python",
        "dataset_scan", "dataset_audit", "dataset_split", "dataset_dedupe",
        "dataset_render", "label_convert", "prelabel_run",
        "train_plan", "train_run", "train_analyze",
        "tune_plan", "tune_run", "deploy_plan", "deploy_run",
    ]
    for name in real:
        assert p.decide(name) in ("allow", "ask"), f"{name} 未正确分类"
