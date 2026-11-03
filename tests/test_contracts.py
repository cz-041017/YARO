"""契约与成本数学的单元测试。"""

from __future__ import annotations

import pytest

from yaro.contracts import (
    ArtifactRef,
    ContractError,
    Evidence,
    ToolResult,
    Usage,
    artifact_uri,
    ensure_known_model,
    estimate_tokens,
    object_key,
    sha256_hex,
)


def test_artifact_ref_roundtrip() -> None:
    ref = ArtifactRef("artifact://run/0001/mask/abcdef012345.png", "a" * 64, "image/png", 123)
    assert ArtifactRef.from_dict(ref.to_dict()) == ref
    assert ref.short == "a" * 12


def test_object_key_two_level_layout() -> None:
    assert object_key("abcdef", "png") == "ab/abcdef.png"
    assert object_key("abcdef", ".png") == "ab/abcdef.png"


def test_artifact_uri_is_short_addressed() -> None:
    uri = artifact_uri("run-1", 7, "mask", "f" * 64, "png")
    assert uri == "artifact://run-1/0007/mask/ffffffffffff.png"


def test_sha256_hex_stable() -> None:
    assert sha256_hex(b"yaro") == sha256_hex(b"yaro")
    assert sha256_hex(b"yaro") != sha256_hex(b"yaro ")


def test_estimate_tokens_monotonic() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("a" * 300) > estimate_tokens("a" * 30)


def test_evidence_rendering_stays_compact() -> None:
    rows = [{"view": index, "rms_px": 0.1} for index in range(100)]
    text = Evidence(kind="table", payload=rows, caption="每视角误差").to_text()
    assert "100 rows total" in text
    assert len(text.splitlines()) < 25


def test_tool_result_text_includes_metrics_and_hint() -> None:
    result = ToolResult(
        ok=True,
        summary="标定完成",
        metrics={"rms_px": 0.1234},
        hint="下一步做手眼标定",
    )
    text = result.to_text()
    assert "OK: 标定完成" in text
    assert "rms_px=0.1234" in text
    assert "next: 下一步做手眼标定" in text


def test_tool_result_truncates_to_budget() -> None:
    result = ToolResult(ok=True, summary="x" * 10_000)
    assert len(result.to_text(max_chars=500)) == 500


def test_usage_cost_matches_price_table() -> None:
    usage = Usage(hit=76_000, miss=4_000, out=2_000, calls=1)
    # flash 谷时：(76000*0.003 + 4000*0.15 + 2000*0.6) / 1e6
    assert usage.cost_usd("deepseek-flash") == pytest.approx(0.002028, rel=1e-6)
    # 峰时翻倍
    assert usage.cost_usd("deepseek-flash", peak=True) == pytest.approx(0.004056, rel=1e-6)
    assert usage.input_total == 80_000
    assert usage.hit_rate == pytest.approx(0.95)


def test_usage_cost_unknown_model_raises() -> None:
    """未知型号直接报错（用户要求 + code review P0-2）。

    历史：最早 ``return 0.0``（改一下 ``YARO_MODEL`` 就能让成本恒 0、预算永不触发），
    中间版本"按最贵档计价"（账目看着正常但没人知道差多少）。现在一律显式报错：
    要么把型号加进价格表，要么别用它。
    """
    usage = Usage(hit=1, miss=1, out=1)
    with pytest.raises(ContractError) as excinfo:
        usage.cost_usd("not-a-model")
    assert "not-a-model" in str(excinfo.value)
    # 认识的型号照常计价，不受影响。
    assert usage.cost_usd("deepseek-flash") == pytest.approx(7.53e-07, rel=1e-3)


def test_ensure_known_model() -> None:
    assert ensure_known_model("deepseek-flash") == "deepseek-flash"
    with pytest.raises(ContractError):
        ensure_known_model("gpt-4o")


def test_usage_add_accumulates() -> None:
    total = Usage().add(Usage(hit=10, miss=5, out=2, calls=1)).add(Usage(hit=1, miss=1, out=1, calls=1))
    assert (total.hit, total.miss, total.out, total.calls) == (11, 6, 3, 2)
