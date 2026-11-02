"""评估器与闸门的单元测试。"""

from __future__ import annotations

import pytest

from yaro.evals import (
    MetricSpec,
    Observation,
    compare,
    cost_per_point,
    fixed_budget_compare,
    pareto_better,
    regression_gate,
)
from yaro.gates import expand_paths, scan_files, scan_source


def _spec(direction: str = "lower", noise: float = 0.01) -> MetricSpec:
    return MetricSpec("rms_px", direction=direction, tolerance=0.0, noise_floor=noise, unit="px")


def test_difference_within_noise_is_same() -> None:
    verdict = compare(_spec(), Observation(0.30, n=5), Observation(0.305, n=5))
    assert verdict.status == "same"
    assert verdict.significant is False


def test_lower_is_better_improvement() -> None:
    verdict = compare(_spec(), Observation(0.30, n=5), Observation(0.20, n=5))
    assert verdict.status == "improved"


def test_lower_is_better_regression() -> None:
    verdict = compare(_spec(), Observation(0.20, n=5), Observation(0.30, n=5))
    assert verdict.status == "worse"


def test_higher_is_better_direction() -> None:
    spec = MetricSpec("map", direction="higher", noise_floor=0.002)
    assert compare(spec, Observation(0.40, n=5), Observation(0.42, n=5)).status == "improved"
    assert compare(spec, Observation(0.42, n=5), Observation(0.40, n=5)).status == "worse"


def test_insufficient_samples_blocked() -> None:
    verdict = compare(_spec(), Observation(0.30, n=1), Observation(0.10, n=1))
    assert verdict.status == "blocked"
    assert verdict.ok is False


def test_hard_limit_blocks_even_with_bad_baseline() -> None:
    spec = MetricSpec("miss_rate", direction="lower", noise_floor=0.001, hard_limit=0.001)
    decision = regression_gate(
        [spec],
        {"miss_rate": Observation(0.01, n=5)},
        {"miss_rate": Observation(0.002, n=5)},
    )
    assert decision.ok is False
    assert any("硬上限" in reason for reason in decision.reasons)


def test_regression_gate_blocks_worse_metric() -> None:
    specs = [MetricSpec("rms_px", "lower", noise_floor=0.01), MetricSpec("map", "higher", noise_floor=0.002)]
    decision = regression_gate(
        specs,
        {"rms_px": Observation(0.20, n=5), "map": Observation(0.40, n=5)},
        {"rms_px": Observation(0.35, n=5), "map": Observation(0.41, n=5)},
    )
    assert decision.ok is False
    assert "rms_px" in " ".join(decision.reasons)


def test_regression_gate_allows_explicit_exception() -> None:
    specs = [MetricSpec("rms_px", "lower", noise_floor=0.01)]
    decision = regression_gate(
        specs,
        {"rms_px": Observation(0.20, n=5)},
        {"rms_px": Observation(0.35, n=5)},
        allow_worse={"rms_px"},
    )
    assert decision.ok is True


def test_regression_gate_missing_metric_blocks() -> None:
    decision = regression_gate([MetricSpec("rms_px", "lower")], {"rms_px": Observation(0.2, n=5)}, {})
    assert decision.ok is False


def test_pareto_better() -> None:
    assert pareto_better({"cost": 1.0, "err": 0.1}, {"cost": 2.0, "err": 0.2}) is True
    assert pareto_better({"cost": 1.0, "err": 0.3}, {"cost": 2.0, "err": 0.2}) is False


def test_cost_per_point() -> None:
    assert cost_per_point(Observation(2.0, cost_usd=1.0)) == pytest.approx(0.5)
    assert cost_per_point(Observation(0.0, cost_usd=1.0)) == float("inf")


def test_fixed_budget_compare_picks_best_within_budget() -> None:
    runs = [(1.0, Observation(0.5)), (5.0, Observation(0.9)), (20.0, Observation(0.99))]
    best = fixed_budget_compare(runs, budget_usd=6.0)
    assert best is not None and best.value == 0.9


def test_gate_blocks_agpl_import() -> None:
    result = scan_source("import ultralytics\n", filename="train.py")
    assert result.ok is False
    assert any(finding.rule == "license" for finding in result.errors)


def test_gate_blocks_eval_and_pickle() -> None:
    result = scan_source("eval('1+1')\nimport pickle\npickle.loads(b'')\n")
    assert result.ok is False
    assert len(result.errors) >= 2


def test_gate_blocks_hardcoded_absolute_path() -> None:
    result = scan_source('path = "/home/robot/data/img.png"\n')
    assert result.ok is False
    assert any(finding.rule == "hardcoded_path" for finding in result.errors)


def test_gate_warns_unseeded_random() -> None:
    result = scan_source("import random\nvalue = random.random()\n")
    assert result.ok is True  # warn 不拦截
    assert any(finding.rule == "reproducibility" for finding in result.findings)


def test_gate_clean_code_passes() -> None:
    result = scan_source("import math\nvalue = math.sqrt(2.0)\n", filename="ok.py")
    assert result.ok is True


def test_gate_scans_directory_instead_of_skipping(tmp_path) -> None:
    """C-7：以前目录被当"文件不存在"跳过且 ok=True，`/api/gates?path=.`
    恒通过，闸门等于没有。"""
    (tmp_path / "bad.py").write_text("import ultralytics\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("import ultralytics\n", encoding="utf-8")
    files = expand_paths([tmp_path])
    assert [path.name for path in files] == ["bad.py"]
    result = scan_files([tmp_path])
    assert result.ok is False
    assert any(finding.rule == "license" for finding in result.errors)
