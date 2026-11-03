"""视觉链路测试：PNG 编解码、数据集审计、切分、渲染、计划生成、工具包装。

全部离线、零外部依赖（torch / onnxruntime / opencv 都不需要）。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from yaro.capabilities.builtin import build_tool_catalog
from yaro.memory.artifacts import ArtifactStore
from yaro.vision import annotate, dataset, plan, png, synth
from yaro.vision.tools import (
    DatasetAuditTool,
    DatasetRenderTool,
    DatasetScanTool,
    build_vision_tools,
)


def _png(width: int = 8, height: int = 6) -> bytes:
    rows = [bytes([(x * 30) % 256 for x in range(width * 3)]) for _ in range(height)]
    return png.encode_png(width, height, rows)


# --------------------------------------------------------------------------- #
# PNG / 渲染
# --------------------------------------------------------------------------- #


def test_png_roundtrip_and_resize() -> None:
    data = _png(9, 5)
    image = png.decode_png(data)
    assert (image.width, image.height) == (9, 5)
    assert image.channels in (3, 4)
    resized = png.resize_nearest(image, 4, 2)
    assert (resized.width, resized.height) == (4, 2)
    assert png.decode_png(png.to_png_bytes(resized)).width == 4


def test_render_boxes_draws_on_png() -> None:
    data = _png(40, 30)
    boxes = [{"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.4, "h": 0.4}]
    out = png.render_boxes(data, boxes, classes=["part"])
    rendered = png.decode_png(out)
    assert (rendered.width, rendered.height) == (40, 30)
    assert out.startswith(png.PNG_SIGNATURE)


def test_contact_sheet() -> None:
    board = png.compose_contact_sheet([png.decode_png(_png(10, 10)), png.decode_png(_png(10, 10))], cols=2)
    sheet = png.to_png_bytes(board)
    assert sheet.startswith(png.PNG_SIGNATURE)
    assert png.decode_png(sheet).width > 10


# --------------------------------------------------------------------------- #
# 合成数据 + 扫描 + 审计
# --------------------------------------------------------------------------- #


@pytest.fixture()
def dataset_dir(tmp_path: Path) -> Path:
    target = tmp_path / "dataset"
    synth.write_synthetic_dataset(target, images=8, seed=7, plant_issues=True)
    return target


def test_synth_writes_layout(dataset_dir: Path) -> None:
    assert (dataset_dir / "images").is_dir()
    assert (dataset_dir / "labels").is_dir()
    assert (dataset_dir / "classes.txt").is_file()
    assert len(list((dataset_dir / "images").iterdir())) >= 8


def test_scan_dataset(dataset_dir: Path) -> None:
    scan = dataset.scan_dataset(dataset_dir)
    assert scan["format"] == "yolo"
    assert scan["layout"] in {"yolo-split", "yolo-flat"}
    assert scan["stats"]["images"] >= 8
    assert scan["stats"]["boxes"] > 0
    assert scan["classes"] == synth.CLASSES


def test_audit_detects_planted_issues(dataset_dir: Path) -> None:
    report = dataset.audit_dataset(dataset_dir)
    codes = {item["code"] for item in report["findings"]}
    assert {"corrupt_image", "missing_label", "box_out_of_range", "duplicate_image", "orphan_label"} <= codes
    assert report["summary"]["errors"] > 0


def test_audit_clean_dataset(tmp_path: Path) -> None:
    target = tmp_path / "clean"
    synth.write_synthetic_dataset(target, images=6, seed=11, plant_issues=False)
    report = dataset.audit_dataset(target)
    assert report["summary"]["errors"] == 0


def test_dedupe_reports_and_quarantines(dataset_dir: Path) -> None:
    report = dataset.dedupe_duplicates(dataset_dir)
    assert report["duplicate_groups"]
    before = len(list((dataset_dir / "images").iterdir()))
    applied = dataset.dedupe_duplicates(dataset_dir, apply=True)
    assert applied["moved"]
    assert len(list((dataset_dir / "images").iterdir())) < before


def test_split_is_deterministic_and_stratified(dataset_dir: Path) -> None:
    first = dataset.plan_split(dataset_dir, ratios=(0.7, 0.2, 0.1), seed=7)
    second = dataset.plan_split(dataset_dir, ratios=(0.7, 0.2, 0.1), seed=7)
    assert first["assignments"] == second["assignments"]
    # 以前这里断言的是 sum(counts.values()) == train+val+test —— 恒真，等于没测（B-17）。
    # 真正要保证的是：每个图恰好落到一个 split 里。
    assert sum(first["counts"].values()) == len(first["assignments"])
    assert set(first["assignments"].values()) <= {"train", "val", "test"}
    assert first["counts"]["train"] > 0
    written = dataset.write_split(dataset_dir, first)
    assert {"train", "val", "test"} <= set(written)
    assert "yaml" in written
    for path in written.values():
        assert Path(path).is_file() or (dataset_dir / path).is_file()


def test_render_dataset_produces_sheet(dataset_dir: Path) -> None:
    report = annotate.render_dataset(dataset_dir, dataset_dir / "runs" / "review", max_images=4)
    assert report["rendered"]
    assert report["sheet"]
    sheet = dataset_dir / report["sheet"]
    assert sheet.is_file()
    assert sheet.read_bytes().startswith(png.PNG_SIGNATURE)


def test_export_coco_and_voc(dataset_dir: Path) -> None:
    coco = annotate.export_coco(dataset_dir)
    assert coco["annotations"] > 0
    assert (dataset_dir / coco["path"]).is_file()
    voc = annotate.export_voc(dataset_dir)
    assert voc["files"] > 0


# --------------------------------------------------------------------------- #
# 计划：训练 / 调优 / 部署
# --------------------------------------------------------------------------- #


def test_build_train_plan(dataset_dir: Path) -> None:
    result = plan.build_train_plan(dataset_dir, backend="torchvision", epochs=5)
    assert result["recipe"]["epochs"] == 5
    names = {Path(path).name for path in result["written"]}
    assert {"recipe.json", "train.py", "acceptance.md"} <= names
    for path in result["written"]:
        assert (dataset_dir / path).is_file()


def test_build_train_plan_rejects_agpl_backend(dataset_dir: Path) -> None:
    with pytest.raises(ValueError):
        plan.build_train_plan(dataset_dir, backend="mmdetection")


def test_parse_and_analyze_log() -> None:
    text = "\n".join(
        json.dumps({"epoch": index + 1, "loss": 1.0 - index * 0.03, "map50": 0.2 + index * 0.01})
        for index in range(8)
    )
    history = plan.parse_train_log(text)
    assert history["count"] == 8
    findings = plan.analyze_training(history)
    assert isinstance(findings, list)


def test_trial_planning_and_summary() -> None:
    combos = plan.plan_trials({"lr": [0.001, 0.01], "imgsz": [512, 640]}, trials=4, seed=7, mode="grid")
    assert len(combos) == 4
    rows = [{"lr": combo["lr"], "imgsz": combo["imgsz"], "map50": 0.3 + index * 0.01} for index, combo in enumerate(combos)]
    summary = plan.summarize_trials(rows, metric="map50", direction="higher")
    assert summary["best"]["map50"] == max(row["map50"] for row in rows)
    assert summary["ordered"]


def test_build_deploy_plan(dataset_dir: Path) -> None:
    result = plan.build_deploy_plan(dataset_dir, backends=("onnx",), latency_target_ms=40.0)
    names = {Path(path).name for path in result["written"]}
    assert {"export_onnx.py", "bench_onnx.py", "Dockerfile", "model_card.md"} <= names
    assert result["latency_target_ms"] == 40.0


# --------------------------------------------------------------------------- #
# 工具包装
# --------------------------------------------------------------------------- #


def test_scan_tool_returns_table_evidence(dataset_dir: Path, tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    tool = DatasetScanTool(dataset_dir.parent)
    result = asyncio.run(tool.run(dir=dataset_dir.name))
    assert result.ok is True
    assert result.evidence
    assert any(item.kind == "table" for item in result.evidence)
    assert "images" in result.metrics


def test_audit_tool_reports_errors(dataset_dir: Path) -> None:
    tool = DatasetAuditTool(dataset_dir.parent)
    result = asyncio.run(tool.run(dir=dataset_dir.name))
    assert result.ok is False
    assert result.metrics["errors"] > 0
    assert result.hint


def test_render_tool_registers_artifact(dataset_dir: Path, tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    tool = DatasetRenderTool(dataset_dir.parent, store, run_id="test", step=1)
    result = asyncio.run(tool.run(dir=dataset_dir.name, max_images=3))
    assert result.ok is True
    assert result.artifacts
    assert len(store) >= 1
    images = [item for item in result.evidence if item.kind == "image"]
    assert images and images[0].payload.media_type == "image/png"


def test_vision_tools_registered_in_catalog(dataset_dir: Path, tmp_path: Path) -> None:
    names = {item["name"] for item in build_tool_catalog(dataset_dir.parent)}
    assert {
        "dataset_scan",
        "dataset_audit",
        "dataset_split",
        "dataset_render",
        "prelabel_run",
        "train_plan",
        "train_run",
        "train_analyze",
        "tune_plan",
        "tune_run",
        "deploy_plan",
        "deploy_run",
    } <= names
    store = ArtifactStore(tmp_path / "artifacts")
    tools = build_vision_tools(dataset_dir.parent, store, run_id="test")
    assert len(tools) == 14
    # 真执行类工具的 spec 名称必须稳定：模型靠名字调用，改名等于断掉行为契约
    tool_names = {tool.spec.name for tool in tools}
    assert {"prelabel_run", "train_run", "tune_run", "deploy_run"} <= tool_names
