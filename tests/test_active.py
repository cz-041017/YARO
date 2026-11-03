"""主动学习 select_hard_examples 的真实链路单测（不 import torch）。

覆盖 `_find_unlabeled` 的 split/flat 布局 + `select_hard_examples` 的诚实降级
（无未标注图 / 无学生权重）。真正跑学生推理的路径在受控子进程里（需 torch），
这里不测——那些属于集成测试。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from yaro.contracts import DegradedBackend
from yaro.vision.active import _find_unlabeled, select_hard_examples


def _write(p: Path, content: str = "") -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _norm(p: str) -> str:
    """把 Windows 反斜杠统一成 /，断言不受平台分隔符影响。"""
    return str(Path(p)).replace("\\", "/")


def test_find_unlabeled_split_aware(tmp_path: Path) -> None:
    """M1 修复（ce7492b4）：yolo-split 数据集按 split 找标注，返回完整相对路径（不折叠 stem）。"""
    _write(tmp_path / "images/train/a.jpg")
    _write(tmp_path / "images/train/b.jpg")
    _write(tmp_path / "images/val/c.jpg")
    _write(tmp_path / "labels/train/a.txt", "0 0.5 0.5 0.1 0.1")  # 有标注
    _write(tmp_path / "labels/train/b.txt", "")                    # 空标注 → 仍算未标注
    # labels/val/c.txt 缺失 → 未标注

    unlabeled = {_norm(x) for x in _find_unlabeled(tmp_path)}

    assert unlabeled == {"images/train/b.jpg", "images/val/c.jpg"}


def test_find_unlabeled_flat(tmp_path: Path) -> None:
    """flat 数据集：labels/<stem>.txt 缺失/空 → 未标注。"""
    _write(tmp_path / "images/a.jpg")
    _write(tmp_path / "images/b.jpg")
    _write(tmp_path / "labels/a.txt", "0 0.5 0.5 0.1 0.1")

    unlabeled = {_norm(x) for x in _find_unlabeled(tmp_path)}

    assert unlabeled == {"images/b.jpg"}


def test_find_unlabeled_coco_raises(tmp_path: Path) -> None:
    """L-1 修复（9e59f045）：COCO 数据集 label 全为 None，主动学习应拒绝而非全判未标注。"""
    _write(
        tmp_path / "coco.json",
        json.dumps(
            {
                "categories": [{"id": 0, "name": "cat"}],
                "images": [{"id": 0, "file_name": "a.jpg", "width": 100, "height": 100}],
                "annotations": [],
            }
        ),
    )

    with pytest.raises(DegradedBackend):
        _find_unlabeled(tmp_path)


def test_select_hard_no_unlabeled(tmp_path: Path) -> None:
    """没有未标注图 → count=0 且不带 ok 键（与失败 ok=False 天然区分）。"""
    _write(tmp_path / "images/a.jpg")
    _write(tmp_path / "labels/a.txt", "0 0.5 0.5 0.1 0.1")

    out = select_hard_examples(tmp_path)

    assert out["count"] == 0
    assert out["hard"] == []
    assert "ok" not in out


def test_select_hard_no_weights(tmp_path: Path) -> None:
    """有未标注图但无 student 权重 → ok=False 诚实降级（不假装挑出来）。"""
    _write(tmp_path / "images/a.jpg")
    _write(tmp_path / "images/b.jpg")
    _write(tmp_path / "labels/a.txt", "0 0.5 0.5 0.1 0.1")

    out = select_hard_examples(tmp_path)

    assert out["ok"] is False
    assert out["count"] == 0
    assert "学生权重不存在" in out["error"]
