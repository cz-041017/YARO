"""确定性合成数据集：让"全链路"在没有真实数据时也能跑通、能回归。

生成的东西必须足够像真数据，才值得被审计：

- YOLO 布局（``images/`` + ``labels/`` + ``classes.txt``）；
- 每张图 1–3 个目标，框由几何体真实位置算出（不是随机数字）；
- **故意埋雷**：完全重复图、缺标注、越界框、截断损坏的 PNG、孤儿标注、空标注。
  审计工具应该能把它们逐个找出来——这也是 ``verify.py`` 的断言来源。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .png import encode_png

CLASSES = ["part", "defect", "scratch"]
_BG = (26, 28, 34)


def _canvas(width: int, height: int) -> bytearray:
    buffer = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            index = (y * width + x) * 3
            buffer[index] = _BG[0] + (x * 12) // max(1, width)
            buffer[index + 1] = _BG[1] + (y * 10) // max(1, height)
            buffer[index + 2] = _BG[2]
    return buffer


def _fill_rect(buffer: bytearray, width: int, height: int, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    for y in range(max(0, y0), min(height, y1)):
        for x in range(max(0, x0), min(width, x1)):
            index = (y * width + x) * 3
            buffer[index] = color[0]
            buffer[index + 1] = color[1]
            buffer[index + 2] = color[2]


def _fill_circle(buffer: bytearray, width: int, height: int, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    radius_sq = radius * radius
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        dy = y - cy
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if (x - cx) ** 2 + dy * dy <= radius_sq:
                index = (y * width + x) * 3
                buffer[index] = color[0]
                buffer[index + 1] = color[1]
                buffer[index + 2] = color[2]


def _rows(buffer: bytearray, width: int, height: int) -> list[bytes]:
    return [bytes(buffer[y * width * 3 : (y + 1) * width * 3]) for y in range(height)]


def make_scene(seed: int, *, width: int = 160, height: int = 120) -> tuple[bytes, list[dict[str, float]]]:
    """生成一张图 + 它的 YOLO 框（cls/cx/cy/w/h，归一化）。"""
    rng = random.Random(seed)
    buffer = _canvas(width, height)
    boxes: list[dict[str, float]] = []
    count = rng.randint(1, 3)
    for index in range(count):
        box_w = rng.randint(22, 54)
        box_h = rng.randint(18, 40)
        x0 = rng.randint(4, max(5, width - box_w - 4))
        y0 = rng.randint(4, max(5, height - box_h - 4))
        cls = 0 if index == 0 else rng.choice([0, 1, 2])
        color = (198, 202, 214) if cls == 0 else (232, 196, 96) if cls == 1 else (120, 196, 232)
        if cls == 2:
            _fill_circle(buffer, width, height, x0 + box_w // 2, y0 + box_h // 2, min(box_w, box_h) // 2, color)
        else:
            _fill_rect(buffer, width, height, x0, y0, x0 + box_w, y0 + box_h, color)
        boxes.append(
            {
                "cls": float(cls),
                "cx": round((x0 + box_w / 2) / width, 6),
                "cy": round((y0 + box_h / 2) / height, 6),
                "w": round(box_w / width, 6),
                "h": round(box_h / height, 6),
            }
        )
    return encode_png(width, height, _rows(buffer, width, height)), boxes


def _write_labels(path: Path, boxes: list[dict[str, float]]) -> None:
    lines = [
        f"{int(box['cls'])} {box['cx']:.6f} {box['cy']:.6f} {box['w']:.6f} {box['h']:.6f}"
        for box in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_synthetic_dataset(
    directory: Path | str,
    *,
    images: int = 12,
    seed: int = 7,
    plant_issues: bool = True,
) -> dict[str, Any]:
    """写出一个带已知问题的 YOLO 数据集，返回清单（幂等：已存在则跳过写图）。"""
    root = Path(directory)
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")

    written: list[str] = []
    for index in range(images):
        name = f"frame_{index:04d}"
        image_path = images_dir / f"{name}.png"
        label_path = labels_dir / f"{name}.txt"
        png, boxes = make_scene(seed + index)
        if not image_path.exists():
            image_path.write_bytes(png)
        _write_labels(label_path, boxes)
        written.append(name)

    planted: list[str] = []
    if plant_issues and written:
        first_png = (images_dir / f"{written[0]}.png").read_bytes()
        first_boxes = _read_labels(labels_dir / f"{written[0]}.txt")

        # 1) 完全重复的图（内容寻址去重能抓到）
        (images_dir / "dup_copy.png").write_bytes(first_png)
        _write_labels(labels_dir / "dup_copy.txt", first_boxes)
        planted.append("duplicate:dup_copy.png")

        # 2) 缺标注（图有、txt 没有）
        (images_dir / "no_label.png").write_bytes(
            make_scene(seed + 100, width=140, height=100)[0]
        )
        planted.append("missing_label:no_label.png")

        # 3) 越界框：一行 cx>1（中心越界），一行 w>1（宽高越界），两种坏法都演示
        bad = labels_dir / "out_of_range.txt"
        bad.write_text(
            "0 1.050000 0.500000 0.200000 0.200000\n"
            "0 0.500000 0.500000 1.400000 0.200000\n",
            encoding="utf-8",
        )
        (images_dir / "out_of_range.png").write_bytes(make_scene(seed + 200, width=150, height=110)[0])
        planted.append("out_of_range:out_of_range.txt")

        # 4) 截断的 PNG（模拟拷贝中断的坏图）
        corrupt = (images_dir / "corrupt.png")
        corrupt.write_bytes(first_png[:48])
        _write_labels(labels_dir / "corrupt.txt", first_boxes[:1])
        planted.append("corrupt:corrupt.png")

        # 5) 孤儿标注（txt 有、图没有）
        (labels_dir / "orphan_label.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        planted.append("orphan_label:orphan_label.txt")

        # 6) 空标注（纯背景图，是合法样本但值得提示）
        (images_dir / "empty_bg.png").write_bytes(make_scene(seed + 300, width=120, height=90)[0])
        (labels_dir / "empty_bg.txt").write_text("", encoding="utf-8")
        planted.append("empty_label:empty_bg.txt")

    return {
        "root": root,
        "images": len([p for p in images_dir.glob("*.png")]),
        "labels": len([p for p in labels_dir.glob("*.txt")]),
        "classes": CLASSES,
        "planted": planted,
        "written": written,
    }


def _read_labels(path: Path) -> list[dict[str, float]]:
    boxes: list[dict[str, float]] = []
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls, cx, cy, w, h = (float(value) for value in parts)
        except ValueError:
            continue
        boxes.append({"cls": cls, "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes
