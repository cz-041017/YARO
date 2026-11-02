"""数据集扫描 / 审计 / 切分 / 去重 / 清单。

这一层是"视觉开发助手"的地基：模型好不好，八成先输在数据上。所以这里的每个
函数都只做确定性计算，输出**可被重放的证据**（表 + 计数 + 文件句柄），
不猜、不调模型、不写原地覆盖（去重只移动不删除，且可 --apply 控制）。
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from ..capabilities.builtin import probe_image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
CLASS_FILENAMES = ("classes.txt", "names.txt", "obj.names")
SPLIT_NAMES = ("train", "val", "test")
_SPLIT_ALIASES = {"train": "train", "val": "val", "valid": "val", "validation": "val", "test": "test"}


# --------------------------------------------------------------------------- #
# 基础读取
# --------------------------------------------------------------------------- #


def read_classes(dataset: Path) -> list[str]:
    for name in CLASS_FILENAMES:
        path = dataset / name
        if path.is_file():
            names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if names:
                return names
    return []


def parse_yolo_label(text: str) -> tuple[list[dict[str, float]], list[str]]:
    """解析 YOLO txt；返回 (boxes, problems)。一行不合法不影响其它行。"""
    boxes: list[dict[str, float]] = []
    problems: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.replace(",", " ").split()
        if len(parts) != 5:
            problems.append(f"第 {line_number} 行字段数 {len(parts)}≠5")
            continue
        try:
            cls_f, cx, cy, w, h = (float(value) for value in parts)
        except ValueError:
            problems.append(f"第 {line_number} 行含非数字")
            continue
        boxes.append({"cls": int(cls_f), "cx": cx, "cy": cy, "w": w, "h": h})
    return boxes, problems


def format_yolo_label(boxes: Iterable[dict[str, float]]) -> str:
    lines = [
        f"{int(box['cls'])} {float(box['cx']):.6f} {float(box['cy']):.6f} {float(box['w']):.6f} {float(box['h']):.6f}"
        for box in boxes
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _read_label_file(path: Path) -> tuple[list[dict[str, float]], list[str]]:
    try:
        return parse_yolo_label(path.read_text(encoding="utf-8"))
    except OSError as exc:  # pragma: no cover - 权限/IO 异常
        return [], [f"读取失败：{exc}"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# 布局探测与扫描
# --------------------------------------------------------------------------- #


def detect_layout(dataset: Path) -> str:
    images = dataset / "images"
    if not images.is_dir():
        return "unknown"
    for child in images.iterdir():
        if child.is_dir() and child.name.lower() in _SPLIT_ALIASES:
            return "yolo-split"
    return "yolo-flat"


def scan_dataset(dataset: Path | str, *, layout: str = "auto") -> dict[str, Any]:
    """扫描 YOLO 数据集（flat 或 split），返回逐图条目与统计。路径相对 dataset。"""
    dataset = Path(dataset)
    if layout == "auto":
        layout = detect_layout(dataset)
    classes = read_classes(dataset)
    items: list[dict[str, Any]] = []
    problems: list[str] = []
    if layout == "unknown":
        coco = _find_coco_json(dataset)
        if coco is not None:
            return scan_coco(dataset, coco)
        return {"dataset": str(dataset), "format": "unknown", "layout": "unknown", "classes": classes, "items": [], "problems": ["找不到 images/ 目录，也不是 COCO 标注"], "stats": _empty_stats()}

    images_root = dataset / "images"
    labels_root = dataset / "labels"
    splits: list[str] = [""] if layout == "yolo-flat" else sorted(
        {_SPLIT_ALIASES[child.name.lower()] for child in images_root.iterdir() if child.is_dir() and child.name.lower() in _SPLIT_ALIASES}
    )
    for split in splits:
        image_dir = images_root / split if split else images_root
        label_dir = labels_root / split if split else labels_root
        if not image_dir.is_dir():
            continue
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            boxes: list[dict[str, float]] = []
            box_problems: list[str] = []
            if label_path.is_file():
                boxes, box_problems = _read_label_file(label_path)
                problems.extend(f"{label_path.relative_to(dataset)}: {item}" for item in box_problems)
            else:
                problems.append(f"缺标注：{label_path.relative_to(dataset)}")
            try:
                info = probe_image(image_path)
            except OSError as exc:  # 读不出来也属于脏数据，不能让扫描整体崩掉
                problems.append(f"读不出图像：{image_path.relative_to(dataset)}（{exc}）")
                info = {"format": "UNKNOWN", "width": 0, "height": 0, "bytes": 0, "orientation": None, "truncated": True}
            items.append(
                {
                    "image": str(image_path.relative_to(dataset)),
                    "label": str(label_path.relative_to(dataset)) if label_path.is_file() else None,
                    "split": split,
                    "width": info["width"],
                    "height": info["height"],
                    "bytes": info["bytes"],
                    "format": info["format"],
                    "orientation": info["orientation"],
                    "truncated": bool(info.get("truncated")),
                    "boxes": boxes,
                    "label_problems": box_problems,
                }
            )
    stats = _compute_stats(items, classes)
    return {
        "dataset": str(dataset),
        "format": "yolo",
        "layout": layout,
        "classes": classes,
        "items": items,
        "problems": problems,
        "stats": stats,
    }


def _empty_stats() -> dict[str, Any]:
    return {"images": 0, "labeled": 0, "unlabeled": 0, "boxes": 0, "per_class": {}, "empty_labels": 0, "splits": {}}


def _compute_stats(items: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    per_class: Counter[int] = Counter()
    empty_labels = 0
    splits: Counter[str] = Counter()
    for item in items:
        splits[item.get("split") or "all"] += 1
        if item.get("label") is None:
            continue
        if not item.get("boxes"):
            empty_labels += 1
        for box in item.get("boxes", []):
            per_class[int(box["cls"])] += 1
    named = {classes[index] if index < len(classes) else f"class_{index}": count for index, count in sorted(per_class.items())}
    return {
        "images": len(items),
        "labeled": sum(1 for item in items if item.get("label")),
        "unlabeled": sum(1 for item in items if not item.get("label")),
        "boxes": sum(len(item.get("boxes", [])) for item in items),
        "per_class": named,
        "empty_labels": empty_labels,
        "splits": dict(splits),
    }


def _find_coco_json(dataset: Path) -> Path | None:
    for candidate in (dataset / "annotations" / "instances.json", dataset / "instances.json", dataset / "annotations.json", dataset / "coco.json"):
        if candidate.is_file():
            return candidate
    return None


def scan_coco(dataset: Path, json_path: Path) -> dict[str, Any]:
    """COCO 只做"读 + 审计 + 转 YOLO"所需的最小解析。"""
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    categories = {int(category["id"]): str(category.get("name", category["id"])) for category in payload.get("categories", [])}
    classes = [categories[key] for key in sorted(categories)]
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        by_image[int(annotation["image_id"])].append(annotation)
    items: list[dict[str, Any]] = []
    for image in payload.get("images", []):
        image_id = int(image["id"])
        file_name = str(image.get("file_name", ""))
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        boxes: list[dict[str, float]] = []
        for annotation in by_image.get(image_id, []):
            bbox = annotation.get("bbox") or [0, 0, 0, 0]
            if width <= 0 or height <= 0:
                continue
            boxes.append(
                {
                    "cls": sorted(categories).index(int(annotation.get("category_id", 0))) if int(annotation.get("category_id", 0)) in categories else 0,
                    "cx": (float(bbox[0]) + float(bbox[2]) / 2) / width,
                    "cy": (float(bbox[1]) + float(bbox[3]) / 2) / height,
                    "w": float(bbox[2]) / width,
                    "h": float(bbox[3]) / height,
                }
            )
        items.append(
            {
                "image": file_name,
                "label": None,
                "split": "",
                "width": width,
                "height": height,
                "bytes": 0,
                "format": "COCO",
                "orientation": None,
                "boxes": boxes,
                "label_problems": [],
            }
        )
    return {
        "dataset": str(dataset),
        "format": "coco",
        "layout": "coco",
        "classes": classes,
        "items": items,
        "problems": [],
        "stats": _compute_stats(items, classes),
        "coco_json": str(json_path.relative_to(dataset)),
        "image_root": str(json_path.parent if (json_path.parent / "images").is_dir() else dataset),
    }


# --------------------------------------------------------------------------- #
# 审计
# --------------------------------------------------------------------------- #


def audit_dataset(dataset: Path | str, *, max_findings: int = 200) -> dict[str, Any]:
    """把数据集的"脏"摊开：损坏、缺标、越界、重复、失衡、方向。"""
    dataset = Path(dataset)
    scan = scan_dataset(dataset)
    findings: list[dict[str, Any]] = []
    classes = scan["classes"]

    def add(severity: str, code: str, detail: str, target: str = "") -> None:
        if len(findings) < max_findings:
            findings.append({"severity": severity, "code": code, "detail": detail, "target": target})

    seen_by_hash: dict[str, list[str]] = defaultdict(list)
    label_files: set[str] = set()
    for item in scan["items"]:
        image_rel = item["image"]
        path = dataset / image_rel
        label_rel = item.get("label")
        if label_rel:
            label_files.add(str(Path(label_rel).relative_to("labels")) if label_rel.startswith("labels/") else label_rel)
        # 1) 损坏 / 被截断 / 非图像
        if item["width"] <= 0 or item["height"] <= 0 or item.get("truncated"):
            add("error", "corrupt_image", "图像头无法解析或被截断（拷贝中断/格式伪装）", image_rel)
            continue
        # 2) 缺标注
        if item.get("label") is None:
            add("warn", "missing_label", "有图无标注：会被当成负样本，确认是否有意为之", image_rel)
        else:
            if not item.get("boxes"):
                add("info", "empty_label", "空标注：合法背景图，但过多说明采样有问题", label_rel or image_rel)
            for problem in item.get("label_problems", []):
                add("error", "bad_label_line", problem, label_rel or image_rel)
        # 3) EXIF 方向
        if item.get("orientation") not in (None, 1):
            add("warn", "exif_orientation", f"EXIF Orientation={item['orientation']}：坐标会整体错位", image_rel)
        # 4) 框合法性
        width, height = item["width"], item["height"]
        for box in item.get("boxes", []):
            cx, cy, w, h = box["cx"], box["cy"], box["w"], box["h"]
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                add("error", "box_out_of_range", f"宽高越界 w={w:.4f} h={h:.4f}", label_rel or image_rel)
                continue
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                add("error", "box_center_out_of_range", f"中心越界 cx={cx:.4f} cy={cy:.4f}", label_rel or image_rel)
            if cx - w / 2 < -1e-6 or cy - h / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy + h / 2 > 1 + 1e-6:
                add("warn", "box_clipped", f"框超出图像边界（cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}）", label_rel or image_rel)
            # 像素宽 = w * 图像宽、像素高 = h * 图像高；以前写成 w*height、h*width
            # 的交叉项，宽图上的真框会被误判成 tiny_box（code review B）。
            if w * width < 4 or h * height < 4:
                add("warn", "tiny_box", f"框小于 4px（{w * width:.1f}×{h * height:.1f}）", label_rel or image_rel)
            if w * h < 1e-4:
                add("warn", "degenerate_box", f"框面积过小（{w * h:.2e}）", label_rel or image_rel)
        # 5) 重复（字节级 sha256）
        if width > 0:
            seen_by_hash[_sha256_file(path)].append(image_rel)

    for digest, names in seen_by_hash.items():
        if len(names) > 1:
            add("warn", "duplicate_image", f"内容完全相同 {len(names)} 份（{digest[:12]}）", ", ".join(names[:5]))

    # 6) 孤儿标注
    images_root = dataset / "images"
    if images_root.is_dir():
        image_stems = {Path(item["image"]).stem for item in scan["items"]}
        label_dirs = [dataset / "labels"] + [path for path in (dataset / "labels").glob("*") if path.is_dir()]
        for label_dir in label_dirs:
            if not label_dir.is_dir():
                continue
            for label_path in sorted(label_dir.glob("*.txt")):
                if label_path.stem not in image_stems:
                    add("warn", "orphan_label", "有标注无图：训练时会静默丢弃", str(label_path.relative_to(dataset)))

    # 7) 类别失衡 / 某类在某个 split 缺失
    per_class = scan["stats"]["per_class"]
    counts = [count for count in per_class.values() if count > 0]
    if len(counts) >= 2 and max(counts) / max(1, min(counts)) > 5:
        add("warn", "class_imbalance", f"类别样本量差 {max(counts) / max(1, min(counts)):.1f} 倍：稀有类难学到", "stats.per_class")
    if scan["classes"] and len(per_class) < len(scan["classes"]):
        missing = [name for name in scan["classes"] if name not in per_class]
        add("warn", "class_without_boxes", "这些类一条标注都没有：" + ", ".join(missing), "classes.txt")

    stats = scan["stats"]
    if stats["images"] and stats["unlabeled"] / stats["images"] > 0.1:
        add("warn", "high_unlabeled_ratio", f"{stats['unlabeled']}/{stats['images']} 张图没有标注", "stats")

    summary = {
        "errors": sum(1 for item in findings if item["severity"] == "error"),
        "warnings": sum(1 for item in findings if item["severity"] == "warn"),
        "infos": sum(1 for item in findings if item["severity"] == "info"),
    }
    return {"dataset": str(dataset), "scan": scan, "findings": findings, "summary": summary}


# --------------------------------------------------------------------------- #
# 切分
# --------------------------------------------------------------------------- #


def plan_split(
    dataset: Path | str,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 7,
    stratify: bool = True,
) -> dict[str, Any]:
    """给出确定性的 split 方案（不移动文件）；稀有类按比例强制进 val/test。"""
    dataset = Path(dataset)
    scan = scan_dataset(dataset)
    items = [item["image"] for item in scan["items"]]
    if not items:
        return {"assignments": {}, "counts": {}, "warnings": ["数据集为空"], "seed": seed, "ratios": list(ratios)}
    ratios = tuple(float(value) for value in ratios)
    if len(ratios) != 3 or any(value < 0 for value in ratios):
        raise ValueError("ratios 必须是三个非负数")
    total = sum(ratios) or 1.0
    ratios = tuple(value / total for value in ratios)
    rng = random.Random(seed)
    assignments: dict[str, str] = {}

    def quota(count: int) -> dict[str, int]:
        train = int(round(count * ratios[0]))
        val = int(round(count * ratios[1]))
        test = count - train - val
        if count >= 3 and val == 0:
            val, train = 1, train - 1
        if count >= 3 and test == 0:
            test, train = 1, train - 1
        return {"train": max(0, train), "val": max(0, val), "test": max(0, test)}

    if stratify:
        by_class: dict[int, list[str]] = defaultdict(list)
        for item, entry in zip(items, scan["items"]):
            classes = sorted({int(box["cls"]) for box in entry.get("boxes", [])})
            by_class[classes[0] if classes else -1].append(item)
        for key in sorted(by_class):
            bucket = by_class[key]
            rng.shuffle(bucket)
            limits = quota(len(bucket))
            cursor = 0
            for split in SPLIT_NAMES:
                for name in bucket[cursor : cursor + limits[split]]:
                    assignments[name] = split
                cursor += limits[split]
            # 余数给 train
            for name in bucket[cursor:]:
                assignments[name] = "train"
    else:
        shuffled = list(items)
        rng.shuffle(shuffled)
        limits = quota(len(shuffled))
        cursor = 0
        for split in SPLIT_NAMES:
            for name in shuffled[cursor : cursor + limits[split]]:
                assignments[name] = split
            cursor += limits[split]
        for name in shuffled[cursor:]:
            assignments[name] = "train"

    counts = Counter(assignments.values())
    class_per_split: dict[str, dict[str, int]] = {split: Counter() for split in SPLIT_NAMES}
    for entry in scan["items"]:
        split = assignments.get(entry["image"])
        if split is None:
            continue
        for box in entry.get("boxes", []):
            name = class_name(int(box["cls"]), scan["classes"])
            class_per_split[split][name] += 1
    warnings: list[str] = []
    for split in SPLIT_NAMES:
        if counts.get(split, 0) == 0:
            warnings.append(f"{split} 为空：比例 {ratios} 在 {len(items)} 张图上无法满足")
    if stratify:
        for index, name in enumerate(scan["classes"]):
            if class_per_split["train"].get(name, 0) == 0 and any(class_per_split[split].get(name, 0) for split in ("val", "test")):
                warnings.append(f"类别 {name} 只出现在 val/test：训练集没有它")
    return {
        "dataset": str(dataset),
        "seed": seed,
        "ratios": list(ratios),
        "stratified": stratify,
        "assignments": assignments,
        "counts": {split: counts.get(split, 0) for split in SPLIT_NAMES},
        "class_per_split": {split: dict(counter) for split, counter in class_per_split.items()},
        "warnings": warnings,
    }


def class_name(index: int, classes: list[str]) -> str:
    return classes[index] if 0 <= index < len(classes) else f"class_{index}"


def write_split(dataset: Path | str, plan: dict[str, Any]) -> dict[str, str]:
    """把切分方案写成文件列表 + YOLO data.yaml（非破坏性：不动图片）。"""
    dataset = Path(dataset)
    splits_dir = dataset / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for split in SPLIT_NAMES:
        names = sorted(name for name, assigned in plan["assignments"].items() if assigned == split)
        path = splits_dir / f"{split}.txt"
        path.write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")
        written[split] = str(path.relative_to(dataset))
    scan = scan_dataset(dataset)
    names_map = {index: name for index, name in enumerate(scan["classes"])}
    lines = [
        "# YARO 生成：非破坏性切分（图片原地不动，用 txt 列表引用）",
        "path: .",
        f"train: {written['train']}",
        f"val: {written['val']}",
        f"test: {written['test']}",
        "names:",
    ]
    for index in sorted(names_map):
        lines.append(f"  {index}: {names_map[index]}")
    yaml_path = dataset / "yaro-data.yaml"
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written["yaml"] = str(yaml_path.relative_to(dataset))
    return written


# --------------------------------------------------------------------------- #
# 去重 / 清单
# --------------------------------------------------------------------------- #


def dedupe_duplicates(
    dataset: Path | str,
    *,
    apply: bool = False,
    dest: str = "_quarantine",
) -> dict[str, Any]:
    """字节级去重：默认只报告；``apply=True`` 时把重复图（连同标注）移入隔离区。"""
    dataset = Path(dataset)
    scan = scan_dataset(dataset)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in scan["items"]:
        path = dataset / item["image"]
        try:
            groups[_sha256_file(path)].append(item)
        except OSError:  # pragma: no cover
            continue
    duplicates = [(digest, items) for digest, items in groups.items() if len(items) > 1]
    moved: list[str] = []
    if apply and duplicates:
        for digest, items in duplicates:
            for item in items[1:]:
                for rel in (item["image"], item.get("label")):
                    if not rel:
                        continue
                    source = dataset / rel
                    if not source.is_file():
                        continue
                    target = dataset / dest / f"{digest[:12]}_{Path(rel).name}"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(source), str(target))
                    moved.append(str(target.relative_to(dataset)))
    return {
        "dataset": str(dataset),
        "applied": apply,
        "duplicate_groups": [
            {"sha256": digest, "images": [item["image"] for item in items]} for digest, items in duplicates
        ],
        "duplicate_images": sum(len(items) - 1 for _, items in duplicates),
        "moved": moved,
        "quarantine": dest if apply else None,
    }


def write_manifest(dataset: Path | str, out_name: str = "manifest.jsonl") -> dict[str, Any]:
    """逐图清单（JSONL）：给下游训练/审计/追溯用的稳定事实源。"""
    dataset = Path(dataset)
    scan = scan_dataset(dataset)
    path = dataset / out_name
    with path.open("w", encoding="utf-8") as handle:
        for item in scan["items"]:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return {"path": str(path.relative_to(dataset)), "rows": len(scan["items"])}
