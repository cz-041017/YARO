"""标注渲染与格式互转：把"数据集里到底是什么"变成人能看、模型能读的证据。

三件事：

1. **渲染**：把 YOLO 框画回图上，拼成联系人表 —— 标注错位、框贴边、类别标错，
   一眼就能看出来；
2. **互转**：YOLO ↔ COCO ↔ VOC。真实项目中标注工具、训练框架、甲方交付格式
   几乎从不一致，转换是最常被低估的脏活；
3. **预标注计划**：ONNX 检测器批量出 proposals（不覆盖人工标注），
   没有 onnxruntime 时明确降级，而不是偷偷换个实现。
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .dataset import IMAGE_SUFFIXES, class_name, scan_dataset
from .png import Image, compose_contact_sheet, decode_png, encode_png, render_boxes


def _write_render(image: Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [image.pixels[index * image.stride : (index + 1) * image.stride] for index in range(image.height)]
    path.write_bytes(encode_png(image.width, image.height, rows))


def render_dataset(
    dataset: Path | str,
    out_dir: Path | str,
    *,
    max_images: int = 6,
    max_side: int = 320,
    sheet: bool = True,
    classes: list[str] | None = None,
) -> dict[str, Any]:
    """渲染前 N 张带框的图 + 一张联系人表。只支持 PNG（JPEG 需 cv2/Pillow，明确跳过）。"""
    dataset = Path(dataset)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scan = scan_dataset(dataset)
    classes = classes or scan["classes"]
    rendered: list[str] = []
    thumbnails: list[Image] = []
    skipped: list[str] = []
    for index, item in enumerate(scan["items"]):
        if len(rendered) >= max(1, int(max_images)):
            break
        image_path = dataset / item["image"]
        if image_path.suffix.lower() != ".png":
            skipped.append(item["image"])
            continue
        try:
            data = image_path.read_bytes()
            png = render_boxes(data, item.get("boxes", []), classes=classes)
            decoded = decode_png(png)
        except Exception as exc:  # noqa: BLE001 - 坏图必须跳过，不能中断整批渲染
            skipped.append(f"{item['image']}（{type(exc).__name__}: {exc}）")
            continue
        scale = min(1.0, max_side / max(1, max(decoded.width, decoded.height)))
        target = out / f"{index:03d}_{image_path.name}"
        if scale < 1.0:
            from .png import resize_nearest

            decoded = resize_nearest(decoded, int(decoded.width * scale), int(decoded.height * scale))
        _write_render(decoded, target)
        rendered.append(str(target.relative_to(dataset)) if target.is_relative_to(dataset) else str(target))
        thumbnails.append(decoded)
    sheet_rel = None
    if sheet and thumbnails:
        board = compose_contact_sheet(thumbnails, cols=min(3, len(thumbnails)), cell=240)
        sheet_path = out / "contact_sheet.png"
        _write_render(board, sheet_path)
        sheet_rel = str(sheet_path.relative_to(dataset)) if sheet_path.is_relative_to(dataset) else str(sheet_path)
    return {
        "dataset": str(dataset),
        "rendered": rendered,
        "sheet": sheet_rel,
        "skipped": skipped,
        "scanned": len(scan["items"]),
    }


# --------------------------------------------------------------------------- #
# 格式互转
# --------------------------------------------------------------------------- #


def export_coco(dataset: Path | str, out_name: str = "exports/instances.json") -> dict[str, Any]:
    dataset = Path(dataset)
    scan = scan_dataset(dataset)
    classes = scan["classes"]
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    category_ids = {name: index + 1 for index, name in enumerate(classes)}
    annotation_id = 1
    for index, item in enumerate(scan["items"], start=1):
        width, height = item["width"], item["height"]
        if width <= 0 or height <= 0:
            continue
        images.append({"id": index, "file_name": item["image"], "width": width, "height": height})
        for box in item.get("boxes", []):
            box_w = box["w"] * width
            box_h = box["h"] * height
            x = (box["cx"] * width) - box_w / 2
            y = (box["cy"] * height) - box_h / 2
            name = class_name(int(box["cls"]), classes)
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": index,
                    "category_id": category_ids.get(name, 1),
                    "bbox": [round(x, 2), round(y, 2), round(box_w, 2), round(box_h, 2)],
                    "area": round(box_w * box_h, 2),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    payload = {
        "info": {"description": "YARO 导出", "version": "1.0"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": category_ids[name], "name": name, "supercategory": "object"} for name in classes],
    }
    path = dataset / out_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(path.relative_to(dataset)), "images": len(images), "annotations": len(annotations)}


def export_voc(dataset: Path | str, out_dir: str = "exports/voc") -> dict[str, Any]:
    dataset = Path(dataset)
    scan = scan_dataset(dataset)
    classes = scan["classes"]
    target = dataset / out_dir
    target.mkdir(parents=True, exist_ok=True)
    written = 0
    for item in scan["items"]:
        if item["width"] <= 0 or item["height"] <= 0:
            continue
        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "filename").text = Path(item["image"]).name
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(item["width"])
        ET.SubElement(size, "height").text = str(item["height"])
        ET.SubElement(size, "depth").text = "3"
        for box in item.get("boxes", []):
            name = class_name(int(box["cls"]), classes)
            x_min = (box["cx"] - box["w"] / 2) * item["width"]
            y_min = (box["cy"] - box["h"] / 2) * item["height"]
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = name
            ET.SubElement(obj, "difficult").text = "0"
            bnd = ET.SubElement(obj, "bndbox")
            ET.SubElement(bnd, "xmin").text = str(max(0, round(x_min)))
            ET.SubElement(bnd, "ymin").text = str(max(0, round(y_min)))
            ET.SubElement(bnd, "xmax").text = str(min(item["width"], round(x_min + box["w"] * item["width"])))
            ET.SubElement(bnd, "ymax").text = str(min(item["height"], round(y_min + box["h"] * item["height"])))
        path = target / f"{Path(item['image']).stem}.xml"
        path.write_text(ET.tostring(annotation, encoding="unicode"), encoding="utf-8")
        written += 1
    return {"path": str(target.relative_to(dataset)), "files": written}


def import_coco_to_yolo(dataset: Path | str, json_path: Path | str, *, copy_images: bool = False) -> dict[str, Any]:
    """COCO → YOLO（写出 labels/ 与 classes.txt；图片可选复制）。"""
    dataset = Path(dataset)
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    categories = {int(category["id"]): str(category.get("name", category["id"])) for category in payload.get("categories", [])}
    order = sorted(categories)
    images_root = dataset / "images"
    labels_root = dataset / "labels"
    images_root.mkdir(parents=True, exist_ok=True)
    labels_root.mkdir(parents=True, exist_ok=True)
    (dataset / "classes.txt").write_text("\n".join(categories[key] for key in order) + "\n", encoding="utf-8")
    by_image: dict[int, list[dict[str, Any]]] = {}
    for annotation in payload.get("annotations", []):
        by_image.setdefault(int(annotation["image_id"]), []).append(annotation)
    written = 0
    for image in payload.get("images", []):
        image_id = int(image["id"])
        file_name = Path(str(image.get("file_name", f"{image_id}.png"))).name
        width = int(image.get("width") or 0)
        height = int(image.get("height") or 0)
        if width <= 0 or height <= 0:
            continue
        lines: list[str] = []
        for annotation in by_image.get(image_id, []):
            bbox = annotation.get("bbox") or [0, 0, 0, 0]
            category = int(annotation.get("category_id", 0))
            cls = order.index(category) if category in order else 0
            cx = (float(bbox[0]) + float(bbox[2]) / 2) / width
            cy = (float(bbox[1]) + float(bbox[3]) / 2) / height
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {float(bbox[2]) / width:.6f} {float(bbox[3]) / height:.6f}")
        (labels_root / f"{Path(file_name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        written += 1
    return {"labels": written, "classes": len(order), "images_dir": str(images_root)}


# --------------------------------------------------------------------------- #
# 预标注（ONNX 检测器）
# --------------------------------------------------------------------------- #


def onnx_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:  # noqa: BLE001 - 缺依赖就是缺依赖
        return False


def prelabel_plan(dataset: Path | str, *, model: str = "models/detector.onnx", conf: float = 0.35, iou: float = 0.5) -> dict[str, Any]:
    """生成可执行的预标注脚本计划；不触碰人工标注（输出到 proposals/）。"""
    dataset = Path(dataset)
    script = f'''# 由 YARO 生成：ONNX 检测器预标注
# 输入：{dataset}/images  输出：{dataset}/proposals（YOLO txt，待人工审核）
import json
from pathlib import Path
import numpy as np
from PIL import Image  # 或 cv2

DATASET = Path({str(dataset)!r})
MODEL = {model!r}
CONF = {conf}
IOU = {iou}

# 1) onnxruntime 推理 -> 每图 N×(cx,cy,w,h,score,cls)
# 2) NMS(iou=IOU)  -> 过滤 score<CONF
# 3) 写 proposals/<stem>.txt（与 labels/ 分离，人工确认后再合并）
# 注意：坐标需按 letterbox 的缩放与 padding 反变换回原图。
'''
    return {
        "dataset": str(dataset),
        "script": script,
        "out_dir": "proposals",
        "onnxruntime": onnx_available(),
        "hint": "先跑预标注再人工审核：低置信度样本进 review 队列，不要直接进训练集",
    }


def prelabel_statistics(proposals_dir: Path | str) -> dict[str, Any]:
    """统计预标注产出：有多少框、置信度分布、多少图是空的。"""
    directory = Path(proposals_dir)
    if not directory.is_dir():
        return {"files": 0, "boxes": 0, "empty": 0, "mean_conf": 0.0}
    scores: list[float] = []
    boxes = 0
    empty = 0
    files = 0
    for path in sorted(directory.glob("*.txt")):
        files += 1
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) >= 6:
                try:
                    scores.append(float(parts[1]))
                except ValueError:
                    continue
            boxes += 1
    return {
        "files": files,
        "boxes": boxes,
        "empty": empty,
        "mean_conf": round(sum(scores) / len(scores), 4) if scores else 0.0,
    }


def list_unreviewed(dataset: Path | str, *, proposals: str = "proposals", labels: str = "labels") -> list[str]:
    """列出只有 proposals、还没有人工标注的图（人工审核队列）。"""
    dataset = Path(dataset)
    proposal_dir = dataset / proposals
    label_dir = dataset / labels
    if not proposal_dir.is_dir():
        return []
    queue: list[str] = []
    for path in sorted(proposal_dir.glob("*.txt")):
        if not (label_dir / path.name).is_file():
            queue.append(path.stem)
    return queue


def supported_image_suffixes() -> set[str]:
    return set(IMAGE_SUFFIXES)
