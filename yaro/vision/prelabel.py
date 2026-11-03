"""真预标注（不是"生成计划"）：对图片集真的跑模型/图像算法，产出 proposals。

三种后端，能力从强到弱，按可用性自动选：

1. ``torchvision``：COCO 预训练检测器（FCOS / FasterRCNN）。
   质量最好的通用选择，但只有 COCO 的 80 类能被"按名字"对上；
   你的类目不在其中时，输出**类无关框**（cls=0），由人工改标签——
   这正是 SAM 数据引擎的做法：模型出框，人只做确认/改类。
2. ``onnx``：你自己训过一版或下载的检测 ONNX，精度取决于该模型。
   约定输出为常见形状：``[1, N, 6]``（x1,y1,x2,y2,score,cls）或 ``[1, N, 5+nc]``（YOLO 风格）。
3. ``contour``：OpenCV 经典方法。对高对比度工业件（定位/分割明显、背景干净）
   常比通用检测器更准，且不需要下载权重、不需要 GPU。

产出（不覆盖人工标注）：
- ``<dataset>/proposals/<stem>.txt``：YOLO 格式，可直接给标注工具加载
- ``<dataset>/proposals/report.json``：后端、阈值、每图框数、类别映射、告警
- 人工审核队列：``annotate.list_unreviewed()`` 读 proposals 与 labels 的差集
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

from ..contracts import DegradedBackend
from .dataset import IMAGE_SUFFIXES, class_name, scan_dataset

BACKENDS = ("torchvision", "onnx", "contour")
_IMAGE_SUFFIXES = tuple(IMAGE_SUFFIXES)


def available_backends() -> dict[str, str]:
    """返回 {后端: 说明}，只含真正可用的。"""
    from ..exec.env import capabilities

    caps = capabilities()
    out: dict[str, str] = {}
    if caps["can_prelabel_torch"]:
        out["torchvision"] = "COCO 预训练检测器（自动按类名匹配，未匹配则输出类无关框）"
    if caps["can_prelabel_onnx"]:
        out["onnx"] = "自备 ONNX 检测模型（需传 --model）"
    if caps["can_prelabel_contour"]:
        out["contour"] = "OpenCV 轮廓法（高对比度固定场景，无需权重/GPU）"
    return out


def _require_numpy():
    try:
        import numpy as np  # noqa: PLC0415

        return np
    except ImportError as exc:
        raise DegradedBackend(
            "缺少 numpy：预标注需要它。安装："
            + _install_hint()
        ) from exc


def _install_hint() -> str:
    from ..exec.env import suggest_install

    return suggest_install(train=False, with_cv=True)


def load_rgb(path: Path):
    """读成 HxWx3 uint8。优先 Pillow，其次 OpenCV，最后用内置 PNG 解码。"""
    np = _require_numpy()
    suffix = path.suffix.lower()
    try:
        from PIL import Image  # noqa: PLC0415

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except ImportError:
        pass
    except (OSError, ValueError) as exc:
        raise DegradedBackend(f"图像解码失败：{path.name}（{exc}）") from exc
    try:
        import cv2  # noqa: PLC0415

        data = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if data is None:
            raise DegradedBackend(f"OpenCV 读不出这张图：{path.name}")
        return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
    except ImportError:
        pass
    if suffix == ".png":
        from .png import decode_png

        image = decode_png(path.read_bytes())
        rgb = image.to_rgb()
        array = np.frombuffer(b"".join(rgb.pixels), dtype=np.uint8)
        return array.reshape(rgb.height, rgb.width, 3).copy()
    raise DegradedBackend(
        f"{path.name} 需要 Pillow 或 OpenCV 才能解码（内置解码器只支持 PNG）。安装：{_install_hint()}"
    )


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _to_yolo(boxes: list[tuple[float, float, float, float]], width: int, height: int, cls: int) -> list[str]:
    """像素框 (x1,y1,x2,y2) -> YOLO 归一化行。太小的框直接丢掉。"""
    lines: list[str] = []
    for x1, y1, x2, y2 in boxes:
        bw = (x2 - x1) / max(width, 1)
        bh = (y2 - y1) / max(height, 1)
        if bw <= 0 or bh <= 0 or bw * bh < 1e-5:
            continue
        cx = _clip01(((x1 + x2) / 2) / max(width, 1))
        cy = _clip01(((y1 + y2) / 2) / max(height, 1))
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {min(bw, 1.0):.6f} {min(bh, 1.0):.6f}")
    return lines


# --------------------------------------------------------------------------- 后端


def _backend_torchvision(dataset: Path, classes: list[str], *, conf: float, device: str):
    try:
        import torch  # noqa: PLC0415
        import torchvision  # noqa: PLC0415
        from torchvision.models.detection import (  # noqa: PLC0415
            FCOS_ResNet50_FPN_Weights,
            fcos_resnet50_fpn,
        )
    except ImportError as exc:
        raise DegradedBackend(f"缺少 torch/torchvision。安装：{_install_hint()}") from exc

    weights = FCOS_ResNet50_FPN_Weights.DEFAULT
    model = fcos_resnet50_fpn(weights=weights).eval()
    torch_device = "cpu"
    if device == "auto":
        if torch.cuda.is_available():
            torch_device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch_device = "mps"
    else:
        torch_device = device
    model.to(torch_device)
    categories = [str(name).lower() for name in weights.meta.get("categories", [])]
    # 按名字把 COCO 类别对到用户类目；对不上的，输出 cls=0 的类无关框
    mapping = {}
    for index, name in enumerate(classes):
        lowered = name.strip().lower()
        if lowered in categories:
            mapping[categories.index(lowered) + 1] = index  # COCO label 从 1 开始

    def predict(rgb):
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(torch_device)
        with torch.no_grad():
            result = model([tensor])[0]
        boxes = result["boxes"].detach().cpu().tolist()
        scores = result["scores"].detach().cpu().tolist()
        labels = result["labels"].detach().cpu().tolist()
        kept: list[tuple[tuple[float, float, float, float], int, float]] = []
        for box, score, label in zip(boxes, scores, labels):
            if score < conf:
                continue
            cls = mapping.get(int(label), 0)
            kept.append(((box[0], box[1], box[2], box[3]), cls, float(score)))
        return kept

    return predict, {
        "backend": "torchvision",
        "model": "fcos_resnet50_fpn(coco)",
        "device": torch_device,
        "class_mapping": {classes[i]: categories[c] for c, i in mapping.items()},
        "class_agnostic": len(mapping) == 0,
        "note": "预训练类别与你的类目不同，输出的是类无关框（cls=0），请在审核时改类。",
    }


def _backend_onnx(model_path: Path, *, conf: float):
    try:
        import numpy as np  # noqa: PLC0415
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError as exc:
        raise DegradedBackend(f"缺少 onnxruntime。安装：{_install_hint()}") from exc
    if not model_path.exists():
        raise DegradedBackend(f"ONNX 模型不存在：{model_path}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    height = int(input_meta.shape[2]) if isinstance(input_meta.shape[2], int) else 640
    width = int(input_meta.shape[3]) if isinstance(input_meta.shape[3], int) else 640

    def predict(rgb):
        import cv2  # noqa: PLC0415

        resized = cv2.resize(rgb, (width, height))
        blob = resized.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        outputs = session.run(None, {input_meta.name: blob})
        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]
        return _parse_onnx_rows(raw, rgb.shape[1], rgb.shape[0], conf)

    return predict, {
        "backend": "onnx",
        "model": str(model_path),
        "input": f"{width}x{height}",
        "note": "假定输出为 [N,6]=xyxy+score+cls 或 [N,5+nc]=YOLO 风格；形状不符会跳过该图。",
    }


def _parse_onnx_rows(raw, width: int, height: int, conf: float):
    np = _require_numpy()
    rows = np.asarray(raw, dtype=float)
    out: list[tuple[tuple[float, float, float, float], int, float]] = []
    if rows.ndim != 2 or rows.shape[0] == 0:
        return out
    cols = rows.shape[1]
    for row in rows:
        if cols == 6:
            x1, y1, x2, y2, score, cls = row
        elif cols >= 6:
            # YOLO: cx,cy,w,h,obj,cls_scores...
            cx, cy, bw, bh, obj = row[:5]
            scores = row[5:]
            cls = int(np.argmax(scores))
            score = float(obj) * float(scores[cls])
            x1, y1 = cx - bw / 2, cy - bh / 2
            x2, y2 = cx + bw / 2, cy + bh / 2
        else:
            continue
        if float(score) < conf:
            continue
        # 归一化坐标 vs 像素坐标：都 <=1 时按归一化处理
        if max(x1, y1, x2, y2) <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        out.append(((float(x1), float(y1), float(x2), float(y2)), int(cls), float(score)))
    return out


def _backend_contour(*, min_area_ratio: float = 0.002, max_boxes: int = 60):
    try:
        import cv2  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        raise DegradedBackend(f"缺少 OpenCV。安装：{_install_hint()}") from exc

    def predict(rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 前景可能是黑底白物，也可能是白底黑物：取占少数的那一类当目标
        if float(np.count_nonzero(binary)) > binary.size * 0.5:
            binary = cv2.bitwise_not(binary)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = rgb.shape[:2]
        min_area = min_area_ratio * width * height
        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            boxes.append(((float(x), float(y), float(x + bw), float(y + bh)), 0, min(1.0, 0.5 + area / (width * height))))
        boxes.sort(key=lambda item: item[2], reverse=True)
        return boxes[:max_boxes]

    return predict, {
        "backend": "contour",
        "note": "Otsu 二值化 + 形态学 + 外轮廓。适合背景干净、目标与背景对比明显的固定场景；类别固定为 0。",
    }


# --------------------------------------------------------------------------- 主入口


def prelabel_dataset(
    dataset: Path | str,
    *,
    backend: str = "auto",
    out: str = "proposals",
    conf: float = 0.35,
    limit: int | None = None,
    subset: list[str] | None = None,
    model: Path | str | None = None,
    device: str = "auto",
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """对数据集跑预标注，写出 proposals/*.txt 与 report.json。"""
    root = Path(dataset).resolve()
    if not root.exists():
        raise DegradedBackend(f"数据集不存在：{root}")
    scan = scan_dataset(root)
    if scan["format"] == "unknown":
        raise DegradedBackend(f"认不出数据集结构（需要 images/ + labels/ 或 YOLO 分目录）：{root}")
    classes = [str(name) for name in scan.get("classes", [])]
    images = [item for item in scan["items"] if item.get("width") and item.get("height")]
    if limit:
        images = images[: int(limit)]
    if subset is not None:
        # M1 修复：按完整相对路径匹配（含 split），不折叠成 stem——否则 yolo-split 数据集
        # 里 train/val 同名图会跨 split 命中多张、把 val/test 图当难例回灌。
        def _norm(p: str) -> str:
            return str(Path(p)).replace("\\", "/")

        wanted = {_norm(s) for s in subset}
        images = [item for item in images if _norm(item["image"]) in wanted]
    if not images:
        raise DegradedBackend("数据集中没有可用图片（或图片头解析失败）")

    warnings: list[str] = []
    chosen = backend
    if chosen == "auto":
        ok = available_backends()
        for candidate in ("torchvision", "onnx", "contour"):
            if candidate in ok and (candidate != "onnx" or model):
                chosen = candidate
                break
        else:
            raise DegradedBackend(
                "没有可用的预标注后端。至少需要其一：torch+torchvision（通用检测）、"
                "onnxruntime+模型、opencv（轮廓法）。安装：" + _install_hint()
            )
    if chosen == "onnx" and not model:
        raise DegradedBackend("backend=onnx 需要同时给 model 路径")
    if chosen not in BACKENDS:
        raise DegradedBackend(f"未知后端：{chosen}（可选 {'/'.join(BACKENDS)} 或 auto）")

    if chosen == "torchvision":
        predict, meta = _backend_torchvision(root, classes, conf=conf, device=device)
    elif chosen == "onnx":
        predict, meta = _backend_onnx(Path(model).resolve(), conf=conf)
    else:
        predict, meta = _backend_contour()

    out_dir = root / out
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    skipped: list[dict[str, str]] = []
    total_boxes = 0
    processed = 0
    for index, item in enumerate(images, start=1):
        image_path = root / item["image"]
        try:
            rgb = load_rgb(image_path)
            height, width = rgb.shape[:2]
            predictions = predict(rgb)
        except DegradedBackend as exc:
            skipped.append({"image": item["image"], "reason": str(exc)})
        except Exception as exc:  # 单图失败不能中断整批
            skipped.append({"image": item["image"], "reason": f"{type(exc).__name__}: {exc}"})
        else:
            lines: list[str] = []
            for box, cls, _score in predictions:
                lines.extend(_to_yolo([box], width, height, cls))
            target = out_dir / (Path(item["image"]).stem + ".txt")
            target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            written.append(str(target.relative_to(root)))
            total_boxes += len(lines)
            processed += 1
        if on_progress is not None:
            try:
                on_progress(index, len(images), item["image"])
            except Exception:
                pass

    if processed == 0:
        raise DegradedBackend(
            "所有图片都处理失败。" + ("首个原因：" + skipped[0]["reason"] if skipped else "")
        )
    if meta.get("class_agnostic"):
        warnings.append(meta.get("note", ""))
    if processed < len(images):
        warnings.append(f"{len(images) - processed} 张图被跳过（详见 report.json 的 skipped）")
    report = {
        "dataset": str(root),
        "backend": chosen,
        "meta": meta,
        "conf": conf,
        "images": len(images),
        "processed": processed,
        "proposals": total_boxes,
        "output": out,
        "classes": classes,
        "warnings": [w for w in warnings if w],
        "skipped": skipped[:50],
        "written": written,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def review_queue(dataset: Path | str, *, proposals: str = "proposals", labels: str = "labels") -> dict[str, Any]:
    """人工审核队列：proposals 里有、labels 里没有（或为空）的图。"""
    root = Path(dataset).resolve()
    prop_dir = root / proposals
    label_dir = root / labels
    if not prop_dir.exists():
        return {"dataset": str(root), "pending": [], "count": 0, "note": "还没有预标注结果，先跑 prelabel"}
    pending: list[str] = []
    for text_file in sorted(prop_dir.glob("*.txt")):
        if text_file.name == "classes.txt":
            continue
        target = label_dir / text_file.name
        if not target.exists() or not target.read_text(encoding="utf-8", errors="replace").strip():
            pending.append(str(text_file.with_suffix(".png").name))
    return {"dataset": str(root), "pending": pending, "count": len(pending)}


def apply_proposals(dataset: Path | str, *, proposals: str = "proposals", labels: str = "labels", overwrite: bool = False) -> dict[str, Any]:
    """把 proposals 提升为正式标注（人工审核后的一步）。不覆盖已有标注，除非 overwrite。"""
    root = Path(dataset).resolve()
    prop_dir = root / proposals
    label_dir = root / labels
    label_dir.mkdir(parents=True, exist_ok=True)
    applied, kept = 0, 0
    for text_file in sorted(prop_dir.glob("*.txt")):
        if text_file.name == "classes.txt":
            continue
        target = label_dir / text_file.name
        if target.exists() and target.read_text(encoding="utf-8", errors="replace").strip() and not overwrite:
            kept += 1
            continue
        target.write_text(text_file.read_text(encoding="utf-8"), encoding="utf-8")
        applied += 1
    return {"dataset": str(root), "applied": applied, "kept": kept, "overwrite": overwrite}
