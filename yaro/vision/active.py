"""主动学习：用训练好的 student 在未标注图上跑推理，按不确定性挑最难例。

难例打分：``1 - max(检测置信度)``；无检测 → 1.0（最难）。分越高的图，student 越不确定，
越值得送回 teacher 重新预标注、送人工审核——这就是"难例回灌"闭环（label→train→选难例→回灌）。

跑在受控子进程里（与 train 一致）：不把 torch import 进 agent 进程，崩溃/显存泄漏隔离；
超时由 runner 兜底。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..contracts import DegradedBackend
from ..exec import runner
from .dataset import scan_dataset

DEFAULT_WEIGHTS = "runs/train/best.pt"


def _render_script(dataset: Path, weights: Path, unlabeled: list[str], imgsz: int) -> str:
    return _SELECT_SCRIPT.replace(
        "__CFG__",
        json.dumps(
            {
                "dataset": str(dataset),
                "weights": str(weights),
                "unlabeled": unlabeled,
                "imgsz": imgsz,
            },
            ensure_ascii=False,
        ),
    )


def _find_unlabeled(root: Path) -> list[str]:
    """未标注图：labels 里缺失或为空的图，返回**完整相对路径**（含 split）。

    用 ``scan_dataset`` 的 ``item["label"]``（已按 split 拼好 ``labels/<split>/<stem>.txt``），
    而不是自己用扁平 ``labels/<stem>.txt`` 重拼——否则 yolo-split 数据集每张图都会被
    误判成未标注（M1 修复）。
    """
    scan = scan_dataset(root)
    if scan["format"] != "yolo":
        # 只支持 YOLO 布局（主动学习的学生权重是 YOLO 检测头，对 COCO 无意义）；
        # "unknown" 之外还有 "coco"，不拦会把它每一张图都判成未标注（L-1 修复）。
        raise DegradedBackend(f"主动学习仅支持 YOLO 数据集，实际是 {scan['format']!r}：{root}")
    unlabeled: list[str] = []
    for item in scan.get("items", []):
        rel = item.get("image")
        if not rel:
            continue
        label_rel = item.get("label")
        if label_rel is None:
            unlabeled.append(rel)  # 缺标注
            continue
        label_path = root / label_rel
        if not label_path.read_text(encoding="utf-8", errors="replace").strip():
            unlabeled.append(rel)  # 空标注
    return unlabeled


def select_hard_examples(
    dataset: Path | str,
    *,
    n: int = 20,
    weights: str = DEFAULT_WEIGHTS,
    imgsz: int = 512,
) -> dict[str, Any]:
    """挑 ``n`` 张 student 最不确定的未标注图，返回 ``{"hard":[完整相对路径], "count", "scores", ...}``。

    没有未标注图 → ``count=0``（无 ``ok`` 键，与失败天然区分）；没有学生权重/没有 torch →
    ``ok=False`` 降级（诚实，不假装挑出来了）。``imgsz`` 与训练时一致（默认 512，见
    train_run.py 的 TrainConfig），否则推理尺度漂移、难例排序失真。
    """
    root = Path(dataset).resolve()
    if not root.exists():
        raise DegradedBackend(f"数据集不存在：{root}")

    unlabeled = _find_unlabeled(root)
    if not unlabeled:
        return {"dataset": str(root), "hard": [], "count": 0, "scores": {},
                "note": "没有未标注图片可挑（labels 已覆盖全部图片）"}

    weights_path = root / weights
    if not weights_path.exists():
        return {"dataset": str(root), "ok": False, "hard": [], "count": 0,
                "scores": {}, "error": f"学生权重不存在：{weights_path}（先训练出 best.pt）"}

    from ..exec.env import capabilities

    caps = capabilities()
    if not caps.get("can_train"):
        return {"dataset": str(root), "ok": False, "hard": [], "count": 0,
                "scores": {}, "error": "缺少 torch/torchvision，无法跑学生推理（难例挑选需要 torch）"}

    script = root / "runs" / "select_hard_generated.py"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(_render_script(root, weights_path, unlabeled, imgsz), encoding="utf-8")

    outcome = runner.run(
        runner.RunSpec(
            argv=[__import__("sys").executable, str(script)],
            cwd=root,
            timeout_s=30 * 60.0,
            label="select_hard",
        ),
    )

    scores: dict[str, float] = {}
    errors: list[dict[str, str]] = []
    parse_failed = False
    if outcome.returncode == 0:
        try:
            rows = json.loads(outcome.stdout.strip() or "[]")
            for row in rows:
                rel = row["image"]
                scores[rel] = float(row.get("score", 1.0))
                if row.get("error"):
                    errors.append({"image": rel, "reason": row["error"]})
        except (json.JSONDecodeError, KeyError, TypeError):
            parse_failed = True
            errors.append({"image": "", "reason": f"解析学生输出失败：{outcome.tail(400)}"})
    else:
        errors.append({"image": "", "reason": outcome.error or outcome.tail(800)})

    # 分高者最难（不确定），取前 n；hard 用完整相对路径（M1 修复：不折叠成 stem，否则
    # 跨 split 同名图碰撞，下一轮 prelabel subset 会命中多张图）。
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[: int(n)]
    hard = [rel for rel, _score in ranked]
    return {
        "dataset": str(root),
        "ok": outcome.returncode == 0 and not parse_failed,
        "hard": hard,
        "count": len(hard),
        "scores": {rel: scores[rel] for rel in hard},
        "unlabeled_total": len(unlabeled),
        "errors": errors[:20],
    }


_SELECT_SCRIPT = r'''
# 由 YARO 生成的难例挑选脚本（受控执行）。改这里没有意义——改 yaro/vision/active.py 的模板。
import json
from pathlib import Path

import torch

CFG = __CFG__
ROOT = Path(CFG["dataset"])


def build_model(num_classes, name):
    from torchvision.models.detection import (
        fasterrcnn_mobilenet_v3_large_fpn,
        retinanet_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.retinanet import RetinaNetClassificationHead

    if name == "student-retinanet-r50":
        model = retinanet_resnet50_fpn_v2(weights=None)
        in_features = model.head.classification_head.conv[0].out_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_features, num_anchors, num_classes, norm_layer=torch.nn.BatchNorm2d
        )
        return model
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    from PIL import Image
    from torchvision.transforms import functional as TF

    # 本 checkpoint 由 YARO train 脚本自己保存（只含 str/int/list/OrderedDict[str,Tensor]），
    # 属受信内容，weights_only=True 可安全反序列化（禁 pickle 任意对象）。
    ckpt = torch.load(CFG["weights"], map_location="cpu", weights_only=True)
    model = build_model(int(ckpt.get("num_classes", 2)), ckpt.get("model", "student-fasterrcnn-mnv3"))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out = []
    for rel in CFG["unlabeled"]:
        path = ROOT / rel
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - 单图失败不能中断整批
            out.append({"image": rel, "error": f"{type(exc).__name__}: {exc}"})
            continue
        # 与训练同尺度（train_run.py YoloDataset.__getitem__：长边缩到 imgsz，保持宽高比），
        # 否则推理尺度漂移、难例排序失真。
        w, h = image.size
        scale = CFG["imgsz"] / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = TF.resize(image, [new_h, new_w])
        tensor = TF.pil_to_tensor(image).float() / 255.0
        with torch.no_grad():
            pred = model([tensor])[0]
        scores = pred["scores"].tolist()
        score = 1.0 - (max(scores) if scores else 0.0)
        out.append({"image": rel, "score": round(score, 6), "boxes": len(scores)})
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''
