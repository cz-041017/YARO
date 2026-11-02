"""真训练执行器：把「训练计划」变成「真的跑起来的一版小模型」。

定位（按 m02564 的结论）：这是给**蒸馏**用的学生模型训练器——
- 默认要求 CUDA：训练是重资产，跑在 CPU 上对"快速可用的小模型"没有意义；
  显式传 ``device="cpu"`` 才允许降级（用于冒烟验证，而不是生产）。
- 默认学生很小：``fasterrcnn_mobilenet_v3_large_fpn``（COCO 预训练迁移），
  目标就是"在产线上跑得动"，不是刷榜。
- 训练脚本由 harness 生成并受控执行（yaro.exec.runner），不接收模型自由拼接的命令。
- 预算优先：``max_minutes`` 到点即停，保留 ``best.pt``；每轮写 ``metrics.jsonl``（可流式看）。

不做的事（故意）：
- 不在 agent 进程内 import torch（崩溃会连带丢会话；超时也无法清理显存）；
- 不信任模型自报分数：mAP 由这里自己算，并写进 metrics.jsonl（裁判与执行分离）。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..contracts import ContractError, DegradedBackend
from ..exec import runner
from . import plan
from .dataset import scan_dataset, write_split, plan_split

TRAIN_MODELS = {
    "student-fasterrcnn-mnv3": "fastercnn_mobilenet_v3_large_fpn（COCO 迁移，最小最省显存，默认）",
    "student-retinanet-r50": "retinanet_resnet50_fpn_v2（精度更高、更慢，显存 ≥8G）",
}
DEFAULT_MODEL = "student-fasterrcnn-mnv3"


@dataclass
class TrainConfig:
    model: str = DEFAULT_MODEL
    epochs: int = 12
    imgsz: int = 512
    batch: int = 2
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    device: str = "auto"  # auto | cuda | cuda:0 | cpu
    seed: int = 7
    max_minutes: float = 60.0
    workers: int = 0  # Windows 上多进程 DataLoader 常出问题，默认 0
    score_thresh: float = 0.05  # 评估时保留阈值（mAP 需要低阈值）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TrainResult:
    ok: bool
    dataset: str
    config: dict[str, Any]
    out_dir: str
    metrics_path: str = ""
    best_checkpoint: str = ""
    last_checkpoint: str = ""
    epochs_done: int = 0
    best_map50: float = 0.0
    stopped_reason: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)
    degradations: list[str] = field(default_factory=list)
    tail: str = ""
    duration_s: float = 0.0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- 能力检查


def train_support(*, device: str = "auto") -> dict[str, Any]:
    """判断本机能否训练。CUDA 是默认门槛（蒸馏场景下 CPU 训练没有工程意义）。"""
    from ..exec.env import capabilities, suggest_install

    caps = capabilities()
    has_torch = caps["can_train"]
    cuda = caps["cuda"]
    requested_cpu = device == "cpu"
    if not has_torch:
        return {
            "ok": False,
            "reason": "缺少 torch / torchvision",
            "hint": suggest_install(train=True),
            "caps": caps,
        }
    if not cuda.get("available") and not requested_cpu:
        return {
            "ok": False,
            "reason": f"没有可用 GPU（{cuda.get('note', '未知')}）；训练默认只在 CUDA 机器上启用",
            "hint": "换一台有 CUDA 的机器；或显式传 device='cpu' 只做冒烟验证（会很慢）",
            "caps": caps,
        }
    return {"ok": True, "reason": "", "caps": caps, "device": _resolve_device(caps, device)}


def _resolve_device(caps: dict[str, Any], device: str) -> str:
    if device in ("auto", ""):
        if caps["cuda"].get("available"):
            return "cuda"
        return "cpu"
    return device


# --------------------------------------------------------------------------- 数据集准备


def _ensure_splits(root: Path, *, ratios: tuple[float, float, float] = (0.7, 0.2, 0.1), seed: int = 7) -> dict[str, Any]:
    """训练脚本要读 splits/{train,val,test}.txt；没有就按可复现规则切一份。"""
    split_dir = root / "splits"
    needed = [split_dir / f"{name}.txt" for name in ("train", "val", "test")]
    if all(path.exists() for path in needed):
        return {"created": False, "splits": [str(p.relative_to(root)) for p in needed]}
    result = plan_split(root, ratios=ratios, seed=seed)
    written = write_split(root, result)
    return {"created": True, "splits": written, "warnings": result.get("warnings", [])}


# --------------------------------------------------------------------------- 生成脚本

_TRAIN_SCRIPT = r'''
# 由 YARO 生成的训练脚本（受控执行）。改这里没有意义——改 yaro/vision/train_run.py 的模板。
import json, os, random, time
from pathlib import Path

CFG = __CFG__

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, retinanet_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.transforms import functional as TF
from PIL import Image

ROOT = Path(CFG["dataset"])
OUT = Path(CFG["out_dir"])
OUT.mkdir(parents=True, exist_ok=True)
METRICS = OUT / "metrics.jsonl"
LOG = OUT / "train.log"

def log(msg):
    line = str(msg)
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")

def read_split(name):
    path = ROOT / "splits" / (name + ".txt")
    if not path.exists():
        return []
    items = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            items.append(ROOT / raw)
    return items

def load_targets(image_path):
    label_path = Path(str(image_path).replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)).with_suffix(".txt")
    if not label_path.exists():
        alt = ROOT / "labels" / (image_path.stem + ".txt")
        label_path = alt if alt.exists() else label_path
    boxes, labels = [], []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cls = int(float(parts[0])); cx, cy, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
            labels.append(cls + 1)  # 0 留给背景
    return boxes, labels

class YoloDataset(Dataset):
    def __init__(self, paths, imgsz, train):
        self.paths = paths
        self.imgsz = imgsz
        self.train = train
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        w, h = image.size
        boxes, labels = load_targets(path)
        scale = self.imgsz / max(w, h)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        image = TF.resize(image, [new_h, new_w])
        tensor = TF.pil_to_tensor(image).float() / 255.0
        if self.train and random.random() < 0.5:
            tensor = torch.flip(tensor, dims=[2])
            boxes = [[1 - b[2], b[1], 1 - b[0], b[3]] for b in boxes]
        pixel_boxes = [[b[0] * new_w, b[1] * new_h, b[2] * new_w, b[3] * new_h] for b in boxes]
        target = {
            "boxes": torch.tensor(pixel_boxes, dtype=torch.float32) if pixel_boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }
        return tensor, target

def collate(batch):
    return tuple(zip(*batch))

def build_model(num_classes, name, weights_default=True):
    if name == "student-retinanet-r50":
        model = retinanet_resnet50_fpn_v2(weights="DEFAULT" if weights_default else None)
        in_features = model.head.classification_head.conv[0].out_channels
        num_anchors = model.head.classification_head.num_anchors
        model.head.classification_head = RetinaNetClassificationHead(
            in_features, num_anchors, num_classes, norm_layer=torch.nn.BatchNorm2d
        )
        return model
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT" if weights_default else None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return torch.zeros((len(a), len(b)))
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter).clamp(min=1e-6)

def average_precision(recalls, precisions):
    mrec = [0.0] + list(recalls) + [1.0]
    mpre = [0.0] + list(precisions) + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap, prev = 0.0, 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap

@torch.no_grad()
def evaluate(model, loader, num_classes, score_thresh, iou_thresh=0.5):
    model.eval()
    per_class = {c: {"tp": [], "scores": [], "npos": 0} for c in range(1, num_classes)}
    total_loss = 0.0
    batches = 0
    for images, targets in loader:
        images = [image.to(DEVICE) for image in images]
        moved = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        # 检测模型只有 train() 模式下 forward(images, targets) 才返回 loss dict；
        # 以前在 eval 模式下调用，返回值没有 loss 键 → val_loss 结构性恒 0，
        # 训练日志里的"验证损失"是假的（code review B-6）。这里单独用 train 模式
        # 前向算一次 loss（no_grad，只更新 BN running stats 不影响权重），再 eval 出框。
        model.train()
        losses = model(images, moved)
        model.eval()
        if isinstance(losses, dict):
            total_loss += float(sum(v.item() for v in losses.values()))
            batches += 1
        out = model(images)
        for pred, target in zip(out, targets):
            scores = pred["scores"].cpu()
            labels = pred["labels"].cpu()
            boxes = pred["boxes"].cpu()
            gt_boxes = target["boxes"]
            gt_labels = target["labels"]
            keep = scores >= score_thresh
            scores, labels, boxes = scores[keep], labels[keep], boxes[keep]
            ious = box_iou(boxes, gt_boxes)
            for cls in range(1, num_classes):
                mask_pred = labels == cls
                mask_gt = gt_labels == cls
                npos = int(mask_gt.sum().item())
                per_class[cls]["npos"] += npos
                if mask_pred.sum() == 0:
                    continue
                pred_boxes = boxes[mask_pred]
                pred_scores = scores[mask_pred]
                order = torch.argsort(-pred_scores)
                pred_boxes = pred_boxes[order]
                pred_scores = pred_scores[order]
                gt = gt_boxes[mask_gt]
                matched = torch.zeros(len(pred_boxes), dtype=torch.bool)
                used_gt = set()
                for i, box in enumerate(pred_boxes):
                    if len(gt) == 0:
                        break
                    overlaps = box_iou(box[None], gt)[0]
                    best = int(torch.argmax(overlaps).item())
                    if float(overlaps[best]) >= iou_thresh and best not in used_gt:
                        used_gt.add(best)
                        matched[i] = True
                tp = matched.float().tolist()
                per_class[cls]["tp"].extend(tp)
                per_class[cls]["scores"].extend(pred_scores.tolist())
    aps = []
    for cls, bucket in per_class.items():
        if bucket["npos"] <= 0 or not bucket["scores"]:
            continue
        order = sorted(range(len(bucket["scores"])), key=lambda i: -bucket["scores"][i])
        tp = np.asarray([bucket["tp"][i] for i in order], dtype=float)
        ctp = np.cumsum(tp)
        cfp = np.cumsum(1 - tp)
        recalls = (ctp / bucket["npos"]).tolist()
        precisions = (ctp / np.maximum(ctp + cfp, 1e-6)).tolist()
        aps.append(average_precision(recalls, precisions))
    return {
        "val_loss": round(total_loss / batches, 6) if batches else 0.0,
        "map50": round(float(np.mean(aps)) if aps else 0.0, 6),
        "classes_evaluated": len(aps),
    }

def main():
    global DEVICE
    torch.manual_seed(CFG["seed"]); random.seed(CFG["seed"]); np.random.seed(CFG["seed"])
    DEVICE = CFG["device"]
    train_paths = read_split("train")
    val_paths = read_split("val")
    if not train_paths:
        raise SystemExit("splits/train.txt 为空——先跑 dataset_split")
    classes = []
    names_file = ROOT / "classes.txt"
    if names_file.exists():
        classes = [line.strip() for line in names_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    num_classes = len(classes) + 1
    if num_classes <= 1:
        num_classes = 2
    log(json.dumps({"event": "start", "train": len(train_paths), "val": len(val_paths),
                    "classes": len(classes), "device": DEVICE, "model": CFG["model"]}, ensure_ascii=False))
    model = build_model(num_classes, CFG["model"], True).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=CFG["lr"], momentum=CFG["momentum"],
                                weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, CFG["epochs"]))
    train_loader = DataLoader(YoloDataset(train_paths, CFG["imgsz"], True), batch_size=CFG["batch"],
                              shuffle=True, num_workers=CFG["workers"], collate_fn=collate)
    val_loader = DataLoader(YoloDataset(val_paths or train_paths[:1], CFG["imgsz"], False),
                            batch_size=1, shuffle=False, num_workers=CFG["workers"], collate_fn=collate)
    best = -1.0
    started = time.time()
    for epoch in range(1, CFG["epochs"] + 1):
        if (time.time() - started) / 60.0 > CFG["max_minutes"]:
            log(json.dumps({"event": "budget_stop", "epoch": epoch}, ensure_ascii=False))
            break
        model.train()
        running = 0.0
        steps = 0
        for images, targets in train_loader:
            images = [image.to(DEVICE) for image in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.item()); steps += 1
        scheduler.step()
        metrics = evaluate(model, val_loader, num_classes, CFG["score_thresh"])
        row = {"epoch": epoch, "loss": round(running / max(steps, 1), 6), **metrics,
               "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
               "seconds": round(time.time() - started, 2)}
        with METRICS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        log(json.dumps({"event": "epoch", **row}, ensure_ascii=False))
        score = metrics["map50"]
        if score > best:
            best = score
            torch.save({"model": CFG["model"], "num_classes": num_classes,
                        "classes": classes, "state_dict": model.state_dict()}, OUT / "best.pt")
        torch.save({"model": CFG["model"], "num_classes": num_classes,
                    "classes": classes, "state_dict": model.state_dict()}, OUT / "last.pt")
    log(json.dumps({"event": "done", "best_map50": best,
                    "minutes": round((time.time() - started) / 60.0, 2)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
'''


def _render_script(config: TrainConfig, dataset: Path, out_dir: Path) -> str:
    payload = {
        "dataset": str(dataset),
        "out_dir": str(out_dir),
        "model": config.model,
        "epochs": int(config.epochs),
        "imgsz": int(config.imgsz),
        "batch": int(config.batch),
        "lr": float(config.lr),
        "momentum": float(config.momentum),
        "weight_decay": float(config.weight_decay),
        "device": config.device,
        "seed": int(config.seed),
        "max_minutes": float(config.max_minutes),
        "workers": int(config.workers),
        "score_thresh": float(config.score_thresh),
    }
    return _TRAIN_SCRIPT.replace("__CFG__", json.dumps(payload, ensure_ascii=False))


# --------------------------------------------------------------------------- 主入口


def train_dataset(
    dataset: Path | str,
    *,
    config: TrainConfig | None = None,
    root: Path | str | None = None,
    out_dir: Path | str | None = None,
    on_line=None,
) -> TrainResult:
    """真的跑一次训练。返回结构化结果（含 metrics 路径、best.pt、诊断结论）。"""
    config = config or TrainConfig()
    dataset_path = Path(dataset).resolve()
    if not dataset_path.exists():
        return TrainResult(ok=False, dataset=str(dataset_path), config=config.to_dict(), out_dir="", error=f"数据集不存在：{dataset_path}")

    support = train_support(device=config.device)
    caps = support.get("caps", {})
    if not support["ok"]:
        return TrainResult(
            ok=False,
            dataset=str(dataset_path),
            config=config.to_dict(),
            out_dir="",
            error=support["reason"],
            findings=[{"code": "env_missing", "detail": support.get("hint", "")}],
        )
    config.device = support.get("device", config.device)

    scan = scan_dataset(dataset_path)
    if scan["format"] == "unknown":
        return TrainResult(
            ok=False,
            dataset=str(dataset_path),
            config=config.to_dict(),
            out_dir="",
            error=f"认不出数据集结构：{dataset_path}（需要 images/ + labels/ 或 YOLO 分目录）",
        )
    split_info = _ensure_splits(dataset_path, seed=config.seed)

    out = Path(out_dir).resolve() if out_dir else (dataset_path / "runs" / "train")
    out.mkdir(parents=True, exist_ok=True)
    script_path = out / "train_generated.py"
    script_path.write_text(_render_script(config, dataset_path, out), encoding="utf-8")
    # 每次训练前清掉上一轮的所有产物：只删 metrics/log 时，上一轮的 best.pt
    # 会被当成这次的结果上报（code review B-5）。
    metrics_file = out / "metrics.jsonl"
    log_file = out / "train.log"
    for stale in (metrics_file, log_file, out / "best.pt", out / "last.pt", out / "train.log"):
        if stale.exists():
            stale.unlink()

    minutes_budget = max(1.0, float(config.max_minutes) + 10.0)
    outcome = runner.run(
        runner.RunSpec(
            argv=[__import__("sys").executable, str(script_path)],
            cwd=dataset_path,
            timeout_s=minutes_budget * 60.0,
            label="train",
        ),
        on_line=on_line,
    )

    rows: list[dict[str, Any]] = []
    if metrics_file.exists():
        for line in metrics_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    best_map = max((float(row.get("map50", 0.0)) for row in rows), default=0.0)
    history = plan.parse_train_log("\n".join(json.dumps(row) for row in rows)) if rows else {"epochs": [], "count": 0, "best_map50": 0.0}
    findings = plan.analyze_training(history.get("epochs", [])) if history.get("epochs") else []

    stopped = ""
    if outcome.timed_out:
        stopped = "timeout"
    elif any('"budget_stop"' in line for line in outcome.stdout.splitlines()):
        stopped = "budget"
    elif outcome.returncode != 0:
        stopped = "error"
    elif rows:
        stopped = "epochs"

    degradations: list[str] = []
    if not caps.get("cuda", {}).get("available") and config.device == "cpu":
        degradations.append("cpu_training")
    if split_info.get("warnings"):
        degradations.append("split_ratio")

    return TrainResult(
        ok=outcome.returncode == 0 and bool(rows),
        dataset=str(dataset_path),
        config=config.to_dict(),
        out_dir=str(out),
        metrics_path=str(metrics_file) if metrics_file.exists() else "",
        best_checkpoint=str(out / "best.pt") if (out / "best.pt").exists() else "",
        last_checkpoint=str(out / "last.pt") if (out / "last.pt").exists() else "",
        epochs_done=len(rows),
        best_map50=round(best_map, 6),
        stopped_reason=stopped,
        findings=findings,
        degradations=degradations,
        tail=outcome.tail(4000),
        duration_s=round(outcome.duration_s, 2),
        error=outcome.error or ("" if outcome.returncode == 0 else outcome.tail(1500)),
    )
