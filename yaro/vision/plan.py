"""训练 / 调优 / 部署：把"要怎么跑"写成可执行、可审计的计划，而不是聊天建议。

三个判断：

1. **harness 不拥有训练循环**。YARO 生成 recipe + 脚本 + 命令，交给沙箱或人执行，
   再把日志解析成指标 —— 这样换框架不用改 harness，也不会把 torch 拖进核心依赖。
2. **调优必须能被重放**。搜索空间、种子、每次 trial 的参数与指标都要落盘，
   否则"上次那个好结果"永远复现不出来。
3. **部署的最小证据是延迟分布**，不是"导出了 ONNX"。
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

from .dataset import scan_dataset

TRAIN_BACKENDS = {
    "torchvision": "PyTorch 官方检测模型（Apache-2.0/BSD），零 AGPL 风险",
    "ultralytics": "YOLO 生态上手最快，但仓库 AGPL-3.0：商用前必须确认许可策略",
    "custom": "你已有的训练命令；YARO 只负责预算、日志与指标",
}

DEPLOY_BACKENDS = {
    "onnx": "onnxruntime —— 跨平台基线，先有它再谈优化",
    "tensorrt": "NVIDIA 边缘/服务器，FP16/INT8 量化后必须回归",
    "openvino": "Intel CPU/iGPU 边缘盒子",
}


def build_train_plan(
    dataset: Path | str,
    *,
    backend: str = "torchvision",
    epochs: int = 30,
    imgsz: int = 640,
    batch: int = 8,
    lr: float = 0.005,
    device: str = "0",
    seed: int = 7,
    out_dir: str = "runs/plan",
) -> dict[str, Any]:
    """生成训练计划：recipe.json + 可直接执行的脚本 + 命令清单。"""
    dataset = Path(dataset)
    if backend not in TRAIN_BACKENDS:
        raise ValueError(f"未知训练后端：{backend}，可选 {sorted(TRAIN_BACKENDS)}")
    scan = scan_dataset(dataset)
    classes = scan["classes"]
    out = dataset / out_dir
    out.mkdir(parents=True, exist_ok=True)
    recipe = {
        "dataset": str(dataset.name),
        "backend": backend,
        "classes": classes,
        "epochs": int(epochs),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "lr": float(lr),
        "device": str(device),
        "seed": int(seed),
        "stats": {key: scan["stats"][key] for key in ("images", "labeled", "boxes", "per_class")},
        "license_note": TRAIN_BACKENDS[backend],
    }
    (out / "recipe.json").write_text(json.dumps(recipe, ensure_ascii=False, indent=2), encoding="utf-8")
    written = ["recipe.json"]

    if backend == "torchvision":
        (out / "train.py").write_text(_torchvision_script(recipe), encoding="utf-8")
        written.append("train.py")
        commands = [f"python {out_dir}/train.py"]
    elif backend == "ultralytics":
        data_yaml = "yaro-data.yaml"
        commands = [
            f"yolo detect train data={data_yaml} model=yolo11n.pt epochs={epochs} imgsz={imgsz} batch={batch} device={device} seed={seed}",
        ]
        (out / "commands.md").write_text(
            "# 训练命令（AGPL 提示：ultralytics 仓库为 AGPL-3.0，商用需法务确认）\n\n"
            + "\n".join(f"- `{command}`" for command in commands)
            + "\n",
            encoding="utf-8",
        )
        written.append("commands.md")
    else:
        commands = ["<你的训练命令> --data yaro-data.yaml --epochs {epochs}".format(epochs=epochs)]
        (out / "commands.md").write_text("# 自定义后端：把真实命令写在这里，YARO 负责执行与解析\n", encoding="utf-8")
        written.append("commands.md")

    # 训练后的评估/导出也提前给出，避免"训完才想怎么验收"
    (out / "acceptance.md").write_text(_acceptance_doc(), encoding="utf-8")
    written.append("acceptance.md")
    return {
        "dataset": str(dataset),
        "raw": str(dataset / out_dir),
        "recipe": recipe,
        "written": [f"{out_dir}/{name}" for name in written],
        "commands": commands,
        "warnings": [warning for warning in _plan_warnings(scan)],
    }


def _plan_warnings(scan: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    stats = scan["stats"]
    if stats["images"] == 0:
        warnings.append("数据集为空：先 dataset_scan 确认路径")
    if stats["unlabeled"]:
        warnings.append(f"{stats['unlabeled']} 张图无标注：训练时会被当负样本")
    per_class = stats["per_class"]
    if per_class:
        minority = min(per_class.values())
        if minority < 20:
            warnings.append(f"最少类别只有 {minority} 个框：先补数据或做重采样，别急着调模型")
    return warnings


def _acceptance_doc() -> str:
    return """# 验收清单（训练完成前就要定好）

- [ ] 固定评估集（不参与训练、不参与调参），样本量 ≥ 100 张或 ≥ 300 个框
- [ ] 指标按类别报：precision / recall / mAP50 / mAP50-95；工业场景另报漏检率、过杀率
- [ ] 阈值扫描：给 precision-recall 曲线，而不是只报一个默认阈值的分数
- [ ] 鲁棒性：不同光照/角度/批次各抽 20 张单独报
- [ ] 与当前线上模型的**同预算**对比（同分辨率、同硬件、同批数据）
- [ ] 模型卡：训练数据版本、超参、指标、已知失效模式
"""


def _torchvision_script(recipe: dict[str, Any]) -> str:
    classes = recipe["classes"]
    return f'''# 由 YARO 生成：torchvision 检测训练（YOLO 标注读取 + 指标 JSONL）
# 依赖：pip install torch torchvision pillow
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as TF
from PIL import Image

DATASET = Path(__file__).resolve().parent.parent
CLASSES = {classes!r}
EPOCHS = {recipe["epochs"]}
BATCH = {recipe["batch"]}
LR = {recipe["lr"]}
IMGSZ = {recipe["imgsz"]}
SEED = {recipe["seed"]}


def read_split(name: str) -> list[str]:
    path = DATASET / "splits" / f"{{name}}.txt"
    if path.is_file():
        names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if names:
            return names
    # 没有 splits 时退化成全量：递归找（split 布局是 images/train/x.png，用 glob 会漏）
    return sorted(str(p.relative_to(DATASET)) for p in (DATASET / "images").rglob("*.png"))


def label_path_for(image_path: Path) -> Path:
    """图片路径 → 标注路径：images/train/x.png → labels/train/x.txt，images/x.png → labels/x.txt。

    之前写成 image_path.parent.parent/"labels"，只在 flat 布局成立；split 布局下会读不到标签，
    于是整批图被当背景训练、loss 照样下降、脚本"成功"——静默错误比崩溃更贵。
    """
    try:
        rel = image_path.relative_to(DATASET / "images")
    except ValueError:  # 不在 images/ 下（理论上不会发生）
        rel = Path(image_path.name)
    return (DATASET / "labels" / rel).with_suffix(".txt")


def read_boxes(image_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    label = label_path_for(image_path)
    boxes, labels = [], []
    if label.is_file():
        with Image.open(image_path) as handle:
            width, height = handle.size
        for line in label.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = (float(v) for v in parts)
            boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
            labels.append(int(cls) + 1)  # 0 留给背景
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)


class YoloDataset(Dataset):
    def __init__(self, split: str) -> None:
        self.names = read_split(split)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int):
        image_path = DATASET / self.names[index]
        with Image.open(image_path) as handle:
            image = TF.to_tensor(handle.convert("RGB"))
        boxes, labels = read_boxes(image_path)
        target = {{"boxes": boxes, "labels": labels, "image_id": torch.tensor([index])}}
        return image, target


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT", num_classes=len(CLASSES) + 1).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=LR, momentum=0.9, weight_decay=5e-4)
    loader = DataLoader(YoloDataset("train"), batch_size=BATCH, shuffle=True, collate_fn=lambda rows: tuple(zip(*rows)))
    metrics_path = Path(__file__).resolve().parent / "metrics.jsonl"
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for images, targets in loader:
            images = [image.to(device) for image in images]
            targets = [{{key: value.to(device) for key, value in target.items()}} for target in targets]
            loss = sum(model(images, targets).values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)
        row = {{"epoch": epoch + 1, "loss": round(total / max(1, len(loader)), 4)}}
        metrics_path.open("a").write(json.dumps(row) + "\\n")
        print(json.dumps(row), flush=True)
    torch.save(model.state_dict(), Path(__file__).resolve().parent / "last.pth")


if __name__ == "__main__":
    main()
'''


_METRIC_PATTERNS = {
    "loss": re.compile(r"(?:loss|train_loss)[:=]\s*([0-9.]+)", re.IGNORECASE),
    "map50": re.compile(r"(?:mAP@?0?\.?5(?:0)?|map50)[:=]\s*([0-9.]+)", re.IGNORECASE),
    "map50_95": re.compile(r"(?:mAP@?0?\.5:0?\.95|map50_?95|map)[:=]\s*([0-9.]+)", re.IGNORECASE),
    "precision": re.compile(r"precision[:=]\s*([0-9.]+)", re.IGNORECASE),
    "recall": re.compile(r"recall[:=]\s*([0-9.]+)", re.IGNORECASE),
}


def parse_train_log(text: str) -> dict[str, Any]:
    """从训练日志里抽指标：兼容 JSONL 行、tqdm 行、ultralytics 表格输出。"""
    epochs: list[dict[str, Any]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict) and ("loss" in payload or "map50" in payload):
                row = {key: payload[key] for key in ("epoch", "loss", "map50", "map50_95", "precision", "recall") if key in payload}
                if "epoch" not in row:
                    row["epoch"] = len(epochs) + 1
                epochs.append(row)
                continue
        row = {}
        for name, pattern in _METRIC_PATTERNS.items():
            match = pattern.search(stripped)
            if match:
                try:
                    row[name] = float(match.group(1))
                except ValueError:
                    pass
        if row:
            row["epoch"] = len(epochs) + 1
            epochs.append(row)
    best = max((row.get("map50", 0.0) for row in epochs), default=0.0)
    return {"epochs": epochs, "count": len(epochs), "best_map50": best}


def analyze_training(history: dict[str, Any]) -> list[dict[str, str]]:
    """把训练曲线讲人话：过拟合、欠拟合、平台期、噪声。"""
    rows = history.get("epochs", [])
    findings: list[dict[str, str]] = []
    losses = [row.get("loss") for row in rows if row.get("loss") is not None]
    maps = [row.get("map50") for row in rows if row.get("map50") is not None]
    if len(losses) >= 5:
        head = sum(losses[:3]) / 3
        tail = sum(losses[-3:]) / 3
        if tail > head * 0.98:
            findings.append({"code": "loss_plateau", "detail": f"loss 前 3 轮均值 {head:.4f} → 后 3 轮 {tail:.4f}，基本没降：检查标签/学习率/预训练权重"})
        if tail > head * 1.1:
            findings.append({"code": "loss_increasing", "detail": "loss 在上升：学习率过大或数据标签错位"})
    if len(maps) >= 6:
        best_index = maps.index(max(maps))
        if best_index < len(maps) - 3 and max(maps) - maps[-1] < 0.002:
            findings.append({"code": "overfit", "detail": f"mAP 在第 {best_index + 1} 轮见顶后不再提升：加数据/增强或早停"})
    if not maps:
        findings.append({"code": "no_val_metric", "detail": "日志里没有 mAP：训练脚本必须每轮在固定评估集上报指标，否则无法判断好坏"})
    return findings


# --------------------------------------------------------------------------- #
# 调优
# --------------------------------------------------------------------------- #


def expand_space(space: dict[str, Iterable[Any]]) -> list[dict[str, Any]]:
    """把参数空间展开成 full grid（参数少时最可靠）。"""
    keys = list(space)
    if not keys:
        return []
    combos: list[dict[str, Any]] = [{}]
    for key in keys:
        values = list(space[key])
        combos = [{**combo, key: value} for combo in combos for value in values]
    return combos


def plan_trials(space: dict[str, Iterable[Any]], *, trials: int = 8, seed: int = 7, mode: str = "grid") -> list[dict[str, Any]]:
    """grid = 全组合（超出 trials 时截断）；random = 有放回随机采样。"""
    if mode == "grid":
        combos = expand_space(space)
        return combos[: max(1, int(trials))]
    if mode != "random":
        raise ValueError(f"未知搜索模式：{mode}")
    rng = random.Random(seed)
    keys = list(space)
    if not keys:
        return []
    picks: list[dict[str, Any]] = []
    for _ in range(max(1, int(trials))):
        picks.append({key: rng.choice(list(space[key])) for key in keys})
    return picks


def summarize_trials(rows: list[dict[str, Any]], *, metric: str = "map50", direction: str = "higher") -> dict[str, Any]:
    """汇总 trial 结果：最优、按指标排序、Pareto 前沿（指标 vs 成本）。"""
    usable = [row for row in rows if isinstance(row.get(metric), (int, float))]
    ordered = sorted(usable, key=lambda row: row[metric], reverse=(direction == "higher"))
    best = ordered[0] if ordered else None
    frontier: list[dict[str, Any]] = []
    for row in ordered:
        cost = float(row.get("cost_usd") or row.get("wall_s") or 0.0)
        if not frontier or all(cost <= float(other.get("cost_usd") or other.get("wall_s") or 0.0) for other in frontier):
            frontier.append({**row, "cost": cost})
    deltas: list[str] = []
    if len(ordered) >= 2:
        gaps = [abs(float(ordered[index][metric]) - float(ordered[index + 1][metric])) for index in range(len(ordered) - 1)]
        deltas.append(f"最好两个 trial 相差 {gaps[0]:.4f}")
        if gaps and sum(gaps[-3:]) / len(gaps[-3:]) < 0.002:
            deltas.append("相邻 trial 差异 < 0.002：可能已到噪声底线，别再堆算力")
    return {
        "metric": metric,
        "direction": direction,
        "trials": len(rows),
        "usable": len(usable),
        "best": best,
        "ordered": ordered,
        "pareto": frontier,
        "notes": deltas,
    }


def write_trials(dataset: Path | str, rows: list[dict[str, Any]], out_name: str = "runs/trials.jsonl") -> str:
    dataset = Path(dataset)
    path = dataset / out_name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path.relative_to(dataset))


# --------------------------------------------------------------------------- #
# 部署
# --------------------------------------------------------------------------- #


def build_deploy_plan(
    dataset: Path | str,
    *,
    weights: str = "runs/plan/last.pth",
    imgsz: int = 640,
    backends: Iterable[str] = ("onnx",),
    out_dir: str = "runs/deploy",
    latency_target_ms: float = 33.0,
) -> dict[str, Any]:
    """部署计划：导出脚本 + 延迟基准 + Dockerfile + 模型卡 + 验收命令。"""
    dataset = Path(dataset)
    chosen = [backend for backend in backends if backend in DEPLOY_BACKENDS]
    if not chosen:
        raise ValueError(f"部署后端必须是 {sorted(DEPLOY_BACKENDS)} 之一")
    out = dataset / out_dir
    out.mkdir(parents=True, exist_ok=True)
    scan = scan_dataset(dataset)
    classes = scan["classes"]
    (out / "export_onnx.py").write_text(_export_script(weights, imgsz, classes), encoding="utf-8")
    (out / "bench_onnx.py").write_text(_bench_script(imgsz, latency_target_ms), encoding="utf-8")
    # Dockerfile 里 COPY serve.py：以前从不生成它，docker build 必然失败（code review B）。
    (out / "serve.py").write_text(_serve_script(imgsz, classes), encoding="utf-8")
    (out / "Dockerfile").write_text(_dockerfile(), encoding="utf-8")
    commands = [
        f"python {out_dir}/export_onnx.py  # 产出 export/model.onnx",
        f"python {out_dir}/bench_onnx.py export/model.onnx  # 延迟分布（p50/p95）",
    ]
    if "tensorrt" in chosen:
        commands.append("trtexec --onnx=export/model.onnx --fp16 --saveEngine=export/model.fp16.engine --avgRuns=200")
        commands.append("# INT8 必须用真实标定集 + 量化后回归，否则长尾精度会断崖")
    if "openvino" in chosen:
        commands.append("mo --input_model export/model.onnx --output_dir export/openvino")
    (out / "commands.md").write_text("# 部署命令\n\n" + "\n".join(f"- `{command}`" for command in commands) + "\n", encoding="utf-8")
    (out / "model_card.md").write_text(_model_card(classes, scan, latency_target_ms), encoding="utf-8")
    return {
        "dataset": str(dataset),
        "raw": str(out),
        "written": [f"{out_dir}/{name}" for name in ("export_onnx.py", "bench_onnx.py", "serve.py", "Dockerfile", "commands.md", "model_card.md")],
        "backends": chosen,
        "commands": commands,
        "latency_target_ms": float(latency_target_ms),
        "deploy_notes": [
            f"{DEPLOY_BACKENDS[name]}：{DEPLOY_BACKENDS[name]}" for name in chosen
        ],
    }


def _export_script(weights: str, imgsz: int, classes: list[str]) -> str:
    return f'''# 由 YARO 生成：把训练权重导出为 ONNX（动态 batch）
import torch

WEIGHTS = {weights!r}
IMGSZ = {imgsz}
CLASSES = {classes!r}


def main() -> None:
    # torchvision 演示：换成你自己的模型构造，只要 forward 返回 predictions
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn

    model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, num_classes=len(CLASSES) + 1)
    state = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    dummy = torch.rand(1, 3, IMGSZ, IMGSZ)
    torch.onnx.export(
        model, (dummy,), "export/model.onnx", input_names=["images"], output_names=["boxes", "labels", "scores"],
        dynamic_axes={{"images": {{0: "batch"}}, "boxes": {{0: "num"}}, "labels": {{0: "num"}}, "scores": {{0: "num"}}}},
        opset_version=17,
    )
    print("导出完成：export/model.onnx")


if __name__ == "__main__":
    main()
'''


def _bench_script(imgsz: int, latency_target_ms: float) -> str:
    return f'''# 由 YARO 生成：ONNX 延迟基准（p50/p95/p99 + 吞吐），报告"能不能上线"的最小证据
import json
import statistics
import sys
import time

import numpy as np
import onnxruntime as ort

IMGSZ = {imgsz}
TARGET_MS = {latency_target_ms}
ITERS = 100


def main() -> None:
    model = sys.argv[1] if len(sys.argv) > 1 else "export/model.onnx"
    session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    dummy = np.random.rand(1, 3, IMGSZ, IMGSZ).astype("float32")
    name = session.get_inputs()[0].name
    for _ in range(5):
        session.run(None, {{name: dummy}})
    timings = []
    for _ in range(ITERS):
        start = time.perf_counter()
        session.run(None, {{name: dummy}})
        timings.append((time.perf_counter() - start) * 1000)
    timings.sort()
    p50 = statistics.median(timings)
    p95 = timings[int(len(timings) * 0.95) - 1]
    report = {{"model": model, "p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "fps": round(1000 / p50, 1), "target_ms": TARGET_MS, "pass": p95 <= TARGET_MS}}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
'''


def _serve_script(imgsz: int, classes: list[str]) -> str:
    """最小推理服务：不引入 web 框架，只用标准库 http.server（镜像更小、依赖更少）。

    生产环境请换成你们自己的服务框架并加上鉴权、并发与超时控制 —— 这个脚本的
    目标是"证明模型能在容器里被加载并返回一次推理"，不是可直接上线的服务。
    """
    return f'''"""由 YARO 生成的推理服务（最小可用，勿直接上线）。

    GET  /health  -> {{"ok": true, "model": ...}}
    POST /predict -> 请求体为 raw 图像字节，返回检测框 JSON
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import onnxruntime as ort

MODEL = sys.argv[1] if len(sys.argv) > 1 else "export/model.onnx"
IMGSZ = {imgsz}
CLASSES = {json.dumps(classes, ensure_ascii=False)}
session = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name


def _preprocess(raw: bytes):
    # TODO: 换成真实解码（Pillow/OpenCV）。这里只做"可加载、可推理"的最小链路。
    arr = np.frombuffer(raw, dtype=np.uint8).astype("float32") / 255.0
    need = 3 * IMGSZ * IMGSZ
    arr = np.resize(arr, need).reshape(1, 3, IMGSZ, IMGSZ)
    return arr


class Handler(BaseHTTPRequestHandler):
    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.split("?")[0] == "/health":
            self._json({{"ok": True, "model": MODEL, "imgsz": IMGSZ, "classes": CLASSES}})
        else:
            self._json({{"error": "not found"}}, 404)

    def do_POST(self):  # noqa: N802
        if self.path.split("?")[0] != "/predict":
            return self._json({{"error": "not found"}}, 404)
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        outputs = session.run(None, {{input_name: _preprocess(raw)}})
        boxes = outputs[0].tolist() if outputs else []
        self._json({{"boxes": boxes[:100], "count": len(boxes)}})

    def log_message(self, *args):  # 生产环境请接结构化日志
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
'''


def _dockerfile() -> str:
    return """# 由 YARO 生成：最小推理镜像（CPU 版；GPU 版换 nvidia/cuda 基础镜像）
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir onnxruntime numpy
COPY export/ /app/export/
COPY serve.py /app/serve.py
EXPOSE 8000
ENTRYPOINT ["python", "serve.py"]
"""


def _model_card(classes: list[str], scan: dict[str, Any], latency_target_ms: float) -> str:
    stats = scan["stats"]
    return f"""# 模型卡（部署前必须填完整）

- 任务：目标检测 / 分割（按实际改）
- 类别：{len(classes)} 类 —— {', '.join(classes) if classes else '（未定义）'}
- 训练数据：{stats['images']} 张图 / {stats['boxes']} 个框；版本 `{scan.get('dataset', '-')}`
- 延迟目标：p95 ≤ {latency_target_ms} ms / 帧
- 精度指标：待填（mAP50 / mAP50-95 / 分类型 precision-recall）
- 已知失效模式：待填（小目标、反光、遮挡、罕见类别）
- 数据合规：原始图不出本机；导出模型不含训练数据
"""


def read_trials(dataset: Path | str, name: str = "runs/trials.jsonl") -> list[dict[str, Any]]:
    path = Path(dataset) / name
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
