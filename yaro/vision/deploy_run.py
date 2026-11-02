"""真部署执行器：真的导出 ONNX，并真的测延迟。

为什么放在子进程里做：``torch.onnx.export`` 会拉起 torch、写入 ONNX protobuf，
在 agent 进程里做会把几百 MB 的显存/内存钉住，还可能因为 CUDA 上下文把后续训练挤掉。
所以这里只做两件事：
1. 生成一个导出脚本 → ``yaro.exec.runner`` 跑它（子进程，超时可杀）；
2. 生成一个 bench 脚本 → 跑它，读回 p50/p95/mean。

判定口径（跟 CONTRACT 一致）：只导出不算落地；必须有延迟数字，并且和
``latency_target_ms`` 比。没有目标延迟时只报数，不给"通过"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..exec import runner
from ..exec.env import capabilities, suggest_install

DEFAULT_IMGSZ = 640
DEFAULT_RUNS = 20


_EXPORT_SCRIPT = r'''
# 由 YARO 生成的 ONNX 导出脚本（受控执行）。
import json, sys
from pathlib import Path

CFG = __CFG__

import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, retinanet_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead

def build(model_name, num_classes):
    if model_name == "student-retinanet-r50":
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
    ckpt_path = Path(CFG["weights"])
    if not ckpt_path.exists():
        print(json.dumps({"ok": False, "error": "checkpoint not found: %s" % ckpt_path}))
        return 1
    # weights_only=True：以前用 False，等于对别人给的 .pt 执行任意 pickle（RCE）。
    # 我们自己的 checkpoint 只有 dict/tensor/str/list，不需要完整 pickle 反序列化。
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    model_name = ckpt.get("model", CFG["model"])
    num_classes = int(ckpt.get("num_classes", 2))
    model = build(model_name, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dummy = torch.rand(1, 3, CFG["imgsz"], CFG["imgsz"])
    out_path = Path(CFG["onnx"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=["images"],
        output_names=["boxes", "labels", "scores"],
        opset_version=int(CFG["opset"]),
        do_constant_folding=True,
    )
    print(json.dumps({
        "ok": True,
        "onnx": str(out_path),
        "model": model_name,
        "num_classes": num_classes,
        "classes": ckpt.get("classes", []),
        "bytes": out_path.stat().st_size if out_path.exists() else 0,
    }))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


_BENCH_SCRIPT = r'''
# 由 YARO 生成的 ONNX 延迟测试脚本（受控执行）。
import json, statistics, time
from pathlib import Path

CFG = __CFG__

import numpy as np
import onnxruntime as ort

def main():
    path = Path(CFG["onnx"])
    if not path.exists():
        print(json.dumps({"ok": False, "error": "onnx not found: %s" % path}))
        return 1
    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                 if p in ort.get_available_providers()]
    session = ort.InferenceSession(str(path), providers=providers)
    imgsz = int(CFG["imgsz"])
    dummy = np.random.rand(1, 3, imgsz, imgsz).astype("float32")
    input_name = session.get_inputs()[0].name
    for _ in range(int(CFG["warmup"])):
        session.run(None, {input_name: dummy})
    timings = []
    for _ in range(int(CFG["runs"])):
        started = time.perf_counter()
        session.run(None, {input_name: dummy})
        timings.append((time.perf_counter() - started) * 1000.0)
    timings.sort()
    timings_arr = np.asarray(timings)
    print(json.dumps({
        "ok": True,
        "providers": providers,
        "runs": len(timings),
        "p50_ms": round(float(np.percentile(timings_arr, 50)), 3),
        "p95_ms": round(float(np.percentile(timings_arr, 95)), 3),
        "mean_ms": round(float(timings_arr.mean()), 3),
        "min_ms": round(float(timings_arr.min()), 3),
        "max_ms": round(float(timings_arr.max()), 3),
        "stdev_ms": round(float(statistics.pstdev(timings)), 3),
    }))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
'''


def export_support() -> dict[str, Any]:
    caps = capabilities()
    packages = caps.get("packages", {})
    missing = []
    if not packages.get("torch", {}).get("present"):
        missing.append("torch")
    if not packages.get("torchvision", {}).get("present"):
        missing.append("torchvision")
    if not packages.get("onnx", {}).get("present"):
        missing.append("onnx")
    return {
        "ok": not missing,
        "missing": missing,
        "hint": suggest_install(train=False) if missing else "",
        "onnxruntime": bool(packages.get("onnxruntime", {}).get("present")),
        "cuda": caps.get("cuda", {}),
    }


def _parse_last_json(text: str) -> dict[str, Any]:
    """子进程会打印多行日志；只认最后一行能解析成 JSON 的。"""
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def deploy_run(
    *,
    weights: Path | str,
    out_dir: Path | str | None = None,
    imgsz: int = DEFAULT_IMGSZ,
    runs: int = DEFAULT_RUNS,
    warmup: int = 3,
    opset: int = 17,
    latency_target_ms: float | None = None,
    model: str = "student-fasterrcnn-mnv3",
    on_line=None,
) -> dict[str, Any]:
    """导出 ONNX + 测延迟。返回结构化结果，永远不抛（用 ok/error 表达）。"""
    support = export_support()
    if not support["ok"]:
        return {
            "ok": False,
            "error": f"缺少依赖：{', '.join(support['missing'])}",
            "hint": support["hint"],
            "written": [],
        }
    weights_path = Path(weights).resolve()
    if not weights_path.exists():
        return {"ok": False, "error": f"checkpoint 不存在：{weights_path}", "written": []}

    out = Path(out_dir).resolve() if out_dir else (weights_path.parent.parent / "deploy")
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / "model.onnx"
    script = (out / "export_onnx_generated.py")
    script.write_text(
        _EXPORT_SCRIPT.replace(
            "__CFG__",
            json.dumps(
                {
                    "weights": str(weights_path),
                    "onnx": str(onnx_path),
                    "imgsz": int(imgsz),
                    "opset": int(opset),
                    "model": model,
                },
                ensure_ascii=False,
            ),
        ),
        encoding="utf-8",
    )

    written: list[str] = [str(script)]
    export_outcome = runner.run(
        runner.RunSpec(
            argv=[__import__("sys").executable, str(script)],
            cwd=out,
            timeout_s=900.0,
            label="export-onnx",
        ),
        on_line=on_line,
    )
    export_payload = _parse_last_json(export_outcome.stdout)
    if not export_payload.get("ok"):
        error = export_payload.get("error") or export_outcome.error or export_outcome.tail(1500)
        return {
            "ok": False,
            "error": error,
            "written": written,
            "tail": export_outcome.tail(2000),
            "dependency_note": "" if support["onnxruntime"] else "未安装 onnxruntime，导出后无法测延迟",
        }
    written.append(str(onnx_path))

    bench: dict[str, Any] = {}
    if support["onnxruntime"]:
        bench_script = out / "bench_onnx_generated.py"
        bench_script.write_text(
            _BENCH_SCRIPT.replace(
                "__CFG__",
                json.dumps(
                    {
                        "onnx": str(onnx_path),
                        "imgsz": int(imgsz),
                        "runs": int(runs),
                        "warmup": int(warmup),
                    },
                    ensure_ascii=False,
                ),
            ),
            encoding="utf-8",
        )
        written.append(str(bench_script))
        bench_outcome = runner.run(
            runner.RunSpec(
                argv=[__import__("sys").executable, str(bench_script)],
                cwd=out,
                timeout_s=600.0,
                label="bench-onnx",
            ),
            on_line=on_line,
        )
        bench = _parse_last_json(bench_outcome.stdout)
        if bench.get("ok"):
            bench_path = out / "bench.json"
            bench_path.write_text(json.dumps(bench, ensure_ascii=False, indent=2), encoding="utf-8")
            written.append(str(bench_path))
        else:
            bench = {"ok": False, "error": bench.get("error") or bench_outcome.tail(800)}

    verdict = "unknown"
    notes: list[str] = []
    if bench.get("ok") and latency_target_ms:
        p95 = float(bench.get("p95_ms", 0.0))
        verdict = "pass" if p95 <= float(latency_target_ms) else "fail"
        notes.append(f"p95 {p95:.2f}ms vs 目标 {float(latency_target_ms):.2f}ms → {verdict}")
    elif bench.get("ok"):
        notes.append("没有给 latency_target_ms，只报数不判定")
    else:
        notes.append("没有跑延迟测试（缺 onnxruntime 或 bench 失败）")

    summary_path = out / "deploy.json"
    payload = {
        # 只导出成功不等于落地：没有拿到可用的延迟/精度证据时不要报 ok=True，
        # 否则工具层会把"导出成功"当成"部署通过"（code review B）。
        "ok": bool(export_payload.get("ok")) and verdict != "fail" and bench.get("ok", False),
        "weights": str(weights_path),
        "onnx": str(onnx_path),
        "export": export_payload,
        "bench": bench,
        "verdict": verdict,
        "latency_target_ms": latency_target_ms,
        "notes": notes,
        "written": written,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(str(summary_path))
    payload["written"] = written
    return payload
