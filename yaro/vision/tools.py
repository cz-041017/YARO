"""把视觉开发链路包装成 agent 工具（异步 + 结构化证据）。

工具是模型能"动手"的唯一方式，所以这里的原则是：

- 返回值永远是 ``ToolResult``：摘要 + 表格证据 + 指标 + 下一步提示；
- 失败要给出**可执行的下一步**，而不是抛栈；
- 渲染出来的证据图会登记进内容寻址库（CAS），GUI 直接显示缩略图；
- 所有写操作都要显式参数（``apply``），默认只读。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..capabilities.builtin import resolve_in_root
from ..contracts import ArtifactRef, Evidence, ToolResult, ToolSpec
from ..memory.artifacts import ArtifactStore
from . import annotate, dataset as ds, plan as planner
from . import deploy_run as deployer
from . import prelabel as prelabeler
from . import train_run as trainer
from . import tune_run as tuner

_MAX_ROWS = 24


def _table(rows: list[dict[str, Any]], caption: str, limit: int = _MAX_ROWS) -> Evidence:
    return Evidence(kind="table", payload=rows[:limit], caption=caption)


def _scan_table(stats: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"指标": "图像数", "值": stats["images"]},
        {"指标": "有标注", "值": stats["labeled"]},
        {"指标": "无标注", "值": stats["unlabeled"]},
        {"指标": "标注框", "值": stats["boxes"]},
        {"指标": "空标注", "值": stats["empty_labels"]},
    ]


class DatasetScanTool:
    spec = ToolSpec(
        name="dataset_scan",
        description="扫描 YOLO/COCO 数据集：图像数、标注数、类别分布、缺标情况。任何数据任务的第一步。",
        parameters={
            "type": "object",
            "properties": {"dir": {"type": "string", "description": "数据集目录（相对工作区），如 demo/dataset"}},
            "required": ["dir"],
            "additionalProperties": False,
        },
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    async def run(self, dir: str) -> ToolResult:  # noqa: A002 - 与 tool schema 对齐
        target = resolve_in_root(self.root, dir)
        if not target.is_dir():
            return ToolResult(ok=False, summary=f"目录不存在：{dir}", hint="先 list_files 确认数据集路径")
        scan = ds.scan_dataset(target)
        stats = scan["stats"]
        if scan["format"] == "unknown":
            return ToolResult(
                ok=False,
                summary=f"{dir} 不是可识别的数据集",
                evidence=[_table([{"问题": problem} for problem in scan.get("problems", [])], "扫描问题")],
                hint="目录里应有 images/ 与 labels/（YOLO），或 annotations/instances.json（COCO）",
            )
        per_class = [{"类别": name, "框数": count} for name, count in stats["per_class"].items()]
        return ToolResult(
            ok=True,
            summary=f"{dir}：{scan['format']} · {stats['images']} 图 / {stats['boxes']} 框 / {len(scan['classes'])} 类",
            evidence=[_table(_scan_table(stats), "数据集概览"), _table(per_class, "类别分布")] if per_class else [_table(_scan_table(stats), "数据集概览")],
            metrics={
                "images": float(stats["images"]),
                "boxes": float(stats["boxes"]),
                "unlabeled": float(stats["unlabeled"]),
                "classes": float(len(scan["classes"])),
            },
            hint="下一步：dataset_audit 找脏数据；dataset_split 做不泄漏切分；dataset_render 看图确认标注质量",
        )


class DatasetAuditTool:
    spec = ToolSpec(
        name="dataset_audit",
        description="数据集体检：损坏图、缺标/孤儿标注、越界框、重复图、类别失衡、EXIF 方向。只读。",
        parameters={
            "type": "object",
            "properties": {"dir": {"type": "string", "description": "数据集目录（相对工作区）"}},
            "required": ["dir"],
            "additionalProperties": False,
        },
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    async def run(self, dir: str) -> ToolResult:  # noqa: A002
        target = resolve_in_root(self.root, dir)
        if not target.is_dir():
            return ToolResult(ok=False, summary=f"目录不存在：{dir}", hint="先 dataset_scan 确认路径")
        report = ds.audit_dataset(target)
        findings = report["findings"]
        summary = report["summary"]
        rows = [
            {"级别": item["severity"], "问题": item["code"], "说明": item["detail"], "位置": item["target"]}
            for item in findings
        ]
        by_code: dict[str, int] = {}
        for item in findings:
            by_code[item["code"]] = by_code.get(item["code"], 0) + 1
        ok = summary["errors"] == 0
        hint = None
        if summary["errors"]:
            hint = "先修 error（损坏图/越界框/坏标注行），再跑一次 audit 确认归零；warn 可带病上线但要在模型卡里记录"
        elif summary["warnings"]:
            hint = "无 error，但有 warn：优先处理 duplicate_image（会泄漏到验证集）与 missing_label（会被当负样本）"
        else:
            hint = "数据体检通过：可以进 dataset_split → dataset_render → train_plan"
        return ToolResult(
            ok=ok,
            summary=f"{dir} 体检：{summary['errors']} error / {summary['warnings']} warn / {summary['infos']} info",
            evidence=[_table(rows, "问题清单"), _table([{"问题类型": code, "次数": count} for code, count in sorted(by_code.items())], "按类型统计")],
            metrics={
                "errors": float(summary["errors"]),
                "warnings": float(summary["warnings"]),
                "images": float(report["scan"]["stats"]["images"]),
            },
            hint=hint,
        )


class DatasetSplitTool:
    spec = ToolSpec(
        name="dataset_split",
        description="按类别分层、带固定种子的 train/val/test 切分；默认只出方案，apply=true 才写 splits/*.txt。",
        parameters={
            "type": "object",
            "properties": {
                "dir": {"type": "string", "description": "数据集目录（相对工作区）"},
                "train": {"type": "number", "description": "训练集比例，默认 0.7"},
                "val": {"type": "number", "description": "验证集比例，默认 0.2"},
                "test": {"type": "number", "description": "测试集比例，默认 0.1"},
                "seed": {"type": "integer", "description": "切分种子，默认 7（同一份数据必须可复现）"},
                "apply": {"type": "boolean", "description": "true 时把 splits/*.txt 与 yaro-data.yaml 落盘"},
            },
            "required": ["dir"],
            "additionalProperties": False,
        },
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    async def run(self, dir: str, train: float = 0.7, val: float = 0.2, test: float = 0.1, seed: int = 7, apply: bool = False) -> ToolResult:  # noqa: A002
        target = resolve_in_root(self.root, dir)
        if not target.is_dir():
            return ToolResult(ok=False, summary=f"目录不存在：{dir}", hint="先 dataset_scan")
        plan = ds.plan_split(target, ratios=(train, val, test), seed=int(seed))
        if not plan["assignments"]:
            return ToolResult(ok=False, summary="数据集为空，无法切分", hint="先补数据或检查 images/ 路径")
        written: dict[str, str] = {}
        if apply:
            written = ds.write_split(target, plan)
        rows = [{"split": split, "图像数": plan["counts"][split]} for split in ds.SPLIT_NAMES]
        class_rows = [
            {"split": split, "类别": name, "框数": count}
            for split in ds.SPLIT_NAMES
            for name, count in sorted(plan["class_per_split"][split].items())
        ]
        evidence = [_table(rows, "切分结果"), _table(class_rows, "各 split 类别分布")]
        if written:
            evidence.append(_table([{"文件": name, "路径": path} for name, path in written.items()], "已写文件"))
        ok = not plan["warnings"]
        return ToolResult(
            ok=ok,
            summary=f"{dir} 切分 train/val/test = {plan['counts']['train']}/{plan['counts']['val']}/{plan['counts']['test']}（seed={seed}{'，已落盘' if apply else '，未落盘'}）",
            evidence=evidence,
            metrics={f"split_{split}": float(plan["counts"][split]) for split in ds.SPLIT_NAMES},
            hint="；".join(plan["warnings"]) if plan["warnings"] else "下一步：dataset_render 抽查每个 split 的标注，再 train_plan",
            degraded=[] if ok else ["split_ratio"],
        )


class DatasetDedupeTool:
    spec = ToolSpec(
        name="dataset_dedupe",
        description="按内容 sha256 找完全重复图（含标注）；apply=true 时移入隔离区（不删除）。",
        parameters={
            "type": "object",
            "properties": {
                "dir": {"type": "string", "description": "数据集目录（相对工作区）"},
                "apply": {"type": "boolean", "description": "true 时真正移动到 _quarantine/，默认只报告"},
            },
            "required": ["dir"],
            "additionalProperties": False,
        },
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    async def run(self, dir: str, apply: bool = False) -> ToolResult:  # noqa: A002
        target = resolve_in_root(self.root, dir)
        if not target.is_dir():
            return ToolResult(ok=False, summary=f"目录不存在：{dir}", hint="先 dataset_scan")
        report = ds.dedupe_duplicates(target, apply=bool(apply))
        groups = report["duplicate_groups"]
        rows = [{"sha256": group["sha256"][:12], "图像": ", ".join(group["images"][:4])} for group in groups]
        return ToolResult(
            ok=True,
            summary=f"{dir} 重复组 {len(groups)}，多余副本 {report['duplicate_images']} 张" + ("（已移入隔离区）" if apply else "（未移动）"),
            evidence=[_table(rows, "重复组")] if rows else [],
            metrics={"duplicate_groups": float(len(groups)), "moved": float(len(report["moved"]))},
            hint=(
                "重复样本必须清掉：同一张图同时出现在 train 与 val 会虚高指标"
                if groups
                else "没有内容级重复；近重复（裁剪/亮度变化）需要 embedding 去重，当前不做"
            ),
        )


class DatasetRenderTool:
    spec = ToolSpec(
        name="dataset_render",
        description="把标注框画回图上并拼成联系人表（证据图），用来肉眼确认标注是否错位/漏标。",
        parameters={
            "type": "object",
            "properties": {
                "dir": {"type": "string", "description": "数据集目录（相对工作区）"},
                "max_images": {"type": "integer", "description": "渲染多少张，默认 6"},
            },
            "required": ["dir"],
            "additionalProperties": False,
        },
    )

    def __init__(self, root: Path | str, store: ArtifactStore | None = None, run_id: str = "", step: int = 0) -> None:
        self.root = Path(root)
        self.store = store
        self.run_id = run_id
        self.step = step

    async def run(self, dir: str, max_images: int = 6) -> ToolResult:  # noqa: A002
        target = resolve_in_root(self.root, dir)
        if not target.is_dir():
            return ToolResult(ok=False, summary=f"目录不存在：{dir}", hint="先 dataset_scan")
        report = annotate.render_dataset(target, target / "runs" / "review", max_images=int(max_images))
        evidence = [
            _table(
                [{"渲染": name} for name in report["rendered"]] + [{"跳过": name} for name in report["skipped"]],
                f"渲染 {len(report['rendered'])} 张",
            )
        ]
        artifacts = []
        sheet = report.get("sheet")
        if sheet and self.store is not None:
            sheet_path = target / sheet
            try:
                ref = self.store.put(
                    sheet_path.read_bytes(),
                    run_id=self.run_id or "adhoc",
                    step=self.step,
                    kind="evidence",
                    media_type="image/png",
                    ext="png",
                )
                evidence.insert(0, Evidence(kind="image", payload=ref, caption=f"{dir} 标注抽查联系人表"))
                artifacts.append(ref)
            except OSError as exc:  # pragma: no cover
                report["skipped"].append(f"联系人表登记失败：{exc}")
        return ToolResult(
            ok=bool(report["rendered"]),
            summary=f"{dir} 渲染 {len(report['rendered'])} 张" + (f"，跳过 {len(report['skipped'])} 张" if report["skipped"] else ""),
            evidence=evidence,
            artifacts=artifacts,
            metrics={"rendered": float(len(report["rendered"])), "skipped": float(len(report["skipped"]))},
            hint="看联系人表：框是否贴紧目标、有没有整类漏标、坐标是否整体偏移（EXIF 方向问题会整体偏移）",
        )


class LabelConvertTool:
    spec = ToolSpec(
        name="label_convert",
        description="标注格式互转：YOLO→COCO / YOLO→VOC（导出到 exports/，不改原标注）。",
        parameters={
            "type": "object",
            "properties": {
                "dir": {"type": "string", "description": "数据集目录（相对工作区）"},
                "to": {"type": "string", "enum": ["coco", "voc"], "description": "目标格式"},
            },
            "required": ["dir", "to"],
            "additionalProperties": False,
        },
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    async def run(self, dir: str, to: str = "coco") -> ToolResult:  # noqa: A002
        target = resolve_in_root(self.root, dir)
        if not target.is_dir():
            return ToolResult(ok=False, summary=f"目录不存在：{dir}", hint="先 dataset_scan")
        if to == "coco":
            report = annotate.export_coco(target)
            rows = [{"文件": report["path"]}, {"图像": report["images"]}, {"标注": report["annotations"]}]
            count = int(report.get("images", 0))
        elif to == "voc":
            report = annotate.export_voc(target)
            files = report.get("files", 0)
            count = int(files) if isinstance(files, (int, float)) else len(files)
            rows = [{"目录": report["path"]}, {"文件数": count}]
        else:
            return ToolResult(ok=False, summary=f"不支持的目标格式：{to}", hint="可选 coco / voc")
        return ToolResult(
            ok=True,
            summary=f"{dir} → {to} 导出完成",
            evidence=[_table(rows, "导出结果")],
            metrics={"files": float(count)},
            hint="导出件在 exports/ 下，属于派生物：原标注不动，交付时给甲方这份",
        )


class TrainPlanTool:
    spec = ToolSpec(
        name="train_plan",
        description="生成可执行的训练计划（recipe + 脚本 + 命令 + 验收清单），不启动训练。",
        parameters={
            "type": "object",
            "properties": {
                "dir": {"type": "string", "description": "数据集目录（相对工作区）"},
                "backend": {"type": "string", "enum": ["torchvision", "ultralytics", "custom"], "description": "训练后端，默认 torchvision（无 AGPL 风险）"},
                "epochs": {"type": "integer", "description": "轮数，默认 30"},
                "imgsz": {"type": "integer", "description": "输入分辨率，默认 640"},
                "batch": {"type": "integer", "description": "batch size，默认 8"},
                "lr": {"type": "number", "description": "学习率，默认 0.005"},
            },
            "required": ["dir"],
            "additionalProperties": False,
        },
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    async def run(self, dir: str, backend: str = "torchvision", epochs: int = 30, imgsz: int = 640, batch: int = 8, lr: float = 0.005) -> ToolResult:  # noqa: A002
        target = resolve_in_root(self.root, dir)
        if not target.is_dir():
            return ToolResult(ok=False, summary=f"目录不存在：{dir}", hint="先 dataset_scan")
        try:
            plan = planner.build_train_plan(target, backend=backend, epochs=int(epochs), imgsz=int(imgsz), batch=int(batch), lr=float(lr))
        except ValueError as exc:
            return ToolResult(ok=False, summary=str(exc), hint="backend 可选 torchvision / ultralytics / custom")
        rows = [{"文件": path} for path in plan["written"]] + [{"命令": command} for command in plan["commands"]]
        return ToolResult(
            ok=True,
            summary=f"{dir} 训练计划已生成（{backend}，{epochs} epochs）",
            evidence=[_table(rows, "计划文件与命令"), _table([{"提示": item} for item in plan["warnings"]], "风险提示")] if plan["warnings"] else [_table(rows, "计划文件与命令")],
            metrics={"epochs": float(epochs), "imgsz": float(imgsz)},
            hint="下一步：在沙箱里执行 train.py（run_python 或宿主命令），把日志贴给 train_analyze 解析",
        )


class TrainAnalyzeTool:
    spec = ToolSpec(
        name="train_analyze",
        description="解析训练日志（JSONL / tqdm / ultralytics 表格），判断过拟合、平台期、指标缺失。",
        parameters={
            "type": "object",
            "properties": {"log": {"type": "string", "description": "训练日志文本（可多行）；或相对工作区的日志文件路径"}},
            "required": ["log"],
            "additionalProperties": False,
        },
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    async def run(self, log: str) -> ToolResult:
        text = log
        candidate = (self.root / log).resolve() if "\n" not in log and len(log) < 512 else None
        if candidate is not None and candidate.is_file():
            try:
                candidate.relative_to(self.root.resolve())
                text = candidate.read_text(encoding="utf-8", errors="replace")
            except ValueError:
                return ToolResult(ok=False, summary="日志路径越界", hint="日志必须在工作区内")
        history = planner.parse_train_log(text)
        findings = planner.analyze_training(history)
        rows = [
            {"epoch": row.get("epoch"), "loss": row.get("loss"), "mAP50": row.get("map50"), "mAP50-95": row.get("map50_95")}
            for row in history["epochs"][-_MAX_ROWS:]
        ]
        return ToolResult(
            ok=bool(history["count"]),
            summary=f"解析到 {history['count']} 轮指标，最佳 mAP50={history['best_map50']:.4f}",
            evidence=[_table(rows, "训练曲线（末尾）"), _table([{"判断": item["code"], "说明": item["detail"]} for item in findings], "诊断")] if rows else [],
            metrics={"epochs": float(history["count"]), "best_map50": float(history["best_map50"])},
            hint=findings[0]["detail"] if findings else "曲线正常：下一步固定评估集做同预算对比，再进 deploy_plan",
        )


class DeployPlanTool:
    spec = ToolSpec(
        name="deploy_plan",
        description="生成部署计划：ONNX 导出、延迟基准脚本、Dockerfile、模型卡与验收命令。",
        parameters={
            "type": "object",
            "properties": {
                "dir": {"type": "string", "description": "数据集/项目目录（相对工作区）"},
                "weights": {"type": "string", "description": "训练权重路径，默认 runs/plan/last.pth"},
                "backends": {"type": "array", "items": {"type": "string", "enum": ["onnx", "tensorrt", "openvino"]}, "description": "部署后端，默认 [onnx]"},
                "imgsz": {"type": "integer", "description": "推理分辨率，默认 640"},
                "latency_target_ms": {"type": "number", "description": "p95 延迟目标，默认 33ms"},
            },
            "required": ["dir"],
            "additionalProperties": False,
        },
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    async def run(self, dir: str, weights: str = "runs/plan/last.pth", backends: list[str] | None = None, imgsz: int = 640, latency_target_ms: float = 33.0) -> ToolResult:  # noqa: A002
        target = resolve_in_root(self.root, dir)
        if not target.is_dir():
            return ToolResult(ok=False, summary=f"目录不存在：{dir}", hint="先 dataset_scan")
        plan = planner.build_deploy_plan(target, weights=weights, imgsz=int(imgsz), backends=backends or ["onnx"], latency_target_ms=float(latency_target_ms))
        rows = [{"文件": path} for path in plan["written"]] + [{"命令": command} for command in plan["commands"]]
        return ToolResult(
            ok=True,
            summary=f"{dir} 部署计划已生成（{', '.join(plan['backends'])}，p95 ≤ {latency_target_ms}ms）",
            evidence=[_table(rows, "部署文件与命令")],
            metrics={"latency_target_ms": float(latency_target_ms)},
            hint="部署的最小证据是 p95 延迟 + 量化后精度回归；只导出 ONNX 不算落地",
        )


class TunePlanTool:
    spec = ToolSpec(
        name="tune_plan",
        description="生成调优试验计划：把参数空间展开成 trials 表（固定种子，可重放）。",
        parameters={
            "type": "object",
            "properties": {
                "space": {"type": "object", "description": "参数空间，形如 {\"lr\":[0.001,0.005],\"imgsz\":[512,640]}"},
                "trials": {"type": "integer", "description": "最多多少个 trial，默认 8"},
                "mode": {"type": "string", "enum": ["grid", "random"], "description": "搜索模式，默认 grid"},
                "seed": {"type": "integer", "description": "随机种子，默认 7"},
            },
            "required": ["space"],
            "additionalProperties": False,
        },
    )

    async def run(self, space: dict[str, Any], trials: int = 8, mode: str = "grid", seed: int = 7) -> ToolResult:
        try:
            combos = planner.plan_trials(space, trials=int(trials), seed=int(seed), mode=mode)
        except ValueError as exc:
            return ToolResult(ok=False, summary=str(exc), hint="mode 可选 grid / random")
        rows = [{"#": index + 1, **combo} for index, combo in enumerate(combos)]
        return ToolResult(
            ok=bool(combos),
            summary=f"计划 {len(combos)} 个 trial（mode={mode}，seed={seed}）",
            evidence=[_table(rows, "试验计划")],
            metrics={"trials": float(len(combos))},
            hint="每个 trial 用同预算训练脚本跑，指标交给 tune_report 汇总；差异小于噪声阈值就不要继续堆算力",
        )


class PrelabelRunTool:
    spec = ToolSpec(
        name="prelabel_run",
        description="真跑预标注：用 torchvision/ONNX/轮廓法在未标注图上产生 proposals（不覆盖人工标注），可选应用为标签。",
        parameters={
            "type": "object",
            "properties": {
                "dir": {"type": "string", "description": "数据集目录（相对工作区）"},
                "backend": {"type": "string", "enum": ["auto", "torchvision", "onnx", "contour"], "description": "预标注后端，auto 自动挑可用最强者"},
                "conf": {"type": "number", "description": "置信度阈值，默认 0.35"},
                "limit": {"type": "integer", "description": "最多处理多少张（先小批验证）"},
                "apply": {"type": "boolean", "description": "true 才把 proposals 写入 labels/（会保留已有标注）"},
            },
            "required": ["dir"],
            "additionalProperties": False,
        },
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    async def run(self, dir: str, backend: str = "auto", conf: float = 0.35, limit: int | None = None, apply: bool = False) -> ToolResult:  # noqa: A002
        target = resolve_in_root(self.root, dir)
        if not target.is_dir():
            return ToolResult(ok=False, summary=f"目录不存在：{dir}", hint="先 dataset_scan")
        try:
            report = prelabeler.prelabel_dataset(
                target, backend=backend, conf=float(conf), limit=int(limit) if limit else None
            )
        except Exception as exc:  # 预标注后端可能缺依赖
            return ToolResult(ok=False, summary=f"预标注失败：{exc}", hint="先看 capabilities；或换 contour 后端")
        rows = [
            {"项": "后端", "值": report["backend"]},
            {"项": "处理图像", "值": report["processed"]},
            {"项": "产生 proposals", "值": report["proposals"]},
            {"项": "跳过", "值": len(report.get("skipped", []))},
            {"项": "产物目录", "值": report["output"]},
        ]
        evidence = [_table(rows, "预标注结果")]
        applied = None
        if apply and report["proposals"]:
            applied = prelabeler.apply_proposals(target)
            evidence.append(_table([{"项": key, "值": value} for key, value in applied.items()], "已写入标注"))
        queue = prelabeler.review_queue(target)
        return ToolResult(
            ok=bool(report["proposals"]),
            summary=f"{dir} 预标注完成：{report['proposals']} 个框（后端 {report['backend']}）",
            evidence=evidence,
            metrics={"proposals": float(report["proposals"]), "processed": float(report["processed"])},
            degraded=list(report.get("warnings", []))[:3],
            hint=(
                f"待人工复核 {queue['count']} 张：建议 dataset_render 抽查后再 apply"
                if not apply
                else "已写入 labels/；接下来 train_run（需要 CUDA 机器）"
            ),
        )


class TrainRunTool:
    spec = ToolSpec(
        name="train_run",
        description="真训练：在本机（默认要求 CUDA）训练一个小检测模型，产出 best.pt + metrics.jsonl + 诊断。",
        parameters={
            "type": "object",
            "properties": {
                "dir": {"type": "string", "description": "数据集目录（相对工作区）"},
                "model": {"type": "string", "enum": list(trainer.TRAIN_MODELS), "description": "学生模型，默认最小最快的 mnv3"},
                "epochs": {"type": "integer", "description": "轮数，默认 12"},
                "imgsz": {"type": "integer", "description": "训练分辨率，默认 512"},
                "batch": {"type": "integer", "description": "batch size，默认 2"},
                "lr": {"type": "number", "description": "学习率，默认 0.005"},
                "device": {"type": "string", "description": "auto|cuda|cuda:0|cpu；默认 auto（无 GPU 会被拒绝）"},
                "max_minutes": {"type": "number", "description": "训练预算（分钟），默认 60"},
            },
            "required": ["dir"],
            "additionalProperties": False,
        },
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    async def run(self, dir: str, model: str = trainer.DEFAULT_MODEL, epochs: int = 12, imgsz: int = 512, batch: int = 2, lr: float = 0.005, device: str = "auto", max_minutes: float = 60.0) -> ToolResult:  # noqa: A002
        target = resolve_in_root(self.root, dir)
        if not target.is_dir():
            return ToolResult(ok=False, summary=f"目录不存在：{dir}", hint="先 dataset_scan / dataset_audit")
        config = trainer.TrainConfig(
            model=model, epochs=int(epochs), imgsz=int(imgsz), batch=int(batch), lr=float(lr),
            device=device, max_minutes=float(max_minutes),
        )
        result = trainer.train_dataset(target, config=config)
        if not result.ok:
            return ToolResult(
                ok=False,
                summary=f"训练未完成（{result.stopped_reason or 'error'}）：{result.error[:400]}",
                evidence=[_table([{"诊断": item["code"], "说明": item["detail"]} for item in result.findings], "环境/数据诊断")] if result.findings else [],
                degraded=list(result.degradations),
                hint="先看 capabilities 与 CONTRACT；无 CUDA 时训练默认被拒绝，可用 device=cpu 冒烟",
            )
        rows = [
            {"项": "模型", "值": result.config["model"]},
            {"项": "轮数", "值": result.epochs_done},
            {"项": "最佳 mAP50", "值": round(result.best_map50, 4)},
            {"项": "耗时(s)", "值": result.duration_s},
            {"项": "best.pt", "值": result.best_checkpoint},
            {"项": "metrics", "值": result.metrics_path},
        ]
        return ToolResult(
            ok=True,
            summary=f"{dir} 训练完成：{result.epochs_done} 轮，mAP50={result.best_map50:.4f}",
            evidence=[
                _table(rows, "训练结果"),
                _table([{"诊断": item["code"], "说明": item["detail"]} for item in result.findings], "曲线诊断"),
            ],
            metrics={"map50": float(result.best_map50), "epochs": float(result.epochs_done)},
            degraded=list(result.degradations),
            hint="下一步 tune_run 做同预算对比；差异小于噪声阈值就不要继续堆算力",
        )


class TuneRunTool:
    spec = ToolSpec(
        name="tune_run",
        description="真调优：按参数空间真的跑多组训练，汇总 trials 并按噪声底线选最好的一组。",
        parameters={
            "type": "object",
            "properties": {
                "dir": {"type": "string", "description": "数据集目录（相对工作区）"},
                "trials": {"type": "integer", "description": "跑几组，默认 4"},
                "mode": {"type": "string", "enum": ["grid", "random"], "description": "搜索模式"},
                "max_minutes_total": {"type": "number", "description": "总体时间预算（分钟）"},
                "space": {"type": "object", "description": "参数空间（默认 epochs/lr/imgsz/batch）"},
            },
            "required": ["dir"],
            "additionalProperties": False,
        },
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    async def run(self, dir: str, trials: int = 4, mode: str = "grid", max_minutes_total: float | None = None, space: dict[str, Any] | None = None) -> ToolResult:  # noqa: A002
        target = resolve_in_root(self.root, dir)
        if not target.is_dir():
            return ToolResult(ok=False, summary=f"目录不存在：{dir}", hint="先 dataset_scan")
        result = tuner.run_trials(
            target, space=space, trials=int(trials), mode=mode,
            max_minutes_total=float(max_minutes_total) if max_minutes_total else None,
        )
        if not result["ok"]:
            return ToolResult(ok=False, summary=f"调优未完成：{result.get('error', '')}", hint=result.get("hint", ""))
        rows = [
            {"#": row["trial"], "mAP50": round(row["map50"], 4), "耗时(s)": row["duration_s"], "停止原因": row["stopped_reason"]}
            for row in result["rows"]
        ]
        best = result["best"]
        return ToolResult(
            ok=True,
            summary=f"跑完 {len(result['rows'])} 组，最佳 mAP50={float(best.get('map50', 0.0)):.4f}",
            evidence=[
                _table(rows, "trials"),
                _table([{"结论": item} for item in result["notes"]], "噪声判定"),
            ],
            metrics={"trials": float(len(result["rows"])), "best_map50": float(best.get("map50", 0.0))},
            hint="用 best.json 里的配置重训一次大的，再进 deploy_run",
        )


class DeployRunTool:
    spec = ToolSpec(
        name="deploy_run",
        description="真部署：把 best.pt 真的导出 ONNX，并真的测 p50/p95 延迟（有目标延迟才判定 pass/fail）。",
        parameters={
            "type": "object",
            "properties": {
                "weights": {"type": "string", "description": "checkpoint 路径（相对工作区），如 demo/dataset/runs/train/best.pt"},
                "imgsz": {"type": "integer", "description": "导出/测速分辨率，默认 640"},
                "runs": {"type": "integer", "description": "测速次数，默认 20"},
                "latency_target_ms": {"type": "number", "description": "p95 目标（毫秒）；给了才判定"},
            },
            "required": ["weights"],
            "additionalProperties": False,
        },
    )

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    async def run(self, weights: str, imgsz: int = 640, runs: int = 20, latency_target_ms: float | None = None) -> ToolResult:
        target = resolve_in_root(self.root, weights)
        result = deployer.deploy_run(
            weights=target, imgsz=int(imgsz), runs=int(runs),
            latency_target_ms=float(latency_target_ms) if latency_target_ms else None,
        )
        if not result.get("ok"):
            return ToolResult(ok=False, summary=f"部署未完成：{result.get('error', '')}", hint=result.get("hint", ""))
        bench = result.get("bench") or {}
        rows = [{"项": "ONNX", "值": result["onnx"]}]
        if bench.get("ok"):
            rows += [
                {"项": "p50(ms)", "值": bench["p50_ms"]},
                {"项": "p95(ms)", "值": bench["p95_ms"]},
                {"项": "providers", "值": ", ".join(bench.get("providers", []))},
                {"项": "判定", "值": result["verdict"]},
            ]
        evidence = [_table(rows, "部署结果")] + ([_table([{"说明": item} for item in result["notes"]], "结论")] if result.get("notes") else [])
        ok = result["verdict"] != "fail"
        return ToolResult(
            ok=ok,
            summary=f"ONNX 导出成功{p50_text(bench)}",
            evidence=evidence,
            metrics={key: float(bench[key]) for key in ("p50_ms", "p95_ms") if key in bench},
            hint="只导出不算落地：还要量化后精度回归 + 目标机实机延迟",
        )


def p50_text(bench: dict[str, Any]) -> str:
    if not bench or not bench.get("ok"):
        return "（未测延迟：缺 onnxruntime）"
    return f"，p50={bench['p50_ms']}ms / p95={bench['p95_ms']}ms"


def build_vision_tools(root: Path | str, store: ArtifactStore | None, *, run_id: str = "", step: int = 0) -> list[Any]:
    """构造视觉链路工具集（供 build_default_tools 懒加载）。"""
    return [
        DatasetScanTool(root),
        DatasetAuditTool(root),
        DatasetSplitTool(root),
        DatasetDedupeTool(root),
        DatasetRenderTool(root, store, run_id, step),
        LabelConvertTool(root),
        PrelabelRunTool(root),
        TrainPlanTool(root),
        TrainRunTool(root),
        TrainAnalyzeTool(root),
        TunePlanTool(),
        TuneRunTool(root),
        DeployPlanTool(root),
        DeployRunTool(root),
    ]
