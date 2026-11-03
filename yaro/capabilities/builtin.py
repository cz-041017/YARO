"""YARO 自带的最小能力集（零依赖）。

为什么这些工具用标准库实现：核心运行时必须"装完就能跑"。OpenCV / torch 这些重依赖
只出现在 ``yaro[cv]`` 里，由标定求解按需懒加载。GUI 的离线 Demo 也复用这一套工具，
所以界面上看到的工具卡、证据图、指标，和真实运行是同一条代码路径。

安全边界：所有路径参数都相对**工作区根目录**解析，越界直接报错，
而不是"读到了就返回"。
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

from ..contracts import ArtifactRef, ContractError, Evidence, Tool, ToolResult, ToolSpec
from ..memory.artifacts import ArtifactStore
from .calib import assess_calibration
from .sandbox import DockerSandbox

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
# 允许在缺少"可选依赖"时把视觉工具整体降级；视觉包内部的真实导入错误不再被吞。
_OPTIONAL_MODULES = frozenset({"yaro.vision", "yaro.vision.tools", "yaro.exec"})
_MEDIA_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".json": "application/json",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".npy": "application/npy",
}


# --------------------------------------------------------------------------- #
# 路径与图像头解析（纯标准库）
# --------------------------------------------------------------------------- #


def resolve_in_root(root: Path, raw: str) -> Path:
    """把工具参数解析成工作区内的绝对路径；越界抛 ContractError。"""
    candidate = Path(raw)
    resolved = (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        resolved.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise ContractError(f"路径越界：{raw!r} 不在工作区 {root} 内") from exc
    return resolved


def _png_size(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    position = 2
    length = len(data)
    while position + 9 < length:
        if data[position] != 0xFF:
            position += 1
            continue
        marker = data[position + 1]
        if marker in (0xD8, 0xD9, 0x01) or 0xD0 <= marker <= 0xD7:
            position += 2
            continue
        segment = struct.unpack(">H", data[position + 2 : position + 4])[0]
        # SOF0..SOF15（跳过 DHT=C4、JPG=C8、DAC=CC）
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[position + 5 : position + 9])
            return int(width), int(height)
        position += 2 + segment
    return None


def _gif_size(data: bytes) -> tuple[int, int] | None:
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        width, height = struct.unpack("<HH", data[6:10])
        return int(width), int(height)
    return None


def _bmp_size(data: bytes) -> tuple[int, int] | None:
    if data[:2] == b"BM" and len(data) >= 26:
        width, height = struct.unpack("<ii", data[18:26])
        return abs(int(width)), abs(int(height))
    return None


def exif_orientation(data: bytes) -> int | None:
    """读 JPEG 的 EXIF Orientation（1=正常）。线上"图全错"最常见的元凶之一。"""
    index = data.find(b"Exif\x00\x00")
    if index < 0:
        return None
    tiff = data[index + 6 :]
    if tiff[:2] == b"II":
        endian = "<"
    elif tiff[:2] == b"MM":
        endian = ">"
    else:
        return None
    if len(tiff) < 8:
        return None
    offset = struct.unpack(endian + "I", tiff[4:8])[0]
    if offset + 2 > len(tiff):
        return None
    count = struct.unpack(endian + "H", tiff[offset : offset + 2])[0]
    for index_entry in range(count):
        entry = offset + 2 + index_entry * 12
        if entry + 12 > len(tiff):
            break
        tag = struct.unpack(endian + "H", tiff[entry : entry + 2])[0]
        if tag == 0x0112:
            return int(struct.unpack(endian + "H", tiff[entry + 8 : entry + 10])[0])
    return None


def _is_complete(name: str, data: bytes) -> bool:
    """粗判文件是否被截断：镜像/拷贝中断的 PNG 照样有合法 IHDR，只解头会漏掉。"""
    if name == "PNG":
        return len(data) >= 12 and data[-8:-4] == b"IEND"
    if name == "JPEG":
        return data.rstrip(b"\x00")[-2:] == b"\xff\xd9"
    if name == "GIF":
        return data.rstrip(b"\x00")[-1:] == b";"
    if name == "BMP":
        if len(data) < 6:
            return False
        declared = struct.unpack("<I", data[2:6])[0]
        return declared == 0 or declared <= len(data)
    return True


def probe_image(path: Path) -> dict[str, Any]:
    """读取图像头，得到格式/尺寸/方向。不解码像素，因此对超大图也安全。

    除了尺寸，还会判一次"是否被截断"：截断的 PNG 依然有合法 IHDR，
    如果只看头就会把坏图当成好图放进训练集（线上最常见的脏数据之一）。
    """
    data = path.read_bytes()
    for name, parser in (("PNG", _png_size), ("JPEG", _jpeg_size), ("GIF", _gif_size), ("BMP", _bmp_size)):
        size = parser(data)
        if size is not None:
            orientation = exif_orientation(data) if name == "JPEG" else None
            truncated = not _is_complete(name, data)
            return {
                "format": name,
                "width": size[0],
                "height": size[1],
                "bytes": len(data),
                "orientation": orientation,
                "truncated": truncated,
            }
    return {
        "format": path.suffix.lstrip(".").upper() or "UNKNOWN",
        "width": 0,
        "height": 0,
        "bytes": len(data),
        "orientation": None,
        "truncated": True,
    }


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #


class ListFilesTool:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.spec = ToolSpec(
            name="list_files",
            description="列出工作区内的文件（相对路径、类型、字节数）。找数据、找出图先调它。",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "相对工作区根目录的目录，默认 ."},
                    "limit": {"type": "integer", "description": "最多返回多少条，默认 60"},
                },
                "additionalProperties": False,
            },
        )

    async def run(self, path: str = ".", limit: int = 60) -> ToolResult:
        target = resolve_in_root(self.root, path)
        if not target.exists():
            return ToolResult(
                ok=False,
                summary=f"目录不存在：{path}",
                hint="先 list_files 上一级目录，确认路径拼写；不要猜路径",
            )
        if not target.is_dir():
            return ToolResult(ok=False, summary=f"{path} 不是目录", hint="用 image_probe 探测单个文件")
        limit = max(1, min(int(limit), 200))
        files: list[Path] = [item for item in sorted(target.rglob("*")) if item.is_file()]
        rows = [
            {
                "path": str(item.relative_to(self.root)),
                "kind": item.suffix.lstrip(".") or "file",
                "bytes": item.stat().st_size,
            }
            for item in files[:limit]
        ]
        return ToolResult(
            ok=True,
            summary=f"{path} 下共 {len(files)} 个文件" + ("" if len(files) <= limit else f"（只列出前 {limit} 个）"),
            evidence=[Evidence(kind="table", payload=rows, caption="文件列表")] if rows else [],
            metrics={"files": float(len(files))},
            hint="对图像用 image_probe 看尺寸与 EXIF 方向；算指标用 run_python（Docker 沙箱）",
        )


IMAGE_PROBE_SPEC = ToolSpec(
    name="image_probe",
    description="读取图像格式/尺寸/EXIF 方向并登记为产物；不解码全图，大图也安全。",
    parameters={
        "type": "object",
        "properties": {"path": {"type": "string", "description": "相对工作区根目录的图像路径"}},
        "required": ["path"],
        "additionalProperties": False,
    },
)


class ImageProbeTool:
    """读图像头 + 把图登记进 CAS（GUI 里直接显示缩略图）。"""

    def __init__(self, root: Path | str, store: ArtifactStore, run_id: str, step: int = 0) -> None:
        self.root = Path(root)
        self.store = store
        self.run_id = run_id
        self.step = step
        self.spec = IMAGE_PROBE_SPEC

    async def run(self, path: str) -> ToolResult:
        target = resolve_in_root(self.root, path)
        if not target.is_file():
            return ToolResult(ok=False, summary=f"文件不存在：{path}", hint="先用 list_files 确认文件路径")
        if target.suffix.lower() not in _IMAGE_SUFFIXES:
            return ToolResult(ok=False, summary=f"不是支持的图像扩展名：{target.suffix}", hint="支持 png/jpg/jpeg/gif/bmp/webp")
        info = probe_image(target)
        media_type = _MEDIA_BY_SUFFIX.get(target.suffix.lower(), "application/octet-stream")
        ref = self.store.put_file(target, run_id=self.run_id, step=self.step, kind="input", media_type=media_type)
        rows = [
            {"field": "格式", "value": info["format"]},
            {"field": "尺寸", "value": f"{info['width']}×{info['height']}"},
            {"field": "像素数", "value": info["width"] * info["height"]},
            {"field": "字节", "value": info["bytes"]},
            {"field": "EXIF 方向", "value": info["orientation"] if info["orientation"] is not None else "无"},
        ]
        hint = None
        if info["orientation"] not in (None, 1):
            hint = f"EXIF Orientation={info['orientation']}：喂给模型前必须先纠正方向，否则坐标框全部错位"
        elif info["width"] and info["width"] < 64:
            hint = "分辨率过低：检查是不是缩略图或误用了 preview 目录"
        else:
            hint = "下一步：评估曝光/清晰度，或直接进入标定判据"
        return ToolResult(
            ok=True,
            summary=f"{target.name}：{info['width']}×{info['height']} {info['format']} {info['bytes']}B",
            evidence=[
                Evidence(kind="image", payload=ref, caption=f"输入图 {target.name}"),
                Evidence(kind="table", payload=rows, caption="图像属性"),
            ],
            artifacts=[ref],
            metrics={
                "width": float(info["width"]),
                "height": float(info["height"]),
                "megapixels": round(info["width"] * info["height"] / 1e6, 3),
            },
            hint=hint,
            # 这张证据图就是用户的原始输入图：默认不出网（egress=full 才允许）。
            sensitive=True,
        )


class CalibAssessTool:
    """标定判据（纯函数包装）。分数由 harness 算，模型只能读结果。"""

    spec = ToolSpec(
        name="calib_assess",
        description="按 YARO-CONTRACT 判定一次标定是否达标（RMS / 覆盖率 / 有效视角）。",
        parameters={
            "type": "object",
            "properties": {
                "rms_px": {"type": "number", "description": "总体重投影误差（像素）"},
                "coverage": {"type": "number", "description": "成功参与求解的视角占比，0–1"},
                "views": {"type": "integer", "description": "有效视角数"},
                "per_view_rms": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "可选：每个视角的重投影误差，用于找离群视角",
                },
            },
            "required": ["rms_px", "coverage", "views"],
            "additionalProperties": False,
        },
    )

    async def run(
        self,
        rms_px: float,
        coverage: float,
        views: int,
        per_view_rms: list[float] | None = None,
    ) -> ToolResult:
        report = assess_calibration(
            rms_px=float(rms_px),
            coverage=float(coverage),
            views=int(views),
            per_view_rms=per_view_rms,
        )
        result = report.to_result()
        # 判据本身作为证据补一条，GUI 里能一眼看到"为什么没过"
        if report.reasons:
            result.evidence.append(
                Evidence(kind="table", payload=[{"原因": reason} for reason in report.reasons], caption="未通过原因")
            )
        return result


class SandboxPythonTool:
    """在 Docker 沙箱里跑一段 Python。没有 Docker 时返回降级结果，而不是在宿主机上跑。"""

    def __init__(self, sandbox: DockerSandbox | None = None, *, timeout_s: float = 120.0) -> None:
        self.sandbox = sandbox or DockerSandbox(timeout_s=timeout_s)
        self.spec = ToolSpec(
            name="run_python",
            description="在隔离的 Docker 沙箱里执行 Python 代码（无网络、只读根、资源受限），用于算指标/画图。",
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "完整可执行的 Python 脚本，标准输出会被截断保留末尾"},
                    "purpose": {"type": "string", "description": "一句话说明这段代码要算什么，便于审计"},
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        )

    async def run(self, code: str, purpose: str = "") -> ToolResult:
        if self.sandbox is None or not self.sandbox.available():
            return ToolResult(
                ok=False,
                summary="docker 不可用，拒绝在宿主机执行生成的代码",
                degraded=["sandbox"],
                hint="安装并启动 Docker 后重试；或显式设置 YARO_ALLOW_UNSAFE_LOCAL=1 用本地沙箱（仅开发环境）",
                metrics={},
            )
        result = self.sandbox.run(code)
        evidence = [
            Evidence(kind="text", payload=result.stdout[-3000:] or "(无标准输出)", caption="stdout"),
        ]
        if result.stderr:
            evidence.append(Evidence(kind="text", payload=result.stderr[-1500:], caption="stderr"))
        return ToolResult(
            ok=result.ok,
            summary=(f"沙箱执行{'成功' if result.ok else '失败'}（exit={result.exit_code}，{result.duration_s:.1f}s）")
            + (f"；purpose={purpose}" if purpose else ""),
            evidence=evidence,
            metrics={"exit_code": float(result.exit_code), "duration_s": round(result.duration_s, 3)},
            degraded=(["timeout"] if result.timed_out else []),
            hint=None if result.ok else "看 stderr 定位；不要原样重试同一段代码，先缩小复现范围",
        )


def build_default_tools(
    root: Path | str,
    store: ArtifactStore,
    *,
    run_id: str,
    step: int = 0,
) -> list[Tool]:
    """GUI / Demo 使用的默认工具集。

    视觉链路工具（数据集/标注/训练/部署）按需懒加载：它们是可选能力，
    而且 `yaro.vision.tools` 反过来依赖本模块，模块顶层 import 会成环。
    """
    tools: list[Tool] = [
        ListFilesTool(root),
        ImageProbeTool(root, store, run_id, step=step),
        CalibAssessTool(),
        SandboxPythonTool(),
    ]
    try:
        from ..vision.tools import build_vision_tools
    except ModuleNotFoundError as exc:
        # 只吞"视觉包本身缺失"：以前任何 ImportError 都被吞掉，视觉层里真正的
        # 导入错误（比如 exec/__init__ 导错名字）会静默变成"能力缺失"，
        # 界面照常工作、训练功能全丢（code review B 的建议）。
        if (exc.name or "") not in _OPTIONAL_MODULES:
            raise
        return tools
    return tools + build_vision_tools(root, store, run_id=run_id, step=step)


def build_tool_catalog(root: Path | str) -> list[dict[str, Any]]:
    """给 GUI 展示的能力清单（只取 spec，不实例化持有产物库的工具，避免副作用）。"""
    specs = [
        ListFilesTool(root).spec,
        IMAGE_PROBE_SPEC,
        CalibAssessTool.spec,
        SandboxPythonTool().spec,
    ]
    try:
        from ..vision import tools as vision_tools
    except ImportError:  # pragma: no cover
        vision_tools = None  # type: ignore[assignment]
    if vision_tools is not None:
        specs.extend(
            [
                vision_tools.DatasetScanTool.spec,
                vision_tools.DatasetAuditTool.spec,
                vision_tools.DatasetSplitTool.spec,
                vision_tools.DatasetDedupeTool.spec,
                vision_tools.DatasetRenderTool.spec,
                vision_tools.LabelConvertTool.spec,
                vision_tools.PrelabelRunTool.spec,
                vision_tools.TrainPlanTool.spec,
                vision_tools.TrainRunTool.spec,
                vision_tools.TrainAnalyzeTool.spec,
                vision_tools.TunePlanTool.spec,
                vision_tools.TuneRunTool.spec,
                vision_tools.DeployPlanTool.spec,
                vision_tools.DeployRunTool.spec,
            ]
        )
    return [{"name": spec.name, "description": spec.description} for spec in specs]
