"""演示资源：用标准库生成真实 PNG，让离线模式一眼就能看到"视觉"。

两个目的：

1. GUI 的 Demo 模式不依赖网络与 API Key，也能跑出工具卡、证据图、指标与判据；
2. 给 `image_probe` 之类的能力层提供一份**确定可复现**的输入。

生成的是灰度棋盘格（模拟标定靶）与渐变图（模拟普通采集帧），文件很小。
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import Any

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def encode_png(width: int, height: int, rows: list[bytes]) -> bytes:
    """编码 8bit RGB PNG。``rows`` 必须是 height 条、每条 width*3 字节。"""
    if len(rows) != height:
        raise ValueError(f"rows 数量应为 {height}，实际 {len(rows)}")
    raw = bytearray()
    for row in rows:
        if len(row) != width * 3:
            raise ValueError(f"每行应为 {width * 3} 字节，实际 {len(row)}")
        raw.append(0)  # filter type 0（None）
        raw.extend(row)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8bit truecolor
    return (
        PNG_SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + _chunk(b"IEND", b"")
    )


def _checker_rows(width: int, height: int, cell: int = 20) -> list[bytes]:
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            on = ((x // cell) + (y // cell)) % 2 == 0
            value = 232 if on else 32
            row += bytes((value, value, value))
        rows.append(bytes(row))
    return rows


def _gradient_rows(width: int, height: int) -> list[bytes]:
    rows: list[bytes] = []
    for y in range(height):
        row = bytearray()
        for x in range(width):
            row += bytes(
                (
                    28 + (x * 180) // max(1, width - 1),
                    60 + (y * 150) // max(1, height - 1),
                    120 + ((x + y) * 80) // max(1, width + height - 2),
                )
            )
        rows.append(bytes(row))
    return rows


def write_demo_pngs(directory: Path | str) -> dict[str, Path]:
    """在 ``directory`` 下写出三张演示图（已存在则跳过，保持幂等）。"""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    plan: dict[str, tuple[int, int, list[bytes]]] = {
        "calib_left.png": (320, 240, _checker_rows(320, 240)),
        "calib_right.png": (320, 240, _checker_rows(320, 240, cell=16)),
        "frame_raw.png": (240, 160, _gradient_rows(240, 160)),
    }
    written: dict[str, Path] = {}
    for name, (width, height, rows) in plan.items():
        path = target / name
        if not path.exists():
            path.write_bytes(encode_png(width, height, rows))
        written[name] = path
    return written


def ensure_demo_assets(directory: Path | str) -> dict[str, Any]:
    """返回 GUI / Demo 模式需要的路径信息。

    ``directory`` 就是 demo 资源目录本身（调用方传 ``Settings.demo_dir``，
    即 ``<workspace>/demo``）；这里不再追加一层 ``demo``，否则图会写到
    ``<workspace>/demo/demo/`` 下，而工具按 ``demo/xxx.png`` 读不到。
    """
    directory = Path(directory)
    files = write_demo_pngs(directory)
    dataset_dir = directory / "dataset"
    dataset_written: dict[str, Any] = {}
    dataset_error: str | None = None
    try:
        from .vision.synth import write_synthetic_dataset

        dataset_written = write_synthetic_dataset(dataset_dir, images=12, seed=7, plant_issues=True)
    except Exception as exc:  # 演示数据生成不应阻断启动，但必须能被 verify 看见
        dataset_error = f"{type(exc).__name__}: {exc}"
        dataset_written = {"error": dataset_error}
    return {
        "dir": directory,
        "relative_dir": "demo",
        "files": sorted(name for name in files),
        "primary": "demo/calib_left.png",
        "secondary": "demo/calib_right.png",
        "frame": "demo/frame_raw.png",
        "dataset": "demo/dataset",
        "dataset_files": sorted(dataset_written.get("written", [])),
        "dataset_planted": sorted(dataset_written.get("planted", [])),
        "dataset_error": dataset_error,
    }
