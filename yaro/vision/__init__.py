"""视觉开发能力层：数据集、标注、训练、调优、部署。

设计约束（与整个 YARO 一致）：

- **核心零依赖**：PNG 编解码、标注格式解析、数据集审计都以标准库实现，
  `pip install yaro` 后立刻能跑。重依赖（torch / onnxruntime / opencv）只在
  真正需要时懒加载，缺失就返回 ``degraded``，绝不在宿主上装依赖。
- **纯函数优先**：审计、切分、转换、渲染都是"输入路径 → 证据"，不依赖模型，
  因此可以被 eval / verify 反复重放。
- **路径安全**：所有对外入口都过 ``resolve_in_root``，越界直接 ContractError。
"""

from __future__ import annotations

from .dataset import (
    audit_dataset,
    dedupe_duplicates,
    plan_split,
    scan_dataset,
    write_manifest,
)
from .png import Image, decode_png, encode_png, render_boxes

__all__ = [
    "Image",
    "audit_dataset",
    "decode_png",
    "dedupe_duplicates",
    "encode_png",
    "plan_split",
    "render_boxes",
    "scan_dataset",
    "write_manifest",
]
