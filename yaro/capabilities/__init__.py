"""能力层：模型能"做"的事，全部在这一层落地。

分工：
- ``registry.py``  工具的注册与异常边界
- ``evidence.py``  把工具输出翻译成模型可读的证据（图/表/标量）
- ``calib.py``     标定闭环：求解 + 判据 + 可执行的失败解释
- ``sandbox.py``   代码执行隔离（Docker 优先，宿主本地执行需显式开启）

刻意不在这里做的事：编排、重试策略、预算控制——那些属于循环层。
"""

from __future__ import annotations

from .calib import CalibReport, CalibThresholds, assess_calibration
from .evidence import estimate_messages_tokens, render_scalar_table, render_vision_parts
from .registry import ToolRegistry
from .sandbox import DockerSandbox, ExecResult, LocalUnsafeSandbox

__all__ = [
    "CalibReport",
    "CalibThresholds",
    "DockerSandbox",
    "ExecResult",
    "LocalUnsafeSandbox",
    "ToolRegistry",
    "assess_calibration",
    "estimate_messages_tokens",
    "render_scalar_table",
    "render_vision_parts",
]
