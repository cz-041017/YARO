"""受限执行层（L1）：把"计划"变成"真的跑"。

设计约束：
- **零依赖**：只用标准库 subprocess/threading/queue。
- **模板化命令**：只执行 harness 自己生成的脚本（train.py / export_onnx.py 等），
  不接受模型自由拼命令；argv 由调用方构造并校验。
- **有预算**：超时即杀进程树；输出只留尾部；失败不抛异常，转成结构化结果。
- **可回放**：返回的 stdout/stderr 由上层写入事件与产物，不在这里落盘。
"""

from .env import capabilities, python_executable, suggest_install, which
from .runner import RunOutcome, RunSpec, run

__all__ = [
    "RunOutcome",
    "RunSpec",
    "capabilities",
    "python_executable",
    "run",
    "suggest_install",
    "which",
]
