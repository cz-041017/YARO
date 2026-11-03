"""YARO —— You Only Run Once.

机器视觉算法开发 harness。基于 OpenVLHarness（MIT）二次开发。

分层（每层只依赖下层）：

    L6 接口      yaro/cli.py
    L5 策略      contract/CONTRACT.md · yaro/gates.py
    L4 编排      yaro/graph/*（LangGraph 状态图：agent 循环 + 蒸馏闭环 + 上下文经济）
    L3 能力      yaro/capabilities/* · yaro/vision/*（证据渲染与视觉链路在此层，不在模型侧）
    L2 记忆产物  yaro/memory/{session,artifacts}.py（事件溯源会话 + 内容寻址产物）
    L1 执行基座  yaro/capabilities/sandbox.py · yaro/provider/*
    L0 资源      DeepSeek API · Docker · GPU（外部）

核心不变式见 README「四个不变式」，代码中被断言执行。
"""

__version__ = "0.7.0"
__all__ = ["__version__"]
