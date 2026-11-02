"""离线演示用的假模型（Demo 模式）。

它不联网、不花钱，但**走的是与真实模型完全相同的循环**：一样的 tool_calls 协议、
一样的 reasoning_content 字段、一样的 Usage 记账、一样的事件流。所以 GUI 在
没有 API Key 的机器上也能演示"工具卡 + 证据图 + 指标 + 判据"长什么样。

剧本覆盖 YARO 的全链路骨架：
``list_files → dataset_scan → dataset_audit → dataset_render → dataset_split``
``→ train_plan → train_analyze``，最后按 CONTRACT 三段式给结论。

脚本按"已经发生过的工具轮次"推进，因此回滚（rewind）之后再跑，会从对应阶段接着走，
表现与真实会话一致。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Sequence

from ..contracts import Usage
from .deepseek import AssistantTurn

_USAGE_PLAN = [
    Usage(hit=18_240, miss=2_180, out=612, calls=1),
    Usage(hit=19_870, miss=1_240, out=548, calls=1),
    Usage(hit=21_310, miss=980, out=704, calls=1),
    Usage(hit=21_905, miss=760, out=486, calls=1),
    Usage(hit=22_640, miss=690, out=1_184, calls=1),
    Usage(hit=23_180, miss=640, out=902, calls=1),
    Usage(hit=23_760, miss=520, out=1_046, calls=1),
    Usage(hit=24_310, miss=480, out=1_262, calls=1),
]

# 演示用的训练日志：前 6 轮上升，后 4 轮几乎不动 → 平台期。
_TRAIN_LOG = "\n".join(
    json.dumps(row, ensure_ascii=False)
    for row in [
        {"epoch": 1, "loss": 2.412, "map50": 0.121, "map50_95": 0.061},
        {"epoch": 2, "loss": 1.884, "map50": 0.263, "map50_95": 0.142},
        {"epoch": 3, "loss": 1.402, "map50": 0.371, "map50_95": 0.214},
        {"epoch": 4, "loss": 1.118, "map50": 0.438, "map50_95": 0.262},
        {"epoch": 5, "loss": 0.962, "map50": 0.472, "map50_95": 0.288},
        {"epoch": 6, "loss": 0.874, "map50": 0.489, "map50_95": 0.301},
        {"epoch": 7, "loss": 0.831, "map50": 0.493, "map50_95": 0.304},
        {"epoch": 8, "loss": 0.808, "map50": 0.495, "map50_95": 0.306},
        {"epoch": 9, "loss": 0.795, "map50": 0.496, "map50_95": 0.307},
        {"epoch": 10, "loss": 0.789, "map50": 0.496, "map50_95": 0.307},
    ]
)


def _call(index: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"demo_call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


class DemoLLM:
    """确定性的剧本模型。``delay`` 只影响观感，不影响语义。"""

    model = "deepseek-flash"

    def __init__(self, *, delay_s: float = 0.35) -> None:
        self.delay_s = delay_s

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        effort: str | None = None,
        thinking: bool = True,
        max_tokens: int | None = None,
    ) -> AssistantTurn:
        rounds = sum(
            1 for message in messages if message.get("role") == "assistant" and message.get("tool_calls")
        )
        index = min(rounds, len(_USAGE_PLAN) - 1)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)

        def turn(content: str, reasoning: str, tool_calls: list[dict[str, Any]]) -> AssistantTurn:
            return AssistantTurn(
                content=content,
                reasoning_content=reasoning,
                tool_calls=tool_calls,
                usage=_USAGE_PLAN[index],
                model=self.model,
                finish_reason="tool_calls" if tool_calls else "stop",
            )

        if rounds == 0:
            return turn(
                "先看工作区里有什么，再决定从哪一步开始。",
                "任务是把演示数据集走完全链路。按 CONTRACT 的规矩不能凭聊天下结论，"
                "必须先拿现场证据。第一步用 list_files 确认数据在哪、有多大。",
                [_call(1, "list_files", {"path": "demo", "limit": 20})],
            )

        if rounds == 1:
            return turn(
                "有 demo/dataset，先扫一遍结构：多少图、多少框、类别分布。",
                "看到 dataset 目录。任何数据任务的第一步都是 dataset_scan："
                "它给出格式（YOLO/COCO）、图像数、标注数、类别分布，后面所有判据都基于这份统计。",
                [_call(2, "dataset_scan", {"dir": "demo/dataset"})],
            )

        if rounds == 2:
            return turn(
                "扫描通过，但扫描不查脏数据。跑一次体检。",
                "scan 只做结构统计，缺标、越界框、重复图、EXIF 方向这些必须靠 audit。"
                "体检结果是后续能不能训练的硬前提，不是可选项。",
                [_call(3, "dataset_audit", {"dir": "demo/dataset"})],
            )

        if rounds == 3:
            return turn(
                "体检发现 error 和 warn，先把标注画回图上看是否错位。",
                "错误分两类：结构性错误（损坏图/坏标注行/越界框）会让训练直接崩或学偏；"
                "统计性警告（缺标/重复）会虚高指标。先渲染证据图确认坐标是否整体偏移——"
                "EXIF 方向问题最典型的表现就是整体旋转 90°。",
                [_call(4, "dataset_render", {"dir": "demo/dataset", "max_images": 6})],
            )

        if rounds == 4:
            return turn(
                "坐标正常。接着做不泄漏切分，并把重复样本找出来。",
                "切分要用固定种子的分层切分，否则每跑一次指标都不可比。"
                "同时按 sha256 找完全重复图：同一张图同时进 train 与 val 是最隐蔽的指标虚高来源。",
                [
                    _call(5, "dataset_split", {"dir": "demo/dataset", "train": 0.7, "val": 0.2, "test": 0.1, "seed": 7}),
                    _call(6, "dataset_dedupe", {"dir": "demo/dataset"}),
                ],
            )

        if rounds == 5:
            return turn(
                "切分比例合理。生成训练计划（先不启动训练）。",
                "训练参数必须有依据：3 类目标、12 张图，属于极小样本，先给保守配置并写明验收清单。"
                "计划里要包含固定评估集、同预算对比、鲁棒性检查——否则训完也无法判断有没有变好。",
                [_call(7, "train_plan", {"dir": "demo/dataset", "backend": "torchvision", "epochs": 10})],
            )

        if rounds == 6:
            return turn(
                "拿一段训练日志做解析，确认诊断器能识别平台期。",
                "训练日志是唯一能证明'训练有效'的原始证据。解析器要能识别平台期/过拟合/指标缺失，"
                "而不是只看最后一轮的数字。",
                [_call(8, "train_analyze", {"log": _TRAIN_LOG})],
            )

        return AssistantTurn(
            content=(
                "**观察**：demo/dataset 为 YOLO 布局，3 类、12 张图；体检报出 error（损坏图、越界框）"
                "与 warn（缺标、重复图、孤儿标注）；标注渲染后坐标无整体偏移；"
                "切分 70/20/10 且各 split 类别覆盖完整；训练计划已生成；"
                "10 轮日志显示 mAP50 从 0.121 升到 0.489 后停在 0.496（6→10 轮仅 +0.007）。\n\n"
                "**判断**：数据侧的问题是可修的（删损坏图、修越界框、清重复、补缺标），"
                "不修就直接训练会把噪声学进模型。训练侧已经进平台期，继续加轮数收益接近噪声底线，"
                "应该先修数据再谈调参——这是小样本项目最常见的顺序错误。\n\n"
                "**下一步**：①按 audit 的 error 清单逐条修，修完重跑 audit 直到 error=0；"
                "②重复图确认后 apply 移入隔离区；③修完数据重跑 train_plan 并用同预算对比；"
                "④部署前跑 deploy_plan，用 p95 延迟与量化后精度回归做验收。\n\n"
                "本次为演示模式，未调用真实模型。"
            ),
            reasoning_content=(
                "证据齐了：扫描统计、体检结论、渲染图、切分方案、训练计划、日志诊断。"
                "结论必须按 CONTRACT 三段式给出，并且明确'先修数据再调参'的顺序，"
                "以及'平台期不要靠加轮数硬顶'——这两条是这个领域最常见的返工来源。"
            ),
            tool_calls=[],
            usage=_USAGE_PLAN[7],
            model=self.model,
            finish_reason="stop",
        )
