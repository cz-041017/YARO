"""最小可运行示例：把 loop / session / artifacts / registry 接起来。

两种跑法：

    python examples/quickstart.py --offline        # 不需要 API Key，演示回滚
    python examples/quickstart.py "给这次任务写个说明"   # 需要 DEEPSEEK_API_KEY

设计意图：让"接线"这件事只需要 20 行，剩下的都交给 harness。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yaro.capabilities.registry import ToolRegistry  # noqa: E402
from yaro.config import Settings, system_prompt  # noqa: E402
from yaro.contracts import Evidence, ToolResult, ToolSpec, Usage  # noqa: E402
from yaro.loop.agent import Agent, Budget  # noqa: E402
from yaro.memory.artifacts import ArtifactStore  # noqa: E402
from yaro.memory.session import SessionStore  # noqa: E402


class EchoTool:
    """最小的一个真实工具：证明工具接口长什么样。"""

    spec = ToolSpec(
        name="echo",
        description="回显文本，用于验证循环与事件链是否工作",
        parameters={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要回显的文本"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    )

    async def run(self, text: str = "") -> ToolResult:
        return ToolResult(
            ok=True,
            summary=f"回显：{text}",
            evidence=[Evidence(kind="scalar", payload=len(text), caption="字符数")],
            metrics={"chars": float(len(text))},
        )


class OfflineLLM:
    """离线假模型：第一轮调工具，第二轮结束。用于在没有 API Key 的环境里验证接线。"""

    model = "offline-fake"

    async def complete(self, messages: list[dict[str, Any]], tools: Any = None, **_: Any) -> Any:
        from yaro.llm.deepseek import AssistantTurn

        if not any(message.get("role") == "tool" for message in messages):
            return AssistantTurn(
                content="",
                reasoning_content="先调用 echo 验证工具链路",
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"text":"hello yaro"}'},
                    }
                ],
                usage=Usage(hit=800, miss=50, out=40, calls=1),
                model=self.model,
            )
        return AssistantTurn(
            content="链路正常：工具已执行，事件已落盘。",
            reasoning_content="确认工具结果后收尾",
            usage=Usage(hit=900, miss=20, out=30, calls=1),
            model=self.model,
        )


async def run_online(task: str) -> None:
    from yaro.llm.deepseek import DeepSeek

    settings = Settings.load().ensure()
    if not settings.api_key:
        raise SystemExit("缺少 API Key：运行 `yaro key set`（推荐）或设置 DEEPSEEK_API_KEY")
    # 出网策略必须显式构造并同时交给传输层与循环层：任何一层缺失都会留下泄漏面。
    policy = settings.egress_policy()
    try:
        policy.check_endpoint(settings.base_url)
    except Exception as exc:  # noqa: BLE001 - 打印人话并退出
        raise SystemExit(f"出网策略拦截：{exc}") from exc
    llm = DeepSeek(
        api_key=settings.api_key,
        base_url=settings.base_url,
        model=settings.model,
        mode="chat",
        egress=policy,
    )
    session = SessionStore.create(settings.sessions_dir)
    agent = Agent(
        llm=llm,
        registry=ToolRegistry([EchoTool()]),
        session=session,
        artifacts=ArtifactStore(settings.artifacts_dir),
        budget=Budget(),
        effort="high",
        system=system_prompt(),
        egress=policy,
    )
    result = await agent.run(task)
    print(f"stop={result.stop_reason} turns={result.turns} cost=${result.cost_usd:.4f}")
    print(result.content)


async def run_offline() -> None:
    home = ROOT / ".yaro-demo"
    session = SessionStore.create(home / "sessions")
    artifacts = ArtifactStore(home / "artifacts")
    agent = Agent(
        llm=OfflineLLM(),  # type: ignore[arg-type]
        registry=ToolRegistry([EchoTool()]),
        session=session,
        artifacts=artifacts,
    )
    result = await agent.run("演示：跑通工具链路并落盘事件")

    print(f"会话 {session.session_id}：stop={result.stop_reason} turns={result.turns}")
    print("事件链：", [event.type for event in session.all_events()])
    print("完整性：", "OK" if session.verify()["ok"] else session.verify()["problems"])

    # 核心演示：回滚 = 移动指针，事件一条都没少
    assistant_events = [event for event in session.all_events() if event.type == "assistant"]
    before = len(session.all_events())
    session.rewind(assistant_events[0].event_id)
    print(f"回滚到第一轮回答后：事件数 {before} → {len(session.all_events())}（不删数据）")
    print("回滚后的消息数：", len(session.messages("main")))
    print(f"产物库：{len(artifacts)} 个对象（去重后 {artifacts.stats()['bytes']} 字节）")


def main() -> int:
    parser = argparse.ArgumentParser(description="YARO 最小示例")
    parser.add_argument("task", nargs="?", help="任务描述（在线模式）")
    parser.add_argument("--offline", action="store_true", help="用假模型离线演示")
    args = parser.parse_args()

    if args.offline or not args.task:
        asyncio.run(run_offline())
        return 0
    asyncio.run(run_online(args.task))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
