"""薄循环：一个 while、四个停条件、五道预算闸。

刻意不引入编排框架（见 ADR-0002）：

- 消息数组完全由我们控制 → reasoning_content 可以原样回传
- 状态全部落在 ``SessionStore`` 的事件链里 → 任意 turn 可回滚
- 分数只由工具返回的 metrics 进入，模型自报的内容不会被当成指标

一轮（turn）的定义：一次模型调用 + 该次调用要求的所有工具执行。
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..capabilities.evidence import estimate_messages_tokens, render_vision_parts
from ..capabilities.registry import ToolRegistry
from ..contracts import ArtifactRef, ContractError, Evidence, Usage, estimate_tokens
from ..egress import DERIVED, RAW, EgressPolicy
from ..llm.deepseek import DeepSeek
from ..memory.artifacts import ArtifactStore
from ..memory.session import SessionStore

_COMPACTION_PROMPT = """你在为一次机器视觉算法开发会话做上下文压缩。

要求：
1. 只保留后续工作必需的事实：任务目标、已经验证过/已否定的假设、确切的数字与阈值、
   产物地址（artifact:// 开头的 URI）、失败原因、下一步待办。
2. 不要复述工具输出的原文，不要保留推理过程。
3. 用紧凑的项目符号，不超过 400 字。
"""


@dataclass
class Budget:
    """所有闸门都是"便宜的 caps"。不要过度工程化卡死检测，但必须有上限。"""

    max_turns: int = 50
    max_cost_usd: float = 0.5
    max_wall_s: float = 7200.0
    max_consecutive_tool_errors: int = 3
    max_identical_calls: int = 2
    compact_at_tokens: int = 200_000
    keep_tail_ratio: float = 0.35
    max_vision_images: int = 2


@dataclass
class TurnOutcome:
    turn: int
    stop_reason: str = ""
    content: str = ""
    tool_names: list[str] = field(default_factory=list)


def evidence_to_dict(evidence: Evidence) -> dict[str, Any]:
    """把证据压成可 JSON 序列化的形式，供 GUI / 审计读取。

    图片只保留 ``ArtifactRef`` 的形状（sha256 + media_type），字节永远不入事件；
    表格最多保留 20 行——事件是账本，不是数据仓库。
    """
    payload = evidence.payload
    if isinstance(payload, ArtifactRef):
        form: Any = payload.to_dict()
    elif isinstance(payload, (str, int, float, bool)) or payload is None:
        form = payload
    elif isinstance(payload, list):
        form = [row if isinstance(row, (str, int, float)) else dict(row) for row in payload[:20]]
    elif isinstance(payload, dict):
        form = {key: payload[key] for key in list(payload)[:40]}
    else:
        form = str(payload)[:1200]
    return {"kind": evidence.kind, "caption": evidence.caption, "payload": form}


@dataclass
class AgentResult:
    run_id: str
    stop_reason: str
    turns: int
    usage: Usage
    cost_usd: float
    content: str
    messages: list[dict[str, Any]] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        *,
        llm: DeepSeek,
        registry: ToolRegistry,
        session: SessionStore,
        artifacts: ArtifactStore,
        budget: Budget | None = None,
        branch: str = "main",
        run_id: str | None = None,
        model_plan: str | None = None,
        effort: str | None = None,
        peak_billing: bool = False,
        system: str = "",
        stop_check: Callable[[], bool] | None = None,
        egress: EgressPolicy | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.session = session
        self.artifacts = artifacts
        self.budget = budget or Budget()
        self.branch = branch
        self.run_id = run_id or session.session_id
        self.model_plan = model_plan
        self.effort = effort
        self.peak_billing = peak_billing
        # CONTRACT.md 的原文。它不进会话（replay 不需要它），只作为每次请求的稳定前缀。
        self.system = system
        # 协作式取消：GUI 点"停止"只置一个 Event，循环在下一个轮边界停下（不打断进行中的调用）
        self.stop_check = stop_check
        # 出网策略：默认 None = 不做任何处理（仅内部测试/演示用）。
        # 生产路径必须传 EgressPolicy，否则原图可能被送到模型服务。
        self.egress = egress

        self.turn_index = 0
        self.usage = Usage()
        self.started_at = time.monotonic()
        self._call_counts: dict[str, int] = {}
        self._consecutive_errors = 0
        self._last_tool_images: list[dict[str, Any]] = []
        self._content = ""

    # ------------------------------------------------------------------ #
    # 预算
    # ------------------------------------------------------------------ #

    @property
    def cost_usd(self) -> float:
        return self.usage.cost_usd(self.llm.model, peak=self.peak_billing)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started_at

    def _check_budget(self) -> str:
        """返回停止原因；空串表示可以继续。"""
        if self.turn_index >= self.budget.max_turns:
            return "max_turns"
        if self.cost_usd >= self.budget.max_cost_usd:
            return "max_cost"
        if self.elapsed_s >= self.budget.max_wall_s:
            return "max_wall"
        if self._consecutive_errors >= self.budget.max_consecutive_tool_errors:
            return "tool_errors"
        return ""

    def _end(self, stop_reason: str, **extra: Any) -> None:
        """写 run_end。**每条停机路径都带 usage/cost**，否则 GUI 上"被停掉的运行"会显示成本为 0。"""
        payload: dict[str, Any] = {
            "stop_reason": stop_reason,
            "turn": self.turn_index,
            "usage": self.usage.to_dict(),
            "cost_usd": round(self.cost_usd, 6),
        }
        payload.update(extra)
        self.session.append("run_end", payload, branch=self.branch)

    def _skip_remaining(self, calls: list[dict[str, Any]], done_ids: list[str]) -> None:
        """把未执行的 tool_call 补成 skipped 的 tool 事件。

        会话里凡是出现带 ``tool_calls`` 的 assistant 消息，后面就必须有对应
        ``tool_call_id`` 的 tool 消息，否则下一轮请求会被服务端 400 拒绝。
        """
        done = set(done_ids)
        for call in calls:
            call_id = str(call.get("id", ""))
            if call_id in done:
                continue
            name = (call.get("function") or {}).get("name", "")
            self.session.append(
                "tool",
                {
                    "tool_call_id": call_id,
                    "name": name,
                    "content": "SKIPPED: 因停机（stuck/预算）未执行",
                    "summary": "未执行",
                    "ok": False,
                    "metrics": {},
                    "evidence": [],
                    "evidence_kinds": [],
                    "artifacts": [],
                    "images": [],
                    "degraded": ["skipped"],
                    "hint": "运行被停止，这一批工具调用没有执行；如需继续请换一个会话或减少重复调用。",
                },
                branch=self.branch,
            )

    # ------------------------------------------------------------------ #
    # 一轮
    # ------------------------------------------------------------------ #

    async def step(self) -> TurnOutcome:
        stop = self._check_budget()
        if stop:
            self._end(stop)
            return TurnOutcome(turn=self.turn_index, stop_reason=stop)
        if self.stop_check is not None and self.stop_check():
            self._end("stopped", detail="用户手动停止（在轮边界生效）")
            return TurnOutcome(turn=self.turn_index, stop_reason="stopped")

        await self._maybe_compact()
        messages = self._request_messages()

        turn = await self.llm.complete(
            messages,
            self.registry.schemas(),
            effort=self.effort,
            max_tokens=None,
        )
        self.usage.add(turn.usage)
        message = turn.to_message()
        self.session.append("assistant", message, branch=self.branch)
        self._content = turn.content
        self.turn_index += 1

        if not turn.tool_calls:
            self._end("done")
            return TurnOutcome(turn=self.turn_index, stop_reason="done", content=turn.content)

        tool_names: list[str] = []
        images: list[dict[str, Any]] = []
        done_ids: list[str] = []
        for call in turn.tool_calls:
            name = call["function"]["name"]
            arguments = call["function"].get("arguments") or "{}"
            tool_names.append(name)
            signature = self._signature(name, arguments)
            self._call_counts[signature] = self._call_counts.get(signature, 0) + 1
            if self._call_counts[signature] > self.budget.max_identical_calls:
                # 已经在 194 行把带 tool_calls 的 assistant 写进会话了，这里直接
                # return 会留下"悬空 tool_call"：续跑时 DeepSeek 要求每个 tool_call 都有
                # 对应的 tool 结果，否则 400 且不可恢复（code review A-3）。
                # 停机前把剩余调用补成 skipped 的 tool 事件。
                self._skip_remaining(turn.tool_calls, done_ids)
                self._end("stuck", detail=f"重复调用 {name}")
                return TurnOutcome(turn=self.turn_index, stop_reason="stuck", tool_names=tool_names)

            result = await self.registry.call(name, arguments)
            if result.ok:
                self._consecutive_errors = 0
            else:
                self._consecutive_errors += 1

            payload = {
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": result.to_text(),
                "summary": result.summary,
                "ok": result.ok,
                "metrics": result.metrics,
                "evidence": [evidence_to_dict(evidence) for evidence in result.evidence],
                "evidence_kinds": [evidence.kind for evidence in result.evidence],
                "artifacts": [artifact.to_dict() for artifact in result.artifacts],
                "images": [
                    {
                        "uri": artifact.uri,
                        "sha256": artifact.sha256,
                        "media_type": artifact.media_type,
                        # raw = 用户原始输入图；derived = harness 渲染的诊断图。
                        # 出网策略按这个标签决定能否把图发给模型。
                        "sensitivity": RAW if result.sensitive else DERIVED,
                    }
                    for artifact in result.artifacts
                    if artifact.media_type.startswith("image/")
                ],
                "degraded": result.degraded,
                "hint": result.hint,
            }
            self.session.append("tool", payload, branch=self.branch)
            done_ids.append(str(call.get("id", "")))
            images.extend(payload["images"])

        self._last_tool_images = images
        return TurnOutcome(turn=self.turn_index, content=turn.content, tool_names=tool_names)

    async def run(self, task: str) -> AgentResult:
        """执行一个任务直到停机。``task`` 作为本轮 user 消息写入会话。"""
        self.session.append("user", {"content": task, "run_id": self.run_id}, branch=self.branch)
        stop_reason = ""
        while not stop_reason:
            outcome = await self.step()
            stop_reason = outcome.stop_reason
        return AgentResult(
            run_id=self.run_id,
            stop_reason=stop_reason,
            turns=self.turn_index,
            usage=self.usage,
            cost_usd=self.cost_usd,
            content=self._content,
            messages=self.session.messages(self.branch),
        )

    # ------------------------------------------------------------------ #
    # 请求组装
    # ------------------------------------------------------------------ #

    def _request_messages(self) -> list[dict[str, Any]]:
        messages = self.session.messages(self.branch)
        policy = self.egress
        if policy is not None:
            # 深拷贝 + 脱敏：会话事件是账本，不能因为出网过滤被就地改写。
            messages = policy.filter_messages(messages)
        if self.system:
            # system 必须是**第一段且永不变**：它是前缀缓存的锚点，改一个字符就全量按 miss 计费。
            system = policy.redact(self.system) if policy is not None else self.system
            messages = [{"role": "system", "content": system}] + messages
        if self._last_tool_images and self.budget.max_vision_images > 0:
            allowed = self._last_tool_images
            dropped = 0
            if policy is not None:
                allowed = [
                    image
                    for image in self._last_tool_images
                    if policy.allows_images(image.get("sensitivity", DERIVED))
                ]
                dropped = len(self._last_tool_images) - len(allowed)
            parts = render_vision_parts(
                allowed[: self.budget.max_vision_images],
                self.artifacts,
                max_images=self.budget.max_vision_images,
            )
            if dropped and policy is not None:
                # 明确告诉模型"图被剥离了"，否则它会开始猜图像内容（污染后续所有判断）。
                parts = list(parts) + [
                    {
                        "type": "text",
                        "text": (
                            f"[隐私策略] egress={policy.mode}：已剥离 {dropped} 张原始证据图，未出网。"
                            "不要猜测图像内容，改用数值证据，或请人类调整出网策略。"
                        ),
                    }
                ]
            if parts:
                messages = messages + [{"role": "user", "content": parts}]
                self._last_tool_images = []  # 只注入一次，避免历史膨胀
        return messages

    @staticmethod
    def _signature(name: str, arguments: str) -> str:
        try:
            normalized = json.dumps(json.loads(arguments), sort_keys=True, ensure_ascii=False)
        except json.JSONDecodeError:
            normalized = arguments
        return f"{name}:{hashlib.sha256(normalized.encode()).hexdigest()[:16]}"

    # ------------------------------------------------------------------ #
    # 压缩
    # ------------------------------------------------------------------ #

    async def _maybe_compact(self) -> None:
        messages = self.session.messages(self.branch)
        if estimate_messages_tokens(messages) < self.budget.compact_at_tokens:
            return
        chain = self.session.lineage(self.branch)
        cut = self._pick_cut(chain)
        if cut is None:
            return
        # 隐私：摘要请求同样要出网，必须和主请求走同一条脱敏管线。
        # 之前这里直接拿 messages 去摘要，egress=text/derived 档下绝对路径、
        # 工具原文、疑似密钥会随摘要外发（code review P0-1）。
        if self.egress is not None:
            messages = self.egress.filter_messages(messages)
        summary = await self._summarize(messages)
        try:
            self.session.append_compaction(summary, cut, branch=self.branch)
        except ContractError as exc:
            # 切点与 append_compaction 的边界校验不一致时不要静默：付过费的摘要
            # 仍要落到事件里，否则上下文没降、下次还会再付一次。
            self.session.append(
                "compaction_error",
                {"summary": summary, "cut_after": cut, "error": str(exc)},
                branch=self.branch,
            )
            # 不 raise：压缩失败不该中断整个任务，下一轮跳到阈值下继续即可。
            return

    def _pick_cut(self, chain: Sequence[Any]) -> str | None:
        """选一个**安全的轮边界**切点。

        两个历史 bug（code review A-4/A-5）：
        1. ``consumed`` 以前累加整条 lineage，包含已经被折叠的前缀 → 切点落在
           已折叠区，压缩后上下文不降反升、每步再付一次摘要费；
        2. 以前允许"无 tool_calls 的 assistant"当切点，但 ``append_compaction``
           要求切点之后第一条是 user。末轮 assistant 后面跟 run_end 就会抛
           ContractError（而且已经付过摘要钱）。
        现在只从上次压缩之后开始累计，并且要求下一条事件是 user。
        """
        total = estimate_messages_tokens(self.session.messages(self.branch))
        target = total * (1.0 - self.budget.keep_tail_ratio)
        start = 0
        for index, event in enumerate(chain):
            if event.type == "compaction":
                start = index + 1
        consumed = 0
        for index in range(start, max(start, len(chain) - 1)):
            event = chain[index]
            consumed += estimate_tokens(json.dumps(event.payload, ensure_ascii=False))
            if consumed < target:
                continue
            if chain[index + 1].type != "user":
                continue
            if event.type not in ("assistant", "tool", "run_end", "compaction"):
                continue
            return event.event_id
        return None

    async def _summarize(self, messages: Sequence[dict[str, Any]]) -> str:
        transcript = []
        for message in messages[-40:]:
            content = message.get("content")
            if isinstance(content, list):
                content = "[多模态内容省略]"
            transcript.append(f"{message.get('role')}: {str(content)[:1500]}")
        prompt = _COMPACTION_PROMPT + "\n\n===== 待压缩历史 =====\n" + "\n".join(transcript)
        turn = await self.llm.complete(
            [{"role": "user", "content": prompt}],
            thinking=False,
            max_tokens=1200,
        )
        self.usage.add(turn.usage)
        summary = turn.content.strip()
        if not summary:
            raise ContractError("压缩摘要为空：拒绝静默丢失上下文，请调大 max_tokens 或缩短历史")
        return summary
