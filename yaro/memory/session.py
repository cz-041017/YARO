"""事件溯源会话树。

把它想成 git：事件是 commit，``heads.json`` 是分支指针，``rewind``/``fork`` 只移动指针。

不变式：

1. **append-only**：事件一旦写入就不修改、不删除；回滚 = 移动 head。
2. **可验证**：每条事件带 ``prev_digest``，``digest`` 覆盖自身全部字段，链式校验。
3. **不可原地覆盖**：``rewind`` 不删任何东西，只是把分支指回头。

布局::

    <root>/<session_id>/events.jsonl
    <root>/<session_id>/heads.json      # {"heads": {branch: event_id}, "bases": {branch: event_id}}
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..contracts import ContractError, new_id, now_ts, sha256_hex

GENESIS = "0" * 64

# 允许作为压缩切点的事件类型：一轮结束之后（下一事件应开始新一轮 user 消息）
_TURN_ENDS = {"run_end", "compaction"}


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


@dataclass
class Event:
    seq: int
    event_id: str
    ts: float
    branch: str
    parent: str | None
    type: str
    payload: dict[str, Any]
    prev_digest: str
    digest: str = ""

    @property
    def body(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "event_id": self.event_id,
            "ts": self.ts,
            "branch": self.branch,
            "parent": self.parent,
            "type": self.type,
            "payload": self.payload,
            "prev_digest": self.prev_digest,
        }

    def compute_digest(self) -> str:
        return sha256_hex(_canonical(self.body))

    def seal(self) -> "Event":
        self.digest = self.compute_digest()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {**self.body, "digest": self.digest}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        return cls(
            seq=int(data["seq"]),
            event_id=str(data["event_id"]),
            ts=float(data["ts"]),
            branch=str(data["branch"]),
            parent=data.get("parent"),
            type=str(data["type"]),
            payload=dict(data.get("payload") or {}),
            prev_digest=str(data["prev_digest"]),
            digest=str(data["digest"]),
        )


@dataclass
class SessionStore:
    """单写者假设：同一 session 同时只应有一个进程追加事件（P1 加文件锁）。"""

    root: Path
    session_id: str
    _events: list[Event] = field(default_factory=list, repr=False)
    _by_id: dict[str, Event] = field(default_factory=dict, repr=False)
    _heads: dict[str, str] = field(default_factory=dict, repr=False)
    _bases: dict[str, str] = field(default_factory=dict, repr=False)
    _head_seq: int = field(default=0, repr=False)

    # ------------------------------------------------------------------ #
    # 构造与持久化
    # ------------------------------------------------------------------ #

    @classmethod
    def create(cls, root: Path | str, session_id: str | None = None, *, branch: str = "main") -> "SessionStore":
        store = cls(root=Path(root), session_id=session_id or new_id())
        store.dir.mkdir(parents=True, exist_ok=True)
        store._heads.setdefault(branch, "")
        store._save_heads()
        return store

    @classmethod
    def open(cls, root: Path | str, session_id: str) -> "SessionStore":
        store = cls(root=Path(root), session_id=session_id)
        store._load()
        return store

    @property
    def dir(self) -> Path:
        return self.root / self.session_id

    @property
    def events_path(self) -> Path:
        return self.dir / "events.jsonl"

    @property
    def heads_path(self) -> Path:
        return self.dir / "heads.json"

    def _load(self) -> None:
        self._events.clear()
        self._by_id.clear()
        if self.events_path.exists():
            for index, line in enumerate(self.events_path.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = Event.from_dict(json.loads(line))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    # 崩溃/断电可能留下半行 JSON。以前这里直接抛 JSONDecodeError，
                    # 之后 open() 永久失败、整个会话打不开（code review A-7）。
                    # 现在把坏行隔离到 events.corrupt-<n>.jsonl 并继续加载前面的记录。
                    self._quarantine_line(index, line, exc)
                    continue
                self._events.append(event)
                self._by_id[event.event_id] = event
        if self.heads_path.exists():
            try:
                data = json.loads(self.heads_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                data = {}  # heads 损坏时按事件回放重建（见下）
            self._heads = {str(k): str(v) for k, v in (data.get("heads") or {}).items()}
            self._bases = {str(k): str(v) for k, v in (data.get("bases") or {}).items()}
            self._head_seq = int(data.get("seq") or 0)
        self._heads.setdefault("main", "")
        self._repair_heads()

    def _quarantine_line(self, index: int, line: str, exc: Exception) -> None:
        try:
            path = self.dir / "events.corrupt.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"line": index, "error": str(exc), "raw": line[:2000]}, ensure_ascii=False) + "\n")
        except OSError:  # pragma: no cover - 只读介质
            pass

    def _ancestry_ids(self, event_id: str) -> set[str]:
        ids: set[str] = set()
        current = event_id
        while current and current not in ids:
            ids.add(current)
            event = self._by_id.get(current)
            if event is None:
                break
            current = event.parent or ""
        return ids

    def _repair_heads(self) -> None:
        """崩在"写事件成功、写 heads 之前"时，head 会落后于 jsonl。

        以前这种已付费的回复重启后静默不可见，verify() 也查不出（code review A-6）。
        heads.json 里记了写入时的 ``seq``：只有当 jsonl 的事件数**多于**那个 seq
        时，才说明是"写了事件没来得及写 head"，这时才把 head 前移到分支最后一条。
        主动 rewind 也走 _save_heads（seq 同步），所以不会被误判。
        """
        tail_lost = len(self._events) > self._head_seq
        changed = False
        for branch in self.branches():
            events = [event for event in self._events if event.branch == branch]
            if not events:
                continue
            last = max(events, key=lambda event: event.seq)
            current = self._heads.get(branch) or ""
            if current == last.event_id:
                continue
            if not current:
                self._heads[branch] = last.event_id
                changed = True
                continue
            if not tail_lost:
                continue
            if current in self._ancestry_ids(last.event_id):
                self._heads[branch] = last.event_id
                changed = True
        if changed:
            try:
                self._save_heads()
            except OSError:  # pragma: no cover - 只读介质
                pass

    def _save_heads(self) -> None:
        data = {
            "session_id": self.session_id,
            "heads": self._heads,
            "bases": self._bases,
            # 写入时的事件计数：重启后用它判断"是不是崩在事件与 heads 之间"。
            "seq": len(self._events),
        }
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.dir), prefix=".heads-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.heads_path)  # 原子替换：崩在中间也不会写出半个 heads
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    # ------------------------------------------------------------------ #
    # 读
    # ------------------------------------------------------------------ #

    @property
    def heads(self) -> dict[str, str]:
        return dict(self._heads)

    def head(self, branch: str = "main") -> str:
        return self._heads.get(branch, "")

    def get(self, event_id: str) -> Event:
        try:
            return self._by_id[event_id]
        except KeyError as exc:
            raise ContractError(f"事件不存在：{event_id}") from exc

    def all_events(self) -> list[Event]:
        return list(self._events)

    def lineage(self, branch: str = "main") -> list[Event]:
        """从 head 沿 parent 回溯到根，按时间正序返回。"""
        chain: list[Event] = []
        current = self.head(branch)
        seen: set[str] = set()
        while current:
            if current in seen:
                raise ContractError(f"事件链出现环：{current}")
            seen.add(current)
            event = self.get(current)
            chain.append(event)
            current = event.parent or ""
        chain.reverse()
        return chain

    def branches(self) -> list[str]:
        return sorted(self._heads)

    # ------------------------------------------------------------------ #
    # 写
    # ------------------------------------------------------------------ #

    def append(self, type: str, payload: dict[str, Any] | None = None, *, branch: str = "main") -> Event:
        head_id = self._heads.get(branch, "")
        parent_event = self._by_id.get(head_id) if head_id else None
        event = Event(
            seq=len(self._events) + 1,
            event_id=new_id(),
            ts=now_ts(),
            branch=branch,
            parent=head_id or None,
            type=type,
            payload=dict(payload or {}),
            prev_digest=parent_event.digest if parent_event is not None else GENESIS,
        ).seal()

        self.dir.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        self._events.append(event)
        self._by_id[event.event_id] = event
        self._heads[branch] = event.event_id
        self._save_heads()
        return event

    def rewind(self, event_id: str, *, branch: str = "main") -> None:
        """把分支指针移回某个历史事件。O(1)，不删数据。"""
        event = self.get(event_id)
        self._assert_on_branch(event_id, branch)
        if event.branch != branch:
            raise ContractError(
                f"事件 {event_id} 属于分支 {event.branch!r}，不能直接作为 {branch!r} 的 head；"
                "跨分支请用 fork()"
            )
        self._heads[branch] = event_id
        self._save_heads()

    def fork(self, event_id: str, new_branch: str, *,
             from_branch: str = "main") -> None:
        """从历史事件处分出新分支。新分支继承该事件之前的全部历史。"""
        if new_branch in self._heads:
            raise ContractError(f"分支已存在：{new_branch}")
        self._assert_on_branch(event_id, from_branch)
        self._heads[new_branch] = event_id
        self._bases[new_branch] = event_id
        self._save_heads()

    def _assert_on_branch(self, event_id: str, branch: str) -> None:
        current = self.head(branch)
        while current:
            if current == event_id:
                return
            current = self.get(current).parent or ""
        raise ContractError(f"事件 {event_id} 不在分支 {branch!r} 的历史上")

    # ------------------------------------------------------------------ #
    # 压缩
    # ------------------------------------------------------------------ #

    def append_compaction(self, summary: str, cut_after: str, *, branch: str = "main") -> Event:
        """用摘要替换 ``cut_after`` 之前的逐字历史。

        强校验：``cut_after`` 必须是**轮边界**，否则会出现"半个工具的调用结果"。
        回滚到压缩之前仍然有效（事件没被删），只是 head 移动后的 lineage 看不到旧事件。
        """
        chain = self.lineage(branch)
        ids = [event.event_id for event in chain]
        if cut_after not in ids:
            raise ContractError(f"cut_after={cut_after} 不在分支 {branch!r} 的历史上")
        index = ids.index(cut_after)
        boundary = chain[index]
        is_turn_end = boundary.type in _TURN_ENDS or (
            boundary.type == "assistant" and not boundary.payload.get("tool_calls")
        )
        if not is_turn_end:
            raise ContractError(
                f"压缩切点必须是轮边界（assistant 无 tool_calls / run_end / compaction），"
                f"实际是 {boundary.type!r}"
            )
        if index + 1 < len(chain) and chain[index + 1].type != "user":
            raise ContractError(
                f"压缩切点之后必须紧跟新的一轮 user 消息，实际是 {chain[index + 1].type!r}"
            )
        return self.append("compaction", {"summary": summary, "cut_after": cut_after}, branch=branch)

    # ------------------------------------------------------------------ #
    # 还原成消息
    # ------------------------------------------------------------------ #

    def messages(self, branch: str = "main") -> list[dict[str, Any]]:
        """把事件链 reduce 成 DeepSeek 消息数组（assistant 必带 reasoning_content）。"""
        chain = self.lineage(branch)

        # 找最后一次压缩：它的 cut_after 之前折叠成摘要，之后（到压缩标记为止）仍是逐字历史
        limit = len(chain)
        start = 0
        summary_text = ""
        for index, event in enumerate(chain):
            if event.type != "compaction":
                continue
            limit = index  # 压缩标记本身不是消息，只重建它之前的事件
            summary_text = str(event.payload.get("summary", ""))
            cut_after = event.payload.get("cut_after")
            start = next(
                (position for position, candidate in enumerate(chain) if candidate.event_id == cut_after),
                -1,
            ) + 1

        messages: list[dict[str, Any]] = []
        if summary_text:
            messages.append(
                {
                    "role": "user",
                    "content": f"[CONTEXT COMPACTION] 以下是本轮之前的工作摘要，逐字历史已折叠：\n\n{summary_text}",
                }
            )

        for event in chain[start:limit]:
            if event.type == "compaction":
                continue
            if event.type == "user":
                messages.append({"role": "user", "content": event.payload.get("content", "")})
            elif event.type == "assistant":
                message: dict[str, Any] = {
                    "role": "assistant",
                    "content": event.payload.get("content", ""),
                    "reasoning_content": event.payload.get("reasoning_content", ""),
                }
                if event.payload.get("tool_calls"):
                    message["tool_calls"] = event.payload["tool_calls"]
                messages.append(message)
            elif event.type == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": event.payload.get("tool_call_id", ""),
                        "content": event.payload.get("content", ""),
                    }
                )
            elif event.type in ("nudge", "note"):
                messages.append({"role": "user", "content": event.payload.get("content", "")})
        return messages

    # ------------------------------------------------------------------ #
    # 完整性
    # ------------------------------------------------------------------ #

    def verify(self) -> dict[str, Any]:
        """链式校验。返回报告而不是抛异常，便于 CLI 打印诊断。"""
        problems: list[str] = []
        for index, event in enumerate(self._events):
            expected_seq = index + 1
            if event.seq != expected_seq:
                problems.append(f"seq 不连续：位置 {index} 期望 {expected_seq}，实际 {event.seq}")
            if event.compute_digest() != event.digest:
                problems.append(f"digest 校验失败：{event.event_id}（type={event.type}）")
            # 链式关系按分支校验（events.jsonl 是跨分支的全局追加顺序，不能用前一行当前驱）
            if event.parent is None:
                if event.prev_digest != GENESIS:
                    problems.append(f"根事件 prev_digest 应为 GENESIS：{event.event_id}")
            else:
                parent = self._by_id.get(event.parent)
                if parent is None:
                    problems.append(f"parent 指向不存在的事件：{event.event_id} -> {event.parent}")
                elif event.prev_digest != parent.digest:
                    problems.append(f"prev_digest 与前驱不符：{event.event_id} 的前驱应为 {parent.event_id}")
        for branch, head_id in self._heads.items():
            if head_id and head_id not in self._by_id:
                problems.append(f"分支 {branch!r} 的 head 指向不存在的事件：{head_id}")
        return {
            "ok": not problems,
            "events": len(self._events),
            "branches": self.branches(),
            "problems": problems,
            "head_digest": self.get(self._heads["main"]).digest if self._heads.get("main") else GENESIS,
        }
