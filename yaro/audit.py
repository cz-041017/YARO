"""审计账本：append-only 事件流 + digest 链校验。

从旧 :class:`yaro.memory.session.SessionStore` 抽出，只保留「写事件 + digest 链 + verify」，
去掉会话树职责（heads / rewind / fork / lineage / compaction）。蒸馏运行时每个节点写一条；
将来 server 迁移到新 LangGraph 图后也用它，与 checkpoint 并存（checkpoint 管状态，账本管不可抵赖）。

不变式：

1. **append-only**：事件一旦写入不修改、不删除。
2. **可验证**：每条事件带 ``prev_digest``，``digest`` 覆盖自身全部字段，链式校验。
3. **坏行隔离**：崩溃可能留下半行 JSON，加载时隔离到 ``<path>.corrupt.jsonl`` 继续读前面的记录。

布局：单文件 ``events.jsonl``（一行一条 JSON，逐条 fsync）。

已知限制：``verify()`` 只做**链内自洽**（seq 连续 / digest 匹配 / prev_digest 对得上），
没有外部 head 锚点，因此**删除末尾 N 条**（尾部截断）无法被检测——防篡改锚点
（HMAC / 把 head_digest 存到账本之外）属于 P1，本版不实现，调用方须知情。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from .contracts import new_id, now_ts, sha256_hex

GENESIS = "0" * 64


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    ).encode("utf-8")


@dataclass
class AuditEvent:
    """一条审计事件。``digest`` 覆盖 ``body`` 的全部字段，``prev_digest`` 指向前一条。"""

    seq: int
    event_id: str
    ts: float
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
            "type": self.type,
            "payload": self.payload,
            "prev_digest": self.prev_digest,
        }

    def compute_digest(self) -> str:
        return sha256_hex(_canonical(self.body))

    def seal(self) -> "AuditEvent":
        self.digest = self.compute_digest()
        return self

    def to_dict(self) -> dict[str, Any]:
        return {**self.body, "digest": self.digest}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        return cls(
            seq=int(data["seq"]),
            event_id=str(data["event_id"]),
            ts=float(data["ts"]),
            type=str(data["type"]),
            payload=dict(data.get("payload") or {}),
            prev_digest=str(data["prev_digest"]),
            digest=str(data["digest"]),
        )


@dataclass
class AuditLedger:
    """单写者假设：同一账本同时只应有一个进程追加（跨进程锁放 P1）。"""

    path: Path
    _events: list[AuditEvent] = field(default_factory=list, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False, init=False)

    # ------------------------------------------------------------------ #
    # 构造与持久化
    # ------------------------------------------------------------------ #

    @classmethod
    def create(cls, path: Path | str) -> "AuditLedger":
        ledger = cls(path=Path(path))
        ledger.path.parent.mkdir(parents=True, exist_ok=True)
        return ledger

    @classmethod
    def open(cls, path: Path | str) -> "AuditLedger":
        ledger = cls(path=Path(path))
        ledger._load()
        return ledger

    def _load(self) -> None:
        self._events.clear()
        if not self.path.exists():
            return
        for index, line in enumerate(self.path.read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                event = AuditEvent.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                self._quarantine(index, line, exc)
                continue
            self._events.append(event)

    def _quarantine(self, index: int, line: str, exc: Exception) -> None:
        try:
            path = self.path.with_suffix(".corrupt.jsonl")
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps({"line": index, "error": str(exc), "raw": line[:2000]}, ensure_ascii=False) + "\n"
                )
        except OSError:  # pragma: no cover - 只读介质
            pass

    # ------------------------------------------------------------------ #
    # 读
    # ------------------------------------------------------------------ #

    def events(self) -> list[AuditEvent]:
        return list(self._events)

    def head_digest(self) -> str:
        return self._events[-1].digest if self._events else GENESIS

    def __len__(self) -> int:
        return len(self._events)

    # ------------------------------------------------------------------ #
    # 写
    # ------------------------------------------------------------------ #

    def append(self, type: str, payload: dict[str, Any] | None = None) -> AuditEvent:
        with self._lock:
            prev = self._events[-1].digest if self._events else GENESIS
            event = AuditEvent(
                seq=len(self._events) + 1,
                event_id=new_id(),
                ts=now_ts(),
                type=type,
                payload=dict(payload or {}),
                prev_digest=prev,
            ).seal()

            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            self._events.append(event)
            return event

    # ------------------------------------------------------------------ #
    # 完整性
    # ------------------------------------------------------------------ #

    def verify(self) -> dict[str, Any]:
        """链式校验。返回报告而不是抛异常，便于打印诊断。"""
        problems: list[str] = []
        for index, event in enumerate(self._events):
            if event.seq != index + 1:
                problems.append(f"seq 不连续：位置 {index} 期望 {index + 1}，实际 {event.seq}")
            if event.compute_digest() != event.digest:
                problems.append(f"digest 校验失败：{event.event_id}（type={event.type}）")
            if index == 0:
                if event.prev_digest != GENESIS:
                    problems.append(f"根事件 prev_digest 应为 GENESIS：{event.event_id}")
            else:
                if event.prev_digest != self._events[index - 1].digest:
                    problems.append(f"prev_digest 与前驱不符：{event.event_id}")
        return {
            "ok": not problems,
            "events": len(self._events),
            "problems": problems,
            "head_digest": self.head_digest(),
        }
