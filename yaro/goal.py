"""Goal：持久目标，跨回合续跑（断点续做）。

对齐 DSH/Codex 的"持久 Goal"缺口（gap #4）：一个目标写盘后，agent 每次续跑都先读
它，知道"上次做到哪、还剩什么、是否已完成/被阻塞"，而不是每次从零开始；控制循环
每轮开始读 goal，结束按状态写回。

设计对齐本 harness 的 Goal 机制（create/update 的 action ∈ complete|blocked|pause|resume|edit，
update 用 revision 做 CAS），持久化为 JSON 文件 + 原子写 + 跨进程建议锁。

状态机（合法转移）：
    active ──complete──▶ completed（终态）
    active ──blocked──▶ blocked
    active ──pause──▶ paused
    paused/blocked ──resume──▶ active（清 blocked_reason）
    active/paused ──edit──▶ 自身（改 objective / max_rounds）
    completed/blocked ──edit──▶ 拒绝
"""

from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

STATUS_ACTIVE = "active"
STATUS_PAUSED = "paused"
STATUS_COMPLETED = "completed"
STATUS_BLOCKED = "blocked"
STATUSES = (STATUS_ACTIVE, STATUS_PAUSED, STATUS_COMPLETED, STATUS_BLOCKED)

_UNSET = object()  # "未提供"哨兵，区别于"显式置 None"

_WINDOWS_RESERVED = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)


class GoalError(Exception):
    """目标状态机被违反（非法动作 / 版本冲突 / 目标不存在）。"""


def new_id() -> str:
    return secrets.token_hex(12)


def now_ts() -> str:
    """微秒级 UTC 时间戳，避免同秒更新排序不稳定。"""
    t = time.time()
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t)) + f".{int((t % 1) * 1_000_000):06d}Z"


@dataclass(frozen=True)
class Goal:
    goal_id: str
    objective: str
    status: str = STATUS_ACTIVE
    revision: int = 1
    max_rounds: int | None = None
    blocked_reason: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "objective": self.objective,
            "status": self.status,
            "revision": self.revision,
            "max_rounds": self.max_rounds,
            "blocked_reason": self.blocked_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Goal":
        if not isinstance(d, dict):
            raise GoalError("goal 记录必须是 JSON 对象")
        status = str(d.get("status", STATUS_ACTIVE))
        if status not in STATUSES:
            raise GoalError(f"非法 status：{status!r}")
        revision = d.get("revision", 1)
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise GoalError(f"非法 revision：{revision!r}")
        max_rounds = d.get("max_rounds")
        if max_rounds is not None:
            if isinstance(max_rounds, bool) or not isinstance(max_rounds, int) or max_rounds < 1:
                raise GoalError(f"非法 max_rounds：{max_rounds!r}")
        return cls(
            goal_id=str(d["goal_id"]),
            objective=str(d["objective"]),
            status=status,
            revision=revision,
            max_rounds=max_rounds,
            blocked_reason=str(d.get("blocked_reason", "")),
            created_at=str(d.get("created_at", "")),
            updated_at=str(d.get("updated_at", "")),
        )


class GoalStore:
    """JSON 文件 + 原子写（tmp+os.replace+fsync）+ 跨进程建议锁。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # 建议锁句柄：store 存活期间持有，用于串行化同 store 的并发 update。
        self._lock_path = self.root / ".lock"
        self._lock_fd = open(self._lock_path, "a+b")
        self._lock_fd.write(b"\0")
        self._lock_fd.flush()

    def _path(self, goal_id: str) -> Path:
        if not goal_id or any(
            ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for ch in goal_id
        ):
            raise GoalError(f"非法 goal_id：{goal_id!r}")
        if goal_id.upper() in _WINDOWS_RESERVED:
            raise GoalError(f"goal_id 不能是 Windows 保留名：{goal_id!r}")
        return self.root / f"{goal_id}.json"

    # -- 建议锁（Windows msvcrt / POSIX fcntl；失败静默降级为单写者假设） -- #
    def _acquire(self) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                self._lock_fd.seek(0)
                msvcrt.locking(self._lock_fd.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_EX)
        except Exception:
            pass

    def _release(self) -> None:
        try:
            if os.name == "nt":
                import msvcrt

                self._lock_fd.seek(0)
                msvcrt.locking(self._lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

    def create(self, objective: str, *, max_rounds: int | None = None) -> Goal:
        if not objective or not objective.strip():
            raise GoalError("目标不能为空")
        if max_rounds is not None and max_rounds < 1:
            raise GoalError(f"max_rounds 必须 >= 1，got {max_rounds}")
        goal = Goal(
            goal_id=new_id(),
            objective=objective.strip(),
            max_rounds=max_rounds,
            created_at=now_ts(),
            updated_at=now_ts(),
        )
        self._acquire()
        try:
            self._write(goal)
        finally:
            self._release()
        return goal

    def get(self, goal_id: str) -> Goal | None:
        p = self._path(goal_id)
        if not p.exists():
            return None
        try:
            return Goal.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, GoalError):
            return None  # 坏文件/非法内容当作不存在，不拖垮调用方

    def list(self) -> list[Goal]:
        out: list[Goal] = []
        for p in sorted(self.root.glob("*.json")):
            try:
                out.append(Goal.from_dict(json.loads(p.read_text(encoding="utf-8"))))
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError, GoalError):
                continue  # 目录同名 / 权限 / 非对象 JSON 单文件隔离
        return sorted(out, key=lambda g: g.updated_at, reverse=True)

    def update(
        self,
        goal_id: str,
        revision: int,
        *,
        action: str,
        objective: str | None = None,
        max_rounds: Any = _UNSET,
        blocked_reason: str = "",
    ) -> Goal:
        self._acquire()
        try:
            goal = self.get(goal_id)
            if goal is None:
                raise GoalError(f"目标不存在：{goal_id}")
            if goal.revision != revision:
                raise GoalError(f"版本冲突：目标已到 revision {goal.revision}，你基于 {revision}")

            changes: dict[str, Any] = {}
            if action == "complete":
                if goal.status not in (STATUS_ACTIVE, STATUS_PAUSED):
                    raise GoalError(f"只有 active/paused 可 complete，当前 {goal.status}")
                changes["status"] = STATUS_COMPLETED
                changes["blocked_reason"] = ""
            elif action == "blocked":
                if goal.status not in (STATUS_ACTIVE, STATUS_PAUSED):
                    raise GoalError(f"只有 active/paused 可 blocked，当前 {goal.status}")
                if not blocked_reason or not blocked_reason.strip():
                    raise GoalError("blocked 必须提供 blocked_reason")
                changes["status"] = STATUS_BLOCKED
                changes["blocked_reason"] = blocked_reason.strip()
            elif action == "pause":
                if goal.status != STATUS_ACTIVE:
                    raise GoalError(f"只有 active 可 pause，当前 {goal.status}")
                changes["status"] = STATUS_PAUSED
                changes["blocked_reason"] = ""
            elif action == "resume":
                if goal.status not in (STATUS_PAUSED, STATUS_BLOCKED):
                    raise GoalError(f"只有 paused/blocked 可 resume，当前 {goal.status}")
                changes["status"] = STATUS_ACTIVE
                changes["blocked_reason"] = ""
            elif action == "edit":
                if goal.status not in (STATUS_ACTIVE, STATUS_PAUSED):
                    raise GoalError(f"只有 active/paused 可 edit，当前 {goal.status}")
                if objective is not None:
                    if not objective.strip():
                        raise GoalError("目标不能为空")
                    changes["objective"] = objective.strip()
                if max_rounds is not _UNSET:
                    if max_rounds is not None and (isinstance(max_rounds, bool) or max_rounds < 1):
                        raise GoalError(f"max_rounds 必须 >= 1，got {max_rounds}")
                    changes["max_rounds"] = max_rounds
            else:
                raise GoalError(f"未知 action：{action!r}，可选 complete/blocked/pause/resume/edit")

            new = replace(
                goal,
                revision=goal.revision + 1,
                updated_at=now_ts(),
                **changes,
            )
            self._write(new)
            return new
        finally:
            self._release()

    def _write(self, goal: Goal) -> None:
        p = self._path(goal.goal_id)
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(goal.to_dict(), ensure_ascii=False, indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
        self._fsync_dir()

    def _fsync_dir(self) -> None:
        try:
            fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass  # Windows 上目录 fsync 不支持，忽略
