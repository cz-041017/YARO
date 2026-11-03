"""本地 Web GUI：一个零依赖的 HTTP 服务 + 三个静态文件。

为什么不用 Streamlit / Gradio / FastAPI+React（见 ``docs/ADR-0003-gui.md``）：

1. **核心零依赖是契约**。``pip install -e .`` 之后 ``yaro gui`` 必须能直接起，
   不能因为缺 node / 缺前端构建链而失败；
2. 前端没有构建步骤 → 没有"bundle 过没过期"这一类不可验证的风险；
3. 轮询而不是 SSE：P0 的并发是"一个人一个会话"，900ms 拉一次增量事件
   （~2KB）比维护长连接 + 心跳 + 重连的状态机便宜得多，也更好调试。

安全边界：只监听 127.0.0.1；静态文件走白名单；产物接口只接受 64 位十六进制
sha256；所有路径参数都相对工作区解析，越界即拒绝。生成代码的执行仍走 Docker
沙箱（``run_python``），HTTP 层不提供任何可直接执行代码的接口。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import traceback
import uuid
import webbrowser
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import __version__
from .capabilities.builtin import build_default_tools, build_tool_catalog
from .mcp import mcp_tools, servers_from_config
from .capabilities.registry import ToolRegistry
from .capabilities.sandbox import DockerSandbox
from .cli import doctor_report
from .config import Settings, system_prompt
from .contracts import ContractError, PRICES_USD_PER_MTOK, Usage
from .demo import ensure_demo_assets
from .egress import MODE_LABELS, summarize_policy
from .gates import scan_files
from .graph.agent import DEFAULT_KEEP_TAIL_RATIO, DEFAULT_MAX_ITERATIONS, DEFAULT_MAX_TOKENS
from .graph.runner import run_agent
from .llm.deepseek import is_peak
from .memory.artifacts import ArtifactStore
from .provider.demo import DemoChatModel
from .provider.guard import GuardedModel
from .provider.models import build_model
from .memory.session import SessionStore
from .secrets import status as secret_status
from .secrets import scrub

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
_STATIC = {
    "index.html": "text/html; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "logo.svg": "image/svg+xml",
}
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# 只服务本机：这组白名单同时用于防 DNS rebinding（Host 头）与防跨站请求（Origin 头）。
_LOCAL_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1", "0.0.0.0"})
# 允许写密钥的客户端地址：只认真正的回环地址（0.0.0.0 只是监听地址，不是来源）
_LOOPBACK_CLIENTS = frozenset({"127.0.0.1", "::1", "127.0.0.2"})


def _hostname_of(authority: str) -> str:
    """从 Host/Origin 的 authority 里取出主机名（去掉端口、去方括号）。"""
    value = (authority or "").strip().lower()
    if value.startswith("["):  # [::1]:8765
        end = value.find("]")
        return value[1:end] if end > 0 else value
    if ":" in value:
        return value.rsplit(":", 1)[0]
    return value
DEFAULT_DEMO_TASK = "把 demo/dataset 走一遍全链路：体检数据、抽查标注、切分、给训练计划并诊断训练日志，按 CONTRACT 给结论。"


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class RunState:
    session_id: str
    demo: bool
    effort: str | None
    started: float
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


class Harness:
    """GUI 的后端门面：会话、运行、产物、闸门、自检。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings.ensure()
        self.artifacts = ArtifactStore(self.settings.artifacts_dir)
        # 演示资源在启动时就铺好：GUI 第一次点「运行」不该因为 demo/dataset 缺失而失败。
        try:
            ensure_demo_assets(self.settings.demo_dir)
        except OSError:  # 只读目录等极端情况下不阻断启动，GUI 会显示数据集缺失
            pass
        # 出网策略在服务启动时固化：运行期间改环境变量不会悄悄放宽隐私边界。
        self.egress = self.settings.egress_policy()
        # 默认只允许本机访问；真正要暴露给内网/容器时显式开 YARO_GUI_PUBLIC=1
        # （同时会关掉 Host/Origin 白名单 —— 那种场景应自己加反向代理与认证）。
        self.allow_public_gui = os.environ.get("YARO_GUI_PUBLIC") == "1"
        self._runs: dict[str, RunState] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # API Key（GUI 里的入口，只允许本机写）
    # ------------------------------------------------------------------ #

    def set_api_key(self, *, action: str, key: str = "") -> dict[str, Any]:
        """把 Key 写进本机的 keyring / ``~/.yaro/secrets.json``（0600）。

        设计边界（很重要）：
        - 只接受本机回环地址的请求（路由层已校验），并且公开模式（``YARO_GUI_PUBLIC=1``）直接拒绝；
        - 响应里**永远不带原文**，只回 ``secrets.status()`` 的掩码与来源；
        - 不写事件、不写日志、不落 traceback；写入路径与 ``yaro key set`` 完全相同。
        """
        from . import secrets as secret_module

        normalized = (action or "").strip().lower()
        if normalized == "set":
            value = (key or "").strip()
            if not value:
                raise ApiError("API Key 不能为空")
            if len(value) < 8:
                raise ApiError("API Key 太短（至少 8 个字符）")
            if len(value) > 512:
                raise ApiError("API Key 太长（最多 512 个字符）")
            if any(ch.isspace() for ch in value):
                raise ApiError("API Key 里不能有空格/换行")
            stored = False
            try:
                stored = bool(secret_module.keyring_set(value))
            except Exception:  # noqa: BLE001 - 没有 keyring 就退回本地文件
                stored = False
            if not stored:
                secret_module.set_api_key(self.settings.home, value)
            self.settings = replace(
                self.settings,
                api_key=value,
                api_key_source="keyring" if stored else "file",
            )
        elif normalized == "clear":
            try:
                secret_module.keyring_clear()
            except Exception:  # noqa: BLE001
                pass
            secret_module.clear_api_key(self.settings.home)
            self.settings = replace(self.settings, api_key="", api_key_source="none")
        else:
            raise ApiError(f"未知操作：{action!r}（可选 set / clear）")
        # 出网策略里也带着密钥用于脱敏，必须同步重建。
        self.egress = self.settings.egress_policy()
        return {"ok": True, "key": secret_module.status(self.settings.home)}

    def set_egress(self, *, mode: str, allow_public: bool) -> dict[str, Any]:
        """GUI 一键改出网策略：校验 → 立即生效（重建不可变策略）→ 落地 ``.env``。

        与 :meth:`set_api_key` 同样的边界：只允许回环地址（路由层校验）、公开模式拒绝、
        响应不带敏感信息。运行时重建 ``self.egress``，所以下一次 ``_run_async`` 就用新策略；
        ``.env`` 落盘保证重启后仍是用户选的值（不依赖进程内环境变量）。
        """
        from .config import update_egress_env
        from .egress import normalize_mode

        try:
            normalized = normalize_mode(mode)
        except ContractError as exc:
            raise ApiError(str(exc), status=400) from exc
        allow = bool(allow_public)
        self.settings = replace(self.settings, egress=normalized, egress_allow_public=allow)
        self.egress = self.settings.egress_policy()
        env_path = update_egress_env(normalized, allow)
        return {
            "ok": True,
            "egress": {
                **self.egress.describe(),
                "target": self.settings.base_url,
                "summary": summarize_policy(self.egress, self.settings.base_url),
                "modes": {m: label for m, label in MODE_LABELS.items()},
            },
            "env_path": str(env_path),
        }

    # ------------------------------------------------------------------ #
    # 配置与会话
    # ------------------------------------------------------------------ #

    def config(self) -> dict[str, Any]:
        sandbox = DockerSandbox(image=self.settings.sandbox_image, allow_net=self.settings.allow_net)
        demo = ensure_demo_assets(self.settings.demo_dir)
        return {
            "version": __version__,
            "model": self.settings.model,
            "model_plan": self.settings.model_plan,
            "has_key": bool(self.settings.api_key),
            "docker": sandbox.available(),
            "workspace": str(self.settings.workspace),
            "home": str(self.settings.home),
            "demo": {
                "dir": str(self.settings.demo_dir),
                "relative_dir": demo["relative_dir"],
                "files": demo["files"],
                "dataset": demo.get("dataset", ""),
                "dataset_files": demo.get("dataset_files", []),
                "dataset_planted": demo.get("dataset_planted", []),
            },
            "session_count": len(self._session_dirs()),
            "artifact_count": len(self.artifacts),
            "artifact_bytes": self.artifacts.stats()["bytes"],
            "budget": {
                "max_iterations": DEFAULT_MAX_ITERATIONS,
                "max_tokens": DEFAULT_MAX_TOKENS,
                "keep_tail_ratio": DEFAULT_KEEP_TAIL_RATIO,
            },
            "prices": PRICES_USD_PER_MTOK,
            "peak_now": is_peak(),
            # 隐私边界：GUI 顶部/左栏要用它显示"当前是否出网、图片是否允许发送"。
            "egress": {
                **self.egress.describe(),
                "target": self.settings.base_url,
                "summary": summarize_policy(self.egress, self.settings.base_url),
                "modes": {mode: label for mode, label in MODE_LABELS.items()},
            },
            "key": secret_status(self.settings.home),
            "tools": [
                {"name": item["name"], "summary": item["description"].split("。")[0][:40]}
                for item in build_tool_catalog(self.settings.workspace)
            ],
            # MCP 对内消费（社区视觉工具）：GUI 工具发现路径展示已配置的服务器；
            # 其具体工具在每次运行 _run_async 里按 servers_from_config 现连现发现。
            "mcp_servers": [
                {"name": s["name"], "transport": s.get("transport", "stdio")}
                for s in self.settings.mcp_servers
            ],
        }

    def _session_dirs(self) -> list[Path]:
        root = self.settings.sessions_dir
        if not root.exists():
            return []
        return sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name)

    def list_sessions(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in self._session_dirs():
            row: dict[str, Any] = {"session_id": path.name, "title": "未命名会话", "events": 0, "verified": True}
            try:
                store = SessionStore.open(self.settings.sessions_dir, path.name)
                events = store.all_events()
                row["events"] = len(events)
                row["verified"] = store.verify()["ok"]
                row["branches"] = store.branches()
                row["updated"] = events[-1].ts if events else 0.0
                for event in events:
                    if event.type == "user":
                        row["title"] = str(event.payload.get("content") or "")[:60].replace("\n", " ") or "空任务"
                        break
            except Exception as exc:  # noqa: BLE001 - 坏会话不该让列表整体失败
                row["verified"] = False
                row["error"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        rows.sort(key=lambda item: item.get("updated", 0.0), reverse=True)
        with self._lock:
            running = set(self._runs)
        for row in rows:
            row["running"] = row["session_id"] in running
        return rows

    def create_session(self) -> dict[str, Any]:
        store = SessionStore.create(self.settings.sessions_dir)
        return {"session_id": store.session_id, "heads": store.heads}

    def store(self, session_id: str) -> SessionStore:
        if not _SESSION_ID.match(session_id):
            raise ApiError(f"非法会话 id：{session_id!r}")
        if not (self.settings.sessions_dir / session_id).is_dir():
            raise ApiError(f"会话不存在：{session_id}", status=404)
        return SessionStore.open(self.settings.sessions_dir, session_id)

    def session_state(self, session_id: str, since: int = 0, *, verify: bool = False) -> dict[str, Any]:
        store = self.store(session_id)
        with self._lock:
            run = self._runs.get(session_id)
        # 只读一次全量事件：以前这里同一轮询里调了 3 次 all_events()，事件上千后
        # 800ms 轮询会把 CPU 放大（code review C-9）。
        all_events = store.all_events()
        lineage = store.lineage()
        events = [event for event in all_events if event.seq > since]
        turns = sum(1 for event in lineage if event.type == "assistant")
        usage = Usage()
        cost = 0.0
        stop_reason = ""
        for event in lineage:
            if event.type != "run_end":
                continue
            stop_reason = str(event.payload.get("stop_reason") or "")
            # 成本与用量要跨 run 累加：以前是覆盖赋值，多轮会话只显示最后一轮
            # （code review A-9 / C）。
            if event.payload.get("usage"):
                usage.add(Usage(**{key: int(value) for key, value in event.payload["usage"].items()}))
            cost += float(event.payload.get("cost_usd") or 0.0)
        return {
            "session_id": session_id,
            "events": [event.to_dict() for event in events],
            "seq": max([event.seq for event in all_events] or [0]),
            "lineage": [event.event_id for event in lineage],
            "head": store.head("main"),
            "heads": store.heads,
            "branches": store.branches(),
            "branch": "main",
            # 全链校验开销随事件数线性增长，只在首屏或显式点击「校验事件链」时做
            "verified": store.verify()["ok"] if (verify or since == 0) else None,
            "running": {
                "active": run is not None,
                "demo": run.demo if run else False,
                "effort": run.effort if run else None,
                "started": run.started if run else 0.0,
            },
            "stats": {
                "turns": turns,
                "events": len(all_events),
                "usage": usage.to_dict(),
                "hit_rate": usage.hit_rate,
                "cost_usd": cost,
                "stop_reason": stop_reason,
            },
        }

    # ------------------------------------------------------------------ #
    # 运行
    # ------------------------------------------------------------------ #

    def start_run(self, session_id: str, *, task: str, demo: bool, effort: str | None) -> dict[str, Any]:
        store = self.store(session_id)
        with self._lock:
            if session_id in self._runs:
                raise ApiError("该会话已有运行中的任务，请先停止或换一个会话", status=409)
            state = RunState(session_id=session_id, demo=demo, effort=effort, started=time.time())
            self._runs[session_id] = state

        task = (task or "").strip() or (DEFAULT_DEMO_TASK if demo else "")
        if not task:
            with self._lock:
                self._runs.pop(session_id, None)
            raise ApiError("任务描述不能为空")
        if not demo and not self.settings.api_key:
            with self._lock:
                self._runs.pop(session_id, None)
            raise ApiError("缺少 API Key：运行 `yaro key set`（推荐）或设置 DEEPSEEK_API_KEY，或打开演示模式")
        if not demo:
            # 出网策略在真正发请求前就拦下来，避免用户等一轮才发现被拒。
            try:
                self.egress.check_endpoint(self.settings.base_url)
            except ContractError as exc:
                with self._lock:
                    self._runs.pop(session_id, None)
                raise ApiError(str(exc)) from exc

        state.thread = threading.Thread(
            target=self._run_thread,
            args=(store, state, task),
            name=f"yaro-run-{session_id[:8]}",
            daemon=True,
        )
        state.thread.start()
        return {"session_id": session_id, "demo": demo, "running": True}

    def _run_thread(self, store: SessionStore, state: RunState, task: str) -> None:
        try:
            asyncio.run(self._run_async(store, state, task))
        except BaseException as exc:  # noqa: BLE001 - 线程边界，必须记账而不是静默死掉
            secret = self.settings.api_key
            detail = scrub(f"{type(exc).__name__}: {exc}", secret)
            try:
                store.append(
                    "run_end",
                    {
                        "stop_reason": "error",
                        "detail": detail,
                        "trace": scrub(traceback.format_exc()[-1500:], secret),
                    },
                    branch="main",
                )
            except BaseException:  # pragma: no cover - 会话也写不进去时只能放弃
                pass
        finally:
            with self._lock:
                self._runs.pop(state.session_id, None)

    async def _run_async(self, store: SessionStore, state: RunState, task: str) -> None:
        settings = self.settings
        run_id = state.session_id
        tools = build_default_tools(
            settings.workspace,
            self.artifacts,
            run_id=run_id,
            step=0,
        )
        # MCP 对内消费：把配置的社区视觉工具 MCP 服务器连进来，工具当一等工具用。
        # 配置/连接失败 fail-closed 但降级（本地工具仍可用），不炸掉整次运行。
        extra_tools: list[Any] = []
        try:
            mcp_servers = servers_from_config(settings.mcp_servers)
        except ValueError as exc:
            logger.warning("MCP 配置错误，跳过 MCP 工具：%s", exc)
            mcp_servers = []
        if mcp_servers:
            try:
                async with mcp_tools(mcp_servers) as extra:
                    extra_tools = list(extra)
            except BaseException as exc:  # noqa: BLE001 - MCP 服务器不可达时降级，不杀整次运行
                logger.warning("MCP 工具加载失败，降级为本地工具集：%s", exc)
                extra_tools = []
        # MCP 工具与内置工具同名时跳过（否则 ToolRegistry 抛 ContractError 穿透运行）；
        # 明确不静默覆盖内置工具（fail-closed：内置工具优先）。
        if extra_tools:
            builtin_names = {getattr(getattr(t, "spec", None), "name", "") for t in tools}
            deduped: list[Any] = []
            for t in extra_tools:
                name = getattr(getattr(t, "spec", None), "name", "")
                if name and name in builtin_names:
                    logger.warning("MCP 工具 %r 与内置工具同名，已跳过", name)
                else:
                    deduped.append(t)
            extra_tools = deduped
        registry = ToolRegistry(tools + extra_tools)
        if state.demo:
            model: Any = DemoChatModel(delay_s=0.0)
            # demo 用假 usage（deepseek-flash 价格），固定 flash 计价，不受 YARO_MODEL 影响（修 L2）。
            cost_model = "deepseek-flash"
        else:
            base = build_model(settings, model=settings.model, effort=state.effort)
            model = GuardedModel(base, policy=self.egress, base_url=settings.base_url)
            cost_model = settings.model
        await run_agent(
            model,
            registry,
            store,
            task,
            system=system_prompt(),
            branch="main",
            run_id=run_id,
            max_iterations=DEFAULT_MAX_ITERATIONS,
            egress=self.egress,
            stop_check=state.stop_event.is_set,
            model_name=cost_model,
            peak=is_peak(),
            artifacts=self.artifacts,
        )

    def stop(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._runs.get(session_id)
        if state is None:
            raise ApiError("该会话没有运行中的任务", status=409)
        state.stop_event.set()
        return {"session_id": session_id, "stopping": True}

    # ------------------------------------------------------------------ #
    # 会话树操作
    # ------------------------------------------------------------------ #

    def rewind(self, session_id: str, event_id: str) -> dict[str, Any]:
        with self._lock:
            if session_id in self._runs:
                raise ApiError("运行中不能回滚，请先停止", status=409)
        store = self.store(session_id)
        store.rewind(event_id)
        return {"session_id": session_id, "heads": store.heads}

    def fork(self, session_id: str, event_id: str, branch: str) -> dict[str, Any]:
        if not _SESSION_ID.match(branch):
            raise ApiError(f"非法分支名：{branch!r}")
        store = self.store(session_id)
        store.fork(event_id, branch)
        return {"session_id": session_id, "heads": store.heads, "branches": store.branches()}

    def verify(self, session_id: str) -> dict[str, Any]:
        return self.store(session_id).verify()

    # ------------------------------------------------------------------ #
    # 产物 / 闸门 / 自检
    # ------------------------------------------------------------------ #

    def artifact_entries(self) -> list[dict[str, Any]]:
        rows = []
        for sha, meta in self.artifacts.entries().items():
            try:
                created = (self.settings.artifacts_dir / "objects" / sha[:2] / f"{sha}.{meta['ext']}").stat().st_mtime
            except OSError:
                created = 0.0
            rows.append(
                {
                    "sha256": sha,
                    "media_type": meta.get("media_type", "application/octet-stream"),
                    "size": int(meta.get("size", 0)),
                    "ext": meta.get("ext", ""),
                    "created": created,
                }
            )
        rows.sort(key=lambda row: row["created"], reverse=True)
        return rows

    def artifact_bytes(self, sha: str) -> tuple[bytes, str]:
        if not _SHA256.match(sha):
            raise ApiError("非法 sha256", status=400)
        entries = self.artifacts.entries()
        if sha not in entries:
            raise ApiError("产物不存在", status=404)
        meta = entries[sha]
        path = self.settings.artifacts_dir / "objects" / sha[:2] / f"{sha}.{meta['ext']}"
        if not path.is_file():
            raise ApiError("产物文件缺失（索引与存储不一致）", status=410)
        return path.read_bytes(), str(meta.get("media_type", "application/octet-stream"))

    def gates(self, relative: str) -> dict[str, Any]:
        target = (self.settings.workspace / (relative or ".")).resolve()
        try:
            target.relative_to(self.settings.workspace)
        except ValueError as exc:
            raise ApiError(f"路径越界：{relative!r}") from exc
        if not target.exists():
            raise ApiError(f"路径不存在：{relative}", status=404)
        result = scan_files([str(target)])
        findings = [
            {
                "severity": str(getattr(item, "severity", "")),
                "rule": str(getattr(item, "rule", "")),
                "message": str(getattr(item, "message", "")),
                "fix": str(getattr(item, "fix", "")),
                "file": str(getattr(item, "file", "")),
                "line": int(getattr(item, "line", 0) or 0),
            }
            for item in result.findings
        ]
        return {"ok": bool(result.ok), "summary": result.summary(), "findings": findings, "path": relative or "."}

    def doctor(self) -> dict[str, Any]:
        return doctor_report(self.settings)


# --------------------------------------------------------------------------- #
# HTTP 层
# --------------------------------------------------------------------------- #


class YaroHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], harness: Harness) -> None:
        super().__init__(address, Handler)
        self.harness = harness


class Handler(BaseHTTPRequestHandler):
    server_version = f"YARO/{__version__}"
    protocol_version = "HTTP/1.1"

    # -- 基础设施 ------------------------------------------------------- #

    @property
    def harness(self) -> Harness:
        return self.server.harness  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - 覆盖父类签名
        # 默认会把每个请求打到 stderr，GUI 轮询下非常吵；调试时设 YARO_GUI_LOG=1
        if os.environ.get("YARO_GUI_LOG") == "1":
            super().log_message(format, *args)

    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 4 * 1024 * 1024:
            raise ApiError("请求体过大", status=413)
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(data, dict):
            raise ApiError("请求体必须是 JSON object")
        return data

    def _guard(self, *, allow_public: bool = False) -> None:
        """本地服务的两道门：Host 必须是本机（防 DNS rebinding），Origin 必须是本机（防跨站 POST）。

        没有认证机制时，这两条是唯一能区分"我自己点的"与"别的网页偷偷发的"的东西。
        命令行工具（curl/测试）不发 Origin，所以照常可用。
        """
        if allow_public:
            return
        host = _hostname_of(self.headers.get("Host") or "")
        if host and host not in _LOCAL_HOSTNAMES:
            raise ApiError(f"拒绝非本机 Host：{host}", status=403)
        origin = (self.headers.get("Origin") or "").strip()
        if origin:
            try:
                origin_host = _hostname_of(urlsplit(origin).netloc)
            except ValueError:
                origin_host = ""
            if origin_host not in _LOCAL_HOSTNAMES:
                raise ApiError(f"拒绝跨站请求：Origin={origin}", status=403)

    def _dispatch(self, method: str) -> None:
        try:
            self._guard(allow_public=bool(getattr(self.harness, "allow_public_gui", False)))
            split = urlsplit(self.path)
            path = split.path
            query = {key: values[-1] for key, values in parse_qs(split.query).items()}
            if method == "GET":
                payload, status = self._route_get(path, query)
            else:
                payload, status = self._route_post(path, query)
            if isinstance(payload, tuple):  # (bytes, content_type)
                self._send_bytes(payload[0], payload[1], status)
            else:
                self._send_json(payload, status)
        except ApiError as exc:
            self._send_json({"error": str(exc)}, exc.status)
        except BrokenPipeError:  # pragma: no cover - 浏览器刷新/关闭
            pass
        except BaseException as exc:  # noqa: BLE001
            # 默认不回 traceback：里面有绝对路径、工作区片段、调用栈，
            # 而 scrub 只能擦 api_key（code review C-3）。排障时显式开 YARO_DEBUG=1。
            secret = getattr(self.harness, "settings", None)
            secret_key = secret.api_key if secret is not None else ""
            request_id = uuid.uuid4().hex[:12]
            payload: dict[str, Any] = {
                "error": scrub(f"{type(exc).__name__}: {exc}", secret_key),
                "request_id": request_id,
            }
            if os.environ.get("YARO_DEBUG") == "1":
                payload["trace"] = scrub(traceback.format_exc()[-1200:], secret_key)
            self._send_json(payload, 500)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    # -- 路由 ----------------------------------------------------------- #

    def _route_get(self, path: str, query: dict[str, str]) -> tuple[Any, int]:
        harness = self.harness
        if path in ("/", "/index.html"):
            return _read_static("index.html"), 200
        if path.startswith("/assets/"):
            name = path[len("/assets/") :]
            if name not in _STATIC:
                raise ApiError(f"未知静态资源：{name}", status=404)
            return _read_static(name), 200
        if path == "/api/config":
            return harness.config(), 200
        if path == "/api/sessions":
            return {"sessions": harness.list_sessions()}, 200
        if path == "/api/artifacts":
            return {"artifacts": harness.artifact_entries()}, 200
        if path == "/api/doctor":
            return harness.doctor(), 200
        if path == "/api/gates":
            return harness.gates(query.get("path", ".")), 200
        if path.startswith("/api/artifact/"):
            sha = path[len("/api/artifact/") :]
            body, content_type = harness.artifact_bytes(sha)
            return (body, content_type), 200
        if path.startswith("/api/sessions/"):
            parts = [segment for segment in path[len("/api/sessions/") :].split("/") if segment]
            if len(parts) != 1:
                raise ApiError("未知接口", status=404)
            # 非法 since 以前直接把 int() 的 ValueError 抛成 500 + traceback
            try:
                since = int(query.get("since") or 0)
            except (TypeError, ValueError):
                raise ApiError(f"非法 since：{query.get('since')!r}") from None
            verify = query.get("verify") in ("1", "true", "yes")
            return harness.session_state(parts[0], since=max(0, since), verify=verify), 200
        raise ApiError(f"未知接口：{path}", status=404)

    def _route_post(self, path: str, query: dict[str, str]) -> tuple[Any, int]:
        harness = self.harness
        if path == "/api/key":
            # 写密钥是最敏感的一个接口：不经过浏览器缓存、只允许回环地址、公开模式拒绝。
            if getattr(harness, "allow_public_gui", False):
                raise ApiError("公开模式下禁止通过 HTTP 写入 API Key（请用 `yaro key set`）", status=403)
            if str(self.client_address[0]) not in _LOOPBACK_CLIENTS:
                raise ApiError("API Key 只允许从本机写入", status=403)
            body = self._read_json()
            return harness.set_api_key(
                action=str(body.get("action") or ""),
                key=str(body.get("key") or ""),
            ), 200
        if path == "/api/egress":
            # 出网策略与密钥同级敏感：只允许回环、公开模式拒绝（可放宽出网本身，但不能从公网改）。
            if getattr(harness, "allow_public_gui", False):
                raise ApiError("公开模式下禁止通过 HTTP 改出网策略（请用 `yaro init`）", status=403)
            if str(self.client_address[0]) not in _LOOPBACK_CLIENTS:
                raise ApiError("出网策略只允许从本机修改", status=403)
            body = self._read_json()
            return harness.set_egress(
                mode=str(body.get("mode") or ""),
                allow_public=body.get("allow_public") is True,
            ), 200
        if path == "/api/sessions":
            return harness.create_session(), 200
        if path.startswith("/api/sessions/"):
            parts = [segment for segment in path[len("/api/sessions/") :].split("/") if segment]
            if not parts:
                raise ApiError("未知接口", status=404)
            session_id = parts[0]
            action = parts[1] if len(parts) > 1 else ""
            body = self._read_json()
            if action == "run":
                demo = body.get("demo") is True
                effort = body.get("effort") or None
                if effort not in (None, "low", "high", "max"):
                    raise ApiError(f"非法 effort：{effort!r}")
                return harness.start_run(session_id, task=str(body.get("task") or ""), demo=demo, effort=effort), 200
            if action == "stop":
                return harness.stop(session_id), 200
            if action == "rewind":
                return harness.rewind(session_id, str(body.get("event_id") or "")), 200
            if action == "fork":
                return harness.fork(session_id, str(body.get("event_id") or ""), str(body.get("branch") or "")), 200
            if action == "verify":
                return harness.verify(session_id), 200
            raise ApiError(f"未知操作：{action!r}", status=404)
        raise ApiError(f"未知接口：{path}", status=404)


def _read_static(name: str) -> tuple[bytes, str]:
    path = WEB_DIR / name
    if not path.is_file():
        raise ApiError(f"静态资源缺失：{name}（安装时是否漏了 package-data？）", status=500)
    return path.read_bytes(), _STATIC[name]


def _bind(host: str, port: int, harness: Harness) -> YaroHTTPServer:
    """端口被占用时顺延尝试，最多 12 个（同一台机器常同时开着别的 dev server）。"""
    last_error: OSError | None = None
    for offset in range(12):
        candidate = port + offset
        try:
            return YaroHTTPServer((host, candidate), harness)
        except OSError as exc:
            last_error = exc
            continue
    raise SystemExit(f"端口 {port}–{port + 11} 都不可用：{last_error}")


_BINDABLE_LOCAL = frozenset({"127.0.0.1", "localhost", "::1"})


def _validate_bind_host(host: str) -> None:
    """把无认证服务放到局域网必须显式开闸（code review C-1）。

    以前 ``yaro gui --host 0.0.0.0`` + ``YARO_GUI_PUBLIC=1`` 一步就把
    "同网段谁都能用你的 API Key 跑任务"的服务暴露出去。
    """
    if host in _BINDABLE_LOCAL:
        return
    if os.environ.get("YARO_GUI_PUBLIC") == "1":
        print(
            f"[警告] 正在把无认证的 GUI 暴露到 {host}。任何能访问该端口的人都能读写你的"
            "工作区与产物、用你的 API Key 发起模型请求。请只在可信内网使用，并在前面加反向代理与认证。"
        )
        return
    raise SystemExit(
        f"拒绝绑定 {host}：YARO GUI 没有认证，默认只允许本机监听。\n"
        "  要暴露到内网请显式设置 YARO_GUI_PUBLIC=1 并自行加反向代理/认证；"
        "  SSH 场景建议用端口转发：ssh -L 8765:127.0.0.1:8765 <host>"
    )


def serve(settings: Settings | None = None, *, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True) -> None:
    _validate_bind_host(host)
    settings = (settings or Settings.load()).ensure()
    harness = Harness(settings)
    httpd = _bind(host, port, harness)
    actual = httpd.server_address[1]
    url = f"http://{host}:{actual}/"
    print(f"YARO GUI → {url}")
    print(f"  workspace: {settings.workspace}")
    print(f"  home:      {settings.home}")
    print(f"  model:     {settings.model}" + ("" if settings.api_key else "（未配置 Key，请用 Demo 模式）"))
    # 隐私边界必须在启动时明确打印出来：用户有权一眼看到当前会不会出网。
    print(f"  隐私:      {summarize_policy(harness.egress, settings.base_url)}")
    if harness.egress.mode == "off":
        print("             默认不出网：真实模型调用会被拒绝；勾选「演示模式」可零 token 走通全流程。")
    print("  Ctrl+C 停止")
    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nbye.")
    finally:
        httpd.server_close()


if __name__ == "__main__":  # pragma: no cover
    serve()
