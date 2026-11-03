"""GUI 后端（HTTP 层 + Harness 门面）的端到端测试。

策略：真的起一个 ThreadingHTTPServer（端口 0 让内核分配），用 urllib 走真实 HTTP，
而不是绕过 Handler 直接调 Harness。理由是这个文件里最容易出错的恰恰是路由、序列化
和静态资源路径，绕过 HTTP 层就测不到它们。

全程离线：demo 运行由 DemoLLM 驱动，不碰 DeepSeek API。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from yaro.server import DEFAULT_DEMO_TASK, Harness, _bind
from yaro.config import Settings


# --------------------------------------------------------------------------- #
# 夹具
# --------------------------------------------------------------------------- #


def make_settings(tmp_path):
    return Settings(
        home=tmp_path / "home",
        workspace=tmp_path / "ws",
        model="deepseek-flash",
        model_plan="deepseek-v4-pro",
        api_key="",
        base_url="https://api.deepseek.com",
        sandbox_image="python:3.12-slim",
        allow_net=False,
        allow_unsafe_local=False,
    )


@pytest.fixture()
def server(tmp_path):
    settings = make_settings(tmp_path)
    settings.workspace.mkdir(parents=True, exist_ok=True)
    harness = Harness(settings)
    httpd = _bind("127.0.0.1", 0, harness)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield {"base": base, "harness": harness, "settings": settings}
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def call(base, path, *, method="GET", body=None, raw=False):
    """返回 (status, payload)；raw=True 时 payload 是 bytes。"""
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:  # 4xx/5xx 也要能读到 body
        payload = exc.read()
        status = exc.code
    if raw:
        return status, payload
    try:
        return status, json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, payload


def wait_idle(base, session_id, timeout=60.0):
    """轮询到运行结束，返回最终 session_state。"""
    deadline = time.time() + timeout
    state = {}
    while time.time() < deadline:
        _, state = call(base, f"/api/sessions/{session_id}?since=0")
        if not state.get("running", {}).get("active"):
            if any(event["type"] == "run_end" for event in state.get("events", [])):
                return state
        time.sleep(0.1)
    raise AssertionError(f"运行没有在 {timeout}s 内结束：{state}")


# --------------------------------------------------------------------------- #
# 静态资源 / 配置
# --------------------------------------------------------------------------- #


def test_static_files_are_served(server):
    base = server["base"]

    status, body = call(base, "/", raw=True)
    assert status == 200
    assert b"YARO" in body

    status, body = call(base, "/assets/app.css", raw=True)
    assert status == 200
    assert b"--bg" in body

    status, body = call(base, "/assets/app.js", raw=True)
    assert status == 200
    assert b"/api/sessions" in body

    status, body = call(base, "/assets/logo.svg", raw=True)
    assert status == 200
    assert b"<svg" in body


def test_unknown_static_is_404(server):
    status, payload = call(server["base"], "/assets/nope.js")
    assert status == 404
    assert "error" in payload


def test_config_exposes_environment_and_demo_assets(server):
    status, config = call(server["base"], "/api/config")
    assert status == 200
    assert config["model"] == "deepseek-flash"
    assert config["has_key"] is False
    assert config["workspace"] == str(server["settings"].workspace)
    # config 会顺手把 demo 图铺到工作区，否则首次 Demo 运行会失败
    assert config["demo"]["relative_dir"] == "demo"
    for name in config["demo"]["files"]:
        assert (server["settings"].workspace / "demo" / name).is_file()
    names = [tool["name"] for tool in config["tools"]]
    # 底座四个工具保持在最前，视觉链路工具随后追加（顺序稳定，便于前缀缓存）
    assert names[:4] == ["list_files", "image_probe", "calib_assess", "run_python"]
    for vision_tool in ("dataset_scan", "dataset_audit", "dataset_split", "dataset_render", "train_plan", "deploy_plan"):
        assert vision_tool in names, f"config.tools 缺少视觉工具 {vision_tool}"
    assert config["demo"]["dataset"] == "demo/dataset"


# --------------------------------------------------------------------------- #
# 会话生命周期
# --------------------------------------------------------------------------- #


def test_create_list_and_read_session(server):
    base = server["base"]

    status, created = call(base, "/api/sessions", method="POST", body={})
    assert status == 200
    session_id = created["session_id"]

    status, listing = call(base, "/api/sessions")
    assert status == 200
    ids = [row["session_id"] for row in listing["sessions"]]
    assert session_id in ids
    row = next(item for item in listing["sessions"] if item["session_id"] == session_id)
    assert row["events"] == 0
    assert row["verified"] is True
    assert row["running"] is False

    status, state = call(base, f"/api/sessions/{session_id}?since=0")
    assert status == 200
    assert state["events"] == []
    assert state["lineage"] == []
    assert state["verified"] is True
    assert state["running"]["active"] is False
    assert state["stats"]["turns"] == 0


def test_two_sessions_get_distinct_ids(server):
    base = server["base"]
    _, first = call(base, "/api/sessions", method="POST", body={})
    _, second = call(base, "/api/sessions", method="POST", body={})
    assert first["session_id"] != second["session_id"]


def test_unknown_session_is_404(server):
    status, payload = call(server["base"], "/api/sessions/does-not-exist?since=0")
    assert status == 404
    assert "error" in payload


# --------------------------------------------------------------------------- #
# Demo 运行（端到端，含产物与统计）
# --------------------------------------------------------------------------- #


def test_demo_run_produces_events_artifacts_and_stats(server):
    base = server["base"]
    call(base, "/api/config")  # 确保 demo 图已就位
    _, created = call(base, "/api/sessions", method="POST", body={})
    session_id = created["session_id"]

    status, started = call(
        base,
        f"/api/sessions/{session_id}/run",
        method="POST",
        body={"task": DEFAULT_DEMO_TASK, "demo": True},
    )
    assert status == 200
    assert started["demo"] is True

    state = wait_idle(base, session_id)
    types = [event["type"] for event in state["events"]]
    assert types[0] == "user"
    assert "assistant" in types
    assert "tool" in types
    assert types[-1] == "run_end"

    end = state["events"][-1]["payload"]
    assert end["stop_reason"] == "done"
    assert end["turn"] >= 1

    # 统计：成本与缓存命中率都来自 Usage 记账，不是模型自报
    assert state["stats"]["cost_usd"] > 0
    assert state["stats"]["hit_rate"] > 0.5
    assert state["stats"]["turns"] >= 4
    assert state["verified"] is True
    assert state["lineage"] and state["head"] == state["lineage"][-1]

    # 工具事件必须带结构化证据（GUI 靠它渲染图/表）
    tool_events = [event for event in state["events"] if event["type"] == "tool"]
    assert tool_events
    assert any(event["payload"].get("evidence") for event in tool_events)
    assert any(event["payload"].get("metrics") for event in tool_events)

    # 产物：dataset_render 会把标注渲染图落进 CAS，并且能通过 HTTP 取回原始字节
    _, artifacts = call(base, "/api/artifacts")
    assert artifacts["artifacts"], "demo 运行应该至少产生一个产物"
    pngs = [row for row in artifacts["artifacts"] if row["media_type"] == "image/png"]
    assert pngs
    status, raw = call(base, f"/api/artifact/{pngs[0]['sha256']}", raw=True)
    assert status == 200
    assert raw.startswith(b"\x89PNG\r\n\x1a\n")


def test_ensure_loop_is_persistent(server):
    """回归：agent 运行必须调度在单一常驻 loop 上（修 Event loop is closed）。

    旧的 ``asyncio.run`` 每次新建/关闭 loop，Windows proactor 下会与 httpx/httpcore
    连接池 abort 竞争，随机抛 RuntimeError: Event loop is closed。
    """
    harness = server["harness"]
    loop1 = harness._ensure_loop()
    loop2 = harness._ensure_loop()
    assert loop1 is loop2
    assert not loop1.is_closed()
    assert loop1.is_running()


def test_concurrent_demo_runs_all_complete(server):
    """回归：并发多次运行不因 loop 生命周期问题互相踩踏、不丢 run_end。"""
    base = server["base"]
    sids = []
    for _ in range(4):
        status, data = call(base, "/api/sessions", method="POST", body={})
        assert status == 200
        sids.append(data["session_id"])
    for sid in sids:
        status, _ = call(base, f"/api/sessions/{sid}/run", method="POST", body={"task": "hi", "demo": True})
        assert status == 200
    for sid in sids:
        state = wait_idle(base, sid)
        re = [e for e in state["events"] if e["type"] == "run_end"][-1]
        assert re["payload"]["stop_reason"] == "done"


def test_since_filter_returns_only_newer_events(server):
    base = server["base"]
    call(base, "/api/config")
    _, created = call(base, "/api/sessions", method="POST", body={})
    session_id = created["session_id"]
    call(base, f"/api/sessions/{session_id}/run", method="POST", body={"task": DEFAULT_DEMO_TASK, "demo": True})
    state = wait_idle(base, session_id)
    last_seq = state["seq"]

    status, tail = call(base, f"/api/sessions/{session_id}?since={last_seq}")
    assert status == 200
    assert tail["events"] == []
    # 而 lineage 与统计仍然是全量的：前端要靠它画时间线
    assert tail["lineage"] == state["lineage"]
    assert tail["stats"]["turns"] == state["stats"]["turns"]


# --------------------------------------------------------------------------- #
# 回滚 / 分叉 / 校验
# --------------------------------------------------------------------------- #


def test_rewind_moves_head_without_deleting_events(server):
    base = server["base"]
    call(base, "/api/config")
    _, created = call(base, "/api/sessions", method="POST", body={})
    session_id = created["session_id"]
    call(base, f"/api/sessions/{session_id}/run", method="POST", body={"task": DEFAULT_DEMO_TASK, "demo": True})
    state = wait_idle(base, session_id)
    events = state["events"]
    total = len(events)
    target = events[1]["event_id"]

    status, rewound = call(base, f"/api/sessions/{session_id}/rewind", method="POST", body={"event_id": target})
    assert status == 200
    assert rewound["heads"]["main"] == target

    _, after = call(base, f"/api/sessions/{session_id}?since=0")
    assert after["head"] == target
    assert len(after["events"]) == total, "回滚只移动指针，事件一条都不能少"
    assert after["lineage"][-1] == target
    assert after["verified"] is True


def test_fork_creates_branch_and_keeps_main_intact(server):
    base = server["base"]
    call(base, "/api/config")
    _, created = call(base, "/api/sessions", method="POST", body={})
    session_id = created["session_id"]
    call(base, f"/api/sessions/{session_id}/run", method="POST", body={"task": DEFAULT_DEMO_TASK, "demo": True})
    state = wait_idle(base, session_id)
    fork_point = state["events"][2]["event_id"]
    main_head = state["head"]

    status, forked = call(
        base,
        f"/api/sessions/{session_id}/fork",
        method="POST",
        body={"event_id": fork_point, "branch": "alt"},
    )
    assert status == 200
    assert set(forked["branches"]) == {"main", "alt"}
    assert forked["heads"]["main"] == main_head
    assert forked["heads"]["alt"] == fork_point

    _, report = call(base, f"/api/sessions/{session_id}/verify", method="POST", body={})
    assert report["ok"] is True


def test_tampered_session_is_reported(server):
    base = server["base"]
    call(base, "/api/config")
    _, created = call(base, "/api/sessions", method="POST", body={})
    session_id = created["session_id"]
    call(base, f"/api/sessions/{session_id}/run", method="POST", body={"task": DEFAULT_DEMO_TASK, "demo": True})
    wait_idle(base, session_id)

    path = server["settings"].sessions_dir / session_id / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 3
    tampered = json.loads(lines[1])
    tampered["payload"]["content"] = "被篡改的内容"
    lines[1] = json.dumps(tampered, ensure_ascii=False, sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _, report = call(base, f"/api/sessions/{session_id}/verify", method="POST", body={})
    assert report["ok"] is False
    assert report["problems"]


# --------------------------------------------------------------------------- #
# 错误路径与安全边界
# --------------------------------------------------------------------------- #


def test_run_without_api_key_and_not_demo_is_rejected(server):
    base = server["base"]
    _, created = call(base, "/api/sessions", method="POST", body={})
    status, payload = call(
        base,
        f"/api/sessions/{created['session_id']}/run",
        method="POST",
        body={"task": "做点事", "demo": False},
    )
    assert status == 400
    assert "DEEPSEEK_API_KEY" in payload["error"]


def test_api_key_set_and_clear_via_gui(server, monkeypatch):
    """GUI 的 Key 入口：写入本机 → config 立刻反映；响应/页面都不回显原文。

    monkeypatch 掉 keyring，避免测试写到真实系统的钥匙串里（统一落到临时 home 的 0600 文件）。
    """
    from yaro import secrets as secret_module

    monkeypatch.setattr(secret_module, "keyring_set", lambda value: False)
    monkeypatch.setattr(secret_module, "keyring_clear", lambda: False)

    base = server["base"]
    fake = "sk-test-1234567890abcdef"
    status, payload = call(base, "/api/key", method="POST", body={"action": "set", "key": fake})
    assert status == 200 and payload["ok"] is True
    assert fake not in json.dumps(payload), "响应里不能出现 Key 原文"
    assert payload["key"]["configured"] is True
    assert payload["key"]["masked"] != fake

    _, cfg = call(base, "/api/config")
    assert cfg["has_key"] is True
    assert cfg["key"]["configured"] is True

    status, payload = call(base, "/api/key", method="POST", body={"action": "clear"})
    assert status == 200
    _, cfg = call(base, "/api/config")
    assert cfg["has_key"] is False


def test_api_key_rejects_bad_input(server, monkeypatch):
    from yaro import secrets as secret_module

    monkeypatch.setattr(secret_module, "keyring_set", lambda value: False)
    base = server["base"]
    for body in (
        {"action": "set", "key": ""},
        {"action": "set", "key": "short"},
        {"action": "set", "key": "sk-has space inside"},
        {"action": "nonsense"},
    ):
        status, payload = call(base, "/api/key", method="POST", body=body)
        assert status == 400, body
        assert "error" in payload


def test_egress_switch_via_gui(server, tmp_path, monkeypatch):
    """GUI 一键改出网：mode 校验 → config 立即反映 → .env 持久化（不碰真实 cwd）。"""
    import yaro.config as config_module

    env_dir = tmp_path / "cwd"
    env_dir.mkdir(parents=True, exist_ok=True)
    original = config_module.update_egress_env

    def fake_update(mode, allow_public, *, directory=None):
        return original(mode, allow_public, directory=env_dir)

    monkeypatch.setattr(config_module, "update_egress_env", fake_update)

    base = server["base"]
    status, payload = call(base, "/api/egress", method="POST", body={"mode": "derived", "allow_public": True})
    assert status == 200, payload
    assert payload["ok"] is True
    assert payload["egress"]["mode"] == "derived"
    assert payload["egress"]["allow_public"] is True

    # config 立即反映（下一次运行就用新策略）
    _, cfg = call(base, "/api/config")
    assert cfg["egress"]["mode"] == "derived"
    assert cfg["egress"]["allow_public"] is True

    # .env 持久化：只改两行、其它键保留
    text = (env_dir / ".env").read_text(encoding="utf-8")
    assert "YARO_EGRESS=derived" in text
    assert "YARO_EGRESS_ALLOW_PUBLIC=1" in text


def test_egress_rejects_bad_mode(server, tmp_path, monkeypatch):
    import yaro.config as config_module

    monkeypatch.setattr(
        config_module, "update_egress_env",
        lambda mode, allow_public, *, directory=None: tmp_path / ".env",
    )
    base = server["base"]
    status, payload = call(base, "/api/egress", method="POST", body={"mode": "nonsense", "allow_public": False})
    assert status == 400
    assert "error" in payload


def test_update_egress_env_preserves_other_keys(tmp_path):
    from yaro.config import update_egress_env

    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=sk-test-abc\nYARO_MODEL=deepseek-flash\nYARO_EGRESS=text\n",
        encoding="utf-8",
    )
    path = update_egress_env("full", True, directory=tmp_path)
    assert path == env_file
    text = env_file.read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY=sk-test-abc" in text
    assert "YARO_MODEL=deepseek-flash" in text
    assert "YARO_EGRESS=full" in text
    assert "YARO_EGRESS_ALLOW_PUBLIC=1" in text
    assert text.count("YARO_EGRESS=") == 1  # 旧行被替换，不是追加


def test_update_egress_env_replaces_spaced_and_export_lines(tmp_path):
    """H1 回归：带空格的 KEY = value 与 export 前缀的旧行也必须被替换，否则降档 fail-open。"""
    from yaro.config import update_egress_env

    env_file = tmp_path / ".env"
    env_file.write_text(
        "DEEPSEEK_API_KEY=sk-test-abc\n"
        "YARO_EGRESS = full\n"
        "export YARO_EGRESS_ALLOW_PUBLIC = 1\n"
        "YARO_MODEL=deepseek-flash\n",
        encoding="utf-8",
    )
    update_egress_env("off", False, directory=tmp_path)
    text = env_file.read_text(encoding="utf-8")
    # 其它键保留
    assert "DEEPSEEK_API_KEY=sk-test-abc" in text
    assert "YARO_MODEL=deepseek-flash" in text
    # 旧的 full / 1（无论空格还是 export 格式）都被删掉，不会在重启后覆盖新值
    assert "YARO_EGRESS = full" not in text
    assert "YARO_EGRESS_ALLOW_PUBLIC = 1" not in text
    assert "export YARO_EGRESS" not in text
    # 新写的 canonical off / 0 各恰好一条
    assert "YARO_EGRESS=off" in text
    assert "YARO_EGRESS_ALLOW_PUBLIC=0" in text
    assert text.count("YARO_EGRESS=") == 1
    assert text.count("YARO_EGRESS_ALLOW_PUBLIC=") == 1


def test_update_egress_env_is_idempotent(tmp_path):
    """L1 回归：重复调用不堆积注释行/空行，键行各保持一条。"""
    from yaro.config import update_egress_env

    env_file = tmp_path / ".env"
    env_file.write_text("YARO_EGRESS=text\n", encoding="utf-8")
    for _ in range(3):
        update_egress_env("off", False, directory=tmp_path)
    text = env_file.read_text(encoding="utf-8")
    assert text.count("# 出网策略（GUI 写入）") == 1
    assert text.count("YARO_EGRESS=") == 1
    assert text.count("YARO_EGRESS_ALLOW_PUBLIC=") == 1


def test_resolve_env_path_prefers_cwd_then_root(tmp_path, monkeypatch):
    """M1 回归：_resolve_env_path 与 load_dotenv 查找顺序一致（cwd 优先，其次 ROOT）。"""
    import yaro.config as config_module

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(config_module, "ROOT", root)
    monkeypatch.chdir(cwd)

    # 都不存在 → 新建到 cwd/.env
    assert config_module._resolve_env_path() == cwd / ".env"

    # 只有 ROOT/.env 存在 → 写到 ROOT（不再新建 cwd/.env 遮蔽它）
    (root / ".env").write_text("YARO_MODEL=deepseek-flash\n", encoding="utf-8")
    assert config_module._resolve_env_path() == root / ".env"

    # cwd/.env 也存在 → cwd 优先（与 load_dotenv 第一命中一致）
    (cwd / ".env").write_text("YARO_EGRESS=off\n", encoding="utf-8")
    assert config_module._resolve_env_path() == cwd / ".env"


def test_empty_task_is_rejected(server):
    base = server["base"]
    _, created = call(base, "/api/sessions", method="POST", body={})
    status, payload = call(
        base,
        f"/api/sessions/{created['session_id']}/run",
        method="POST",
        body={"task": "   ", "demo": False},
    )
    assert status == 400
    assert "任务描述" in payload["error"]


def test_illegal_effort_is_rejected(server):
    base = server["base"]
    _, created = call(base, "/api/sessions", method="POST", body={})
    status, payload = call(
        base,
        f"/api/sessions/{created['session_id']}/run",
        method="POST",
        body={"task": "x", "demo": True, "effort": "turbo"},
    )
    assert status == 400
    assert "effort" in payload["error"]


def test_artifact_requires_valid_sha(server):
    status, payload = call(server["base"], "/api/artifact/not-a-sha")
    assert status == 400
    assert "sha256" in payload["error"]

    status, payload = call(server["base"], "/api/artifact/" + "0" * 64)
    assert status == 404


def test_gates_rejects_path_escape(server):
    status, payload = call(server["base"], "/api/gates?path=../../etc")
    assert status == 400
    assert "越界" in payload["error"]


def test_gates_scans_workspace(server):
    base = server["base"]
    workspace = server["settings"].workspace
    (workspace / "bad.py").write_text(
        "import ultralytics\nimport pickle\npickle.load(open('/tmp/x','rb'))\n",
        encoding="utf-8",
    )
    status, result = call(base, "/api/gates?path=bad.py")
    assert status == 200
    assert result["ok"] is False
    rules = {finding["rule"] for finding in result["findings"]}
    assert rules, result


def test_doctor_reports_checks(server):
    status, payload = call(server["base"], "/api/doctor")
    assert status == 200
    assert payload["ok"] in (True, False)
    assert payload["checks"]
    assert any(check["check"] == "workspace" for check in payload["checks"])


def test_stop_without_run_is_409(server):
    base = server["base"]
    _, created = call(base, "/api/sessions", method="POST", body={})
    status, payload = call(base, f"/api/sessions/{created['session_id']}/stop", method="POST", body={})
    assert status == 409
    assert "error" in payload


def test_unknown_api_is_404(server):
    status, payload = call(server["base"], "/api/nope")
    assert status == 404
    assert "error" in payload


def test_options_preflight_returns_204_not_501(server):
    """OPTIONS 预检不再返回 501（基类 Unsupported method），而是 204 + 本机 Origin 回显。"""
    base = server["base"]
    request = urllib.request.Request(
        base + "/api/sessions",
        data=None,
        headers={"Origin": "http://127.0.0.1:8765"},
        method="OPTIONS",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        assert response.status == 204
        assert response.headers.get("Access-Control-Allow-Origin") == "http://127.0.0.1:8765"
        assert "POST" in (response.headers.get("Access-Control-Allow-Methods") or "")


def test_options_preflight_rejects_foreign_origin(server):
    """非本机 Origin 的预检仍被 _guard 拒（403），不因加了 OPTIONS 而放开跨源。"""
    base = server["base"]
    request = urllib.request.Request(
        base + "/api/sessions",
        data=None,
        headers={"Origin": "http://evil.example.com"},
        method="OPTIONS",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(request, timeout=20)
    assert exc.value.code == 403
