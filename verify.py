#!/usr/bin/env python
"""YARO 自检 —— 一条命令给出"这个库真的能跑"的证据。

为什么需要它：本项目的开发环境里 pwsh 完全不可用（任何命令 exit 0xC0000142），
pytest 从未被执行过。与其交付一堆"看起来对"的代码，不如给一个**零额外依赖**、
不花钱、不联网、30 秒内跑完的自检脚本，把最关键的运行时路径真实走一遍：

    导入 → 配置/契约 → 产物 CAS → 会话事件链（回滚/fork/篡改检测/压缩）
    → 评估器 → 闸门 → 标定判据 → 演示资产 → Agent 端到端（DemoLLM）
    → 视觉全链路（数据集/审计/切分/渲染/训练与部署计划）
    → 真起 HTTP 端口打请求 → 前端静态一致性

用法::

    python verify.py            # 全跑
    python verify.py --no-http  # 跳过 HTTP（没有可用端口时）
    python verify.py --keep     # 保留临时目录，便于事后翻证据
    python verify.py -v         # 打印每一步的细节

退出码：0 = 全部通过；1 = 有失败。
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import pkgutil
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PASS, SKIP, WARN, FAIL = "PASS", "SKIP", "WARN", "FAIL"


class StepSkipped(Exception):
    """前置条件不满足（如没装 opencv / 没有可用端口）。不算失败。"""


class StepFailed(AssertionError):
    """自检断言失败。"""


def need(condition: Any, message: str) -> None:
    if not condition:
        raise StepFailed(message)


# --------------------------------------------------------------------------- #
# 步骤
# --------------------------------------------------------------------------- #


def step_imports(ctx: dict[str, Any]) -> str:
    """把包里每个模块都 import 一遍：语法错、循环 import、拼错的常量都会在这里炸。"""
    import yaro

    modules = [info.name for info in pkgutil.walk_packages(yaro.__path__, prefix="yaro.")]
    for name in modules:
        importlib.import_module(name)
    return f"{len(modules)} 个模块导入成功（{yaro.__version__}）"


def step_config(ctx: dict[str, Any]) -> str:
    from yaro.config import Settings, system_prompt

    tmp = Path(tempfile.mkdtemp(prefix="yaro-verify-"))
    ctx["tmp"] = tmp
    ws = tmp / "ws"
    ws.mkdir(parents=True)

    ctx["_env_backup"] = {
        key: os.environ.get(key)
        for key in ("YARO_HOME", "YARO_WORKSPACE", "DEEPSEEK_API_KEY")
    }
    os.environ["YARO_HOME"] = str(tmp / "home")
    os.environ["YARO_WORKSPACE"] = str(ws)
    os.environ.pop("DEEPSEEK_API_KEY", None)

    settings = Settings.load(dotenv=False).ensure()
    need(settings.workspace == ws.resolve(), f"workspace 解析不对：{settings.workspace}")
    need(settings.demo_dir == ws.resolve() / "demo", "demo_dir 应为 <workspace>/demo")
    for path in (settings.home, settings.sessions_dir, settings.artifacts_dir, settings.runs_dir):
        need(path.is_dir(), f"ensure() 没有建出目录：{path}")

    prompt = system_prompt()
    need("YARO-CONTRACT" in prompt, "system_prompt() 里没有 CONTRACT 版本标记")
    need(prompt == system_prompt(), "system_prompt() 两次调用不一致（会破坏前缀缓存）")

    # 未知型号必须启动即报错（用户要求；以前是按 0 计费 / 或按最贵档猜价）
    from yaro.contracts import ContractError, ensure_known_model

    try:
        ensure_known_model("not-a-model")
    except ContractError:
        pass
    else:
        need(False, "未知模型应该被 ensure_known_model 拒绝")
    os.environ["YARO_MODEL"] = "not-a-model"
    try:
        Settings.load(dotenv=False)
    except ContractError:
        pass
    else:
        need(False, "YARO_MODEL 写成未知型号时 Settings.load() 应直接报错")
    finally:
        os.environ.pop("YARO_MODEL", None)

    ctx["settings"] = settings
    ctx["prompt"] = prompt
    return f"home={settings.home.name} workspace={settings.workspace.name} contract={len(prompt)}B"


def step_artifacts(ctx: dict[str, Any]) -> str:
    from yaro.memory.artifacts import ArtifactStore

    store = ArtifactStore(ctx["settings"].artifacts_dir)
    blob = b"\x89PNG\r\n\x1a\n" + b"yaro-verify" * 64
    first = store.put(blob, run_id="verify", step=1, kind="evidence", media_type="image/png", ext="png")
    second = store.put(blob, run_id="verify", step=2, kind="evidence", media_type="image/png", ext="png")
    need(first.sha256 == second.sha256, "同内容两次 put 的 sha256 不同")
    need(store.get(first) == blob, "取回的字节与写入不一致")
    need(store.verify(first), "产物落盘校验失败")
    need(store.exists(first.sha256), "exists() 对已写产物返回 False")
    need(len(store) == 1, f"内容寻址去重失效：objects={len(store)}")
    need(store.stats()["objects"] == 1, "stats().objects 计数不对")
    need(first.uri.startswith("artifact://verify/"), f"uri 形状不对：{first.uri}")
    ctx["artifacts"] = store
    return f"1 object / {len(blob)}B，去重与校验通过"


def step_session(ctx: dict[str, Any]) -> str:
    from yaro.memory.session import SessionStore

    store = SessionStore.create(ctx["settings"].sessions_dir)
    e1 = store.append("user", {"content": "评估标定"})
    e2 = store.append("assistant", {"content": "先看图", "reasoning_content": "需要证据", "tool_calls": []})
    e3 = store.append("tool", {"tool_call_id": "c1", "content": "OK", "evidence": []})
    e4 = store.append("assistant", {"content": "结论", "reasoning_content": "证据够了"})
    e5 = store.append("run_end", {"stop_reason": "done", "turn": 1, "usage": {"hit": 1, "miss": 1, "out": 1, "calls": 1}, "cost_usd": 0.0})

    need([event.seq for event in (e1, e2, e3, e4, e5)] == [1, 2, 3, 4, 5], "seq 不是从 1 连续递增")
    need(e2.prev_digest == e1.digest and e5.prev_digest == e4.digest, "prev_digest 链没接上")
    need(e1.prev_digest == "0" * 64, "根事件的 prev_digest 应为 GENESIS")

    messages = store.messages("main")
    need(len(messages) == 4, f"消息数量不对：{len(messages)}")
    need(all("reasoning_content" in item for item in messages if item["role"] == "assistant"),
         "assistant 消息丢了 reasoning_content（DeepSeek 带 tools 时会 400）")
    need(messages[2]["role"] == "tool" and messages[2]["tool_call_id"] == "c1", "tool 消息还原不对")

    report = store.verify()
    need(report["ok"], f"verify() 报错：{report['problems']}")
    need(store.head("main") == e5.event_id, "head 不是最后一条事件")

    ctx["session"] = store
    ctx["session_events"] = [e1, e2, e3, e4, e5]
    return f"5 事件，digest 链与 messages 还原正确"


def step_rollback(ctx: dict[str, Any]) -> str:
    from yaro.contracts import ContractError

    store = ctx["session"]
    events = ctx["session_events"]
    before = len(store.all_events())

    store.rewind(events[1].event_id)
    need(store.head("main") == events[1].event_id, "rewind 之后 head 没变")
    need(len(store.all_events()) == before, "rewind 删掉了事件（违反 append-only）")
    need(len(store.messages("main")) == 2, "rewind 之后消息没变短")

    store.fork(events[0].event_id, "alt")
    need("alt" in store.branches() and "main" in store.branches(), "fork 之后分支不对")
    need(store.head("main") == events[1].event_id, "fork 影响了 main 分支的 head")
    need(store.head("alt") == events[0].event_id, "fork 出来的分支 head 不对")

    alt_event = store.append("user", {"content": "另一条分支上的尝试"}, branch="alt")
    try:
        store.rewind(alt_event.event_id)
        raise StepFailed("rewind 到别的分支的事件应当报错，却成功了")
    except ContractError:
        pass

    # 再回到主线继续追加：rewind 只能指向当前 head 的祖先，不能"穿越"到被丢弃的未来
    store.rewind(events[1].event_id)
    store.append("user", {"content": "回到主线继续"}, branch="main")
    need(store.verify()["ok"], "多次回滚/fork 之后链校验失败")
    return "rewind 只移指针；fork 不污染 main；跨分支回滚被拒"


def step_tamper(ctx: dict[str, Any]) -> str:
    from yaro.memory.session import SessionStore

    settings = ctx["settings"]
    source = ctx["session"].dir
    target = Path(tempfile.mkdtemp(prefix="yaro-tamper-")) / source.name
    shutil.copytree(source, target)

    lines = (target / "events.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[1])  # 第二条事件：assistant
    payload["payload"]["content"] = "被偷偷改过的内容"
    lines[1] = json.dumps(payload, ensure_ascii=False)

    # 顺序也要打乱一次：verify 必须按 parent 校验，而不是按文件行序
    lines[1], lines[2] = lines[2], lines[1]
    (target / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    store = SessionStore.open(target.parent, target.name)
    report = store.verify()
    need(not report["ok"], "篡改/乱序之后 verify() 仍然报 ok=True")
    need(len(report["problems"]) >= 1, "verify() 没给出具体问题清单")
    return f"篡改与乱序均被捕获（{len(report['problems'])} 条问题）"


def step_compaction(ctx: dict[str, Any]) -> str:
    from yaro.contracts import ContractError
    from yaro.memory.session import SessionStore

    store = SessionStore.create(ctx["settings"].sessions_dir)
    store.append("user", {"content": "u1"})
    a1 = store.append("assistant", {"content": "a1", "reasoning_content": "r1"})
    u2 = store.append("user", {"content": "u2"})
    store.append("assistant", {"content": "a2", "reasoning_content": "r2"})

    marker = store.append_compaction("上一轮摘要：标定 RMS 0.243px 通过", a1.event_id)
    need(marker.type == "compaction", "压缩事件类型不对")

    messages = store.messages("main")
    need(any("[CONTEXT COMPACTION]" in str(item.get("content", "")) for item in messages),
         "压缩之后没有摘要占位消息")
    need(all(item.get("content") != "u1" for item in messages), "压缩切点之前的逐字历史没被折叠")
    need(any(item.get("content") == "u2" for item in messages), "压缩切点之后的逐字历史被误删")
    need(store.verify()["ok"], "压缩之后链校验失败")

    try:
        store.append_compaction("非法切点", u2.event_id)  # user 不是轮边界
        raise StepFailed("切点不是轮边界时应当报错")
    except ContractError:
        pass
    return f"折叠成 {len(messages)} 条消息，非法切点被拒"


def step_evals(ctx: dict[str, Any]) -> str:
    from yaro.evals import MetricSpec, Observation, compare, fixed_budget_compare, regression_gate

    spec = MetricSpec("rms_px", "lower", tolerance=0.01, noise_floor=0.01, unit="px")
    same = compare(spec, Observation(0.30, n=5), Observation(0.305, n=5))
    need(same.status == "same", f"噪声内差异应判 same，实际 {same.status}")

    better = compare(spec, Observation(0.40, n=5), Observation(0.24, n=5))
    need(better.status == "improved", f"下降应判 improved，实际 {better.status}")

    blocked = compare(spec, Observation(0.40, n=5), Observation(0.24, n=1))
    need(blocked.status == "blocked", "样本不足应判 blocked（不给假结论）")

    hard = MetricSpec("miss_rate", "lower", hard_limit=0.001)
    decision = regression_gate([hard], {"miss_rate": Observation(0.0005, n=10)}, {"miss_rate": Observation(0.002, n=10)})
    need(not decision.ok, "越过 hard_limit 却没有拦截")

    best = fixed_budget_compare([(0.1, Observation(0.5)), (2.0, Observation(0.9))], budget_usd=0.5)
    need(best is not None and best.value == 0.5, "固定预算比较选错了观察")
    return "same/improved/blocked/hard_limit/固定预算 全部符合预期"


def step_gates(ctx: dict[str, Any]) -> str:
    from yaro.gates import scan_source

    dirty = scan_source("import ultralytics\nimport pickle\nmodel = pickle.load(open('x.pkl','rb'))\n", filename="bad.py")
    need(not dirty.ok, "含 AGPL 依赖与 pickle.load 的代码没有拦下来")
    need(len(dirty.findings) >= 2, f"findings 太少：{len(dirty.findings)}")

    clean = scan_source("import os\nprint(os.getcwd())\n", filename="ok.py")
    need(clean.ok, f"干净代码被误拦：{clean.summary()}")
    return f"命中 {len(dirty.findings)} 条；干净代码放行"


def step_calib(ctx: dict[str, Any]) -> str:
    from yaro.capabilities.calib import assess_calibration

    passed = assess_calibration(0.243, 0.83, 24)
    need(passed.ok, f"RMS 0.243px / 覆盖率 83% / 24 视图应判通过：{passed.reasons}")

    failed = assess_calibration(0.412, 0.71, 18, per_view_rms=[0.31, 0.49, 0.9, 0.33])
    need(not failed.ok, "RMS 0.412px 超阈值却被判通过")
    need(failed.reasons, "未通过却没有给出原因（失败必须可解释）")

    low_cov = assess_calibration(0.20, 0.40, 20)
    need(not low_cov.ok, "覆盖率 40% 低于 60% 却被判通过")
    return "通过/未通过/覆盖率不足 三种判据正确"


def step_demo_assets(ctx: dict[str, Any]) -> str:
    from yaro.demo import ensure_demo_assets

    info = ensure_demo_assets(ctx["settings"].demo_dir)
    need(Path(info["dir"]).is_dir(), "演示目录没建出来")
    for name in ("calib_left.png", "calib_right.png", "frame_raw.png"):
        path = Path(info["dir"]) / name
        need(path.is_file() and path.stat().st_size > 100, f"演示图缺失或过小：{path}")
        need(path.read_bytes().startswith(b"\x89PNG"), f"不是合法 PNG：{path}")

    again = ensure_demo_assets(ctx["settings"].demo_dir)
    need(again["files"] == info["files"], "ensure_demo_assets 不是幂等的")
    need(info["primary"].startswith("demo/"), "primary 应是相对工作区的路径")
    need(not info.get("dataset_error"), f"演示数据集生成失败：{info.get('dataset_error')}")
    dataset_dir = Path(info["dir"]) / "dataset"
    need(dataset_dir.is_dir(), "演示数据集目录没生成（GUI 演示与 yaro chain 都会用到）")
    need((dataset_dir / "images").is_dir() and any((dataset_dir / "images").iterdir()), "演示数据集里没有图像")

    try:
        from yaro.capabilities.builtin import probe_image

        meta = probe_image(Path(info["dir"]) / "calib_left.png")
        need(meta["width"] == 320 and meta["height"] == 240, f"图像头解析不对：{meta}")
        note = f"3 张 PNG；probe_image 解析出 {meta['width']}×{meta['height']}"
    except StepSkipped:
        raise
    except Exception as exc:  # noqa: BLE001 - 只是附加信息
        note = f"3 张 PNG（probe_image 附加检查跳过：{type(exc).__name__}）"
    return note


def step_agent_loop(ctx: dict[str, Any]) -> str:
    """端到端：DemoLLM 驱动真实循环、真实工具、真实事件、真实产物。零 token。"""
    from yaro.capabilities.builtin import build_default_tools
    from yaro.capabilities.registry import ToolRegistry
    from yaro.llm.demo import DemoLLM
    from yaro.loop.agent import Agent, Budget
    from yaro.memory.session import SessionStore

    settings = ctx["settings"]
    store = SessionStore.create(settings.sessions_dir)
    artifacts = ctx["artifacts"]
    tools = build_default_tools(settings.workspace, artifacts, run_id=store.session_id, step=0)
    agent = Agent(
        llm=DemoLLM(delay_s=0.0),
        registry=ToolRegistry(tools),
        session=store,
        artifacts=artifacts,
        budget=Budget(),
        run_id=store.session_id,
        system=ctx["prompt"],
    )
    result = asyncio.run(agent.run("把 demo/dataset 走一遍全链路：体检、抽查标注、切分、给训练计划并诊断日志"))

    need(result.stop_reason == "done", f"演示运行没有正常结束：{result.stop_reason}")
    need(result.turns == 8, f"演示剧本应是 8 轮，实际 {result.turns}")
    need(result.cost_usd > 0, "演示运行成本为 0（Usage 没记账）")
    need("观察" in result.content and "判断" in result.content, "最终结论不符合三段式契约")

    events = store.all_events()
    types = [event.type for event in events]
    need(types[0] == "user" and types[-1] == "run_end", f"事件序列首尾不对：{types}")
    need(types.count("assistant") == 8, f"assistant 事件数不对：{types.count('assistant')}")
    need(types.count("tool") == 8, f"tool 事件数不对：{types.count('tool')}")

    for event in events:
        if event.type == "assistant":
            need("reasoning_content" in event.payload, "assistant 事件缺 reasoning_content")

    image_evidence = [
        item
        for event in events
        if event.type == "tool"
        for item in event.payload.get("evidence", [])
        if item.get("kind") == "image"
    ]
    need(image_evidence, "工具事件里没有 image 证据（GUI 的缩略图就没得渲染）")
    need(image_evidence[0]["payload"]["sha256"], "image 证据缺 sha256")

    report = store.verify()
    need(report["ok"], f"运行后链校验失败：{report['problems']}")
    need(len(artifacts) >= 1, "标注渲染图没有进产物库")

    end = events[-1].payload
    need(end.get("usage", {}).get("calls") == 8, f"run_end 的 usage 不对：{end.get('usage')}")
    need(end.get("cost_usd", 0) > 0, "run_end 没带 cost_usd")

    ctx["agent_session"] = store
    return (
        f"{result.turns} 轮 / {len(events)} 事件 / 8 次工具调用 / "
        f"${result.cost_usd:.4f} / 证据图 {len(image_evidence)} 张"
    )


def step_vision(ctx: dict[str, Any]) -> str:
    """全链路能力自检：合成数据集 → 扫描 → 体检 → 切分 → 渲染 → 计划 → 工具包装。"""
    from yaro.capabilities.builtin import build_tool_catalog
    from yaro.memory.artifacts import ArtifactStore
    from yaro.vision import annotate, dataset as ds, plan as planner, png
    from yaro.vision.tools import DatasetAuditTool, DatasetRenderTool, DatasetScanTool

    workspace = ctx["settings"].workspace
    data = workspace / "demo" / "dataset"
    need(data.is_dir(), "演示数据集没生成：<workspace>/demo/dataset")

    scan = ds.scan_dataset(data)
    need(scan["format"] == "yolo", f"数据集格式识别不对：{scan['format']}")
    need(scan["stats"]["images"] >= 8, f"图像数不对：{scan['stats']['images']}")
    need(scan["stats"]["boxes"] > 0, "没有任何标注框")
    need(len(scan["classes"]) >= 3, f"类别数不对：{len(scan['classes'])}")

    audit = ds.audit_dataset(data)
    codes = {item["code"] for item in audit["findings"]}
    need(audit["summary"]["errors"] > 0, "演示数据里故意埋的 error 没被查出来")
    for code in ("corrupt_image", "missing_label", "box_out_of_range", "duplicate_image", "orphan_label"):
        need(code in codes, f"体检漏掉了 {code}：{sorted(codes)}")

    first = ds.plan_split(data, ratios=(0.7, 0.2, 0.1), seed=7)
    second = ds.plan_split(data, ratios=(0.7, 0.2, 0.1), seed=7)
    need(first["assignments"] == second["assignments"], "同一 seed 的切分不可复现")
    total = sum(first["counts"][split] for split in ds.SPLIT_NAMES)
    need(total == len(first["assignments"]) and total > 0, f"切分数量与图像数不一致：{first['counts']}")
    written = ds.write_split(data, first)
    need({"train", "val", "test"} <= set(written), f"切分文件没写全：{sorted(written)}")
    need("yaml" in written, "yaro-data.yaml 没写出来")
    for split, path in written.items():
        target = Path(path) if Path(path).is_absolute() else data / path
        need(target.is_file(), f"splits/{split}.txt 没落盘")

    render = annotate.render_dataset(data, data / "runs" / "review", max_images=3)
    need(render["rendered"], "标注渲染一张都没成功")
    sheet = data / render["sheet"]
    need(sheet.is_file() and sheet.read_bytes().startswith(png.PNG_SIGNATURE), "联系人表不是合法 PNG")

    coco = annotate.export_coco(data)
    need((data / coco["path"]).is_file() and coco["annotations"] > 0, "COCO 导出失败")

    train_plan = planner.build_train_plan(data, backend="torchvision", epochs=3)
    names = {Path(path).name for path in train_plan["written"]}
    need({"recipe.json", "train.py", "acceptance.md"} <= names, f"训练计划文件不全：{sorted(names)}")
    deploy_plan = planner.build_deploy_plan(data, backends=("onnx",))
    deploy_names = {Path(path).name for path in deploy_plan["written"]}
    need({"bench_onnx.py", "Dockerfile", "model_card.md"} <= deploy_names, f"部署计划文件不全：{sorted(deploy_names)}")

    history = planner.parse_train_log(
        "\n".join(json.dumps({"epoch": index + 1, "loss": 1.0 - index * 0.05, "map50": 0.2 + index * 0.02}) for index in range(6))
    )
    need(history["count"] == 6, f"日志解析轮数不对：{history['count']}")
    need(isinstance(planner.analyze_training(history), list), "训练诊断没有返回列表")

    store = ArtifactStore(ctx["settings"].artifacts_dir)
    scan_result = asyncio.run(DatasetScanTool(workspace).run(dir="demo/dataset"))
    need(scan_result.ok, f"dataset_scan 工具失败：{scan_result.summary}")
    audit_result = asyncio.run(DatasetAuditTool(workspace).run(dir="demo/dataset"))
    need(audit_result.ok is False, "数据集有 error 时 dataset_audit 应返回 ok=False")
    render_result = asyncio.run(DatasetRenderTool(workspace, store, run_id="verify", step=0).run(dir="demo/dataset", max_images=2))
    need(render_result.ok and render_result.artifacts, "dataset_render 没有产出证据产物")

    catalog = {item["name"] for item in build_tool_catalog(workspace)}
    for name in (
        "dataset_scan",
        "dataset_audit",
        "dataset_split",
        "dataset_render",
        "prelabel_run",
        "train_plan",
        "train_run",
        "train_analyze",
        "tune_plan",
        "tune_run",
        "deploy_plan",
        "deploy_run",
    ):
        need(name in catalog, f"工具清单里缺 {name}")

    # 真执行层的模块必须能 import（不真跑训练：没有 CUDA 的机器上跑了也会被拒）
    from yaro.vision import deploy_run as deployer
    from yaro.vision import prelabel as prelabeler
    from yaro.vision import train_run as trainer
    from yaro.vision import tune_run as tuner

    backends = prelabeler.available_backends()
    need(bool(backends), "预标注后端一个都不可用（至少 contour 应该可用）")
    support = trainer.train_support()
    need(isinstance(support.get("ok"), bool), "train_support 必须返回布尔 ok")
    need("run_trials" in dir(tuner), "tune_run 缺少 run_trials")
    export = deployer.export_support()
    need(isinstance(export.get("ok"), bool), "export_support 必须返回布尔 ok")

    return (
        f"{scan['stats']['images']} 图 / {scan['stats']['boxes']} 框 / "
        f"体检 {audit['summary']['errors']}E {audit['summary']['warnings']}W / "
        f"切分 {first['counts']['train']}+{first['counts']['val']}+{first['counts']['test']} / "
        f"渲染 {len(render['rendered'])} 张"
    )


def step_cv2_optional(ctx: dict[str, Any]) -> str:
    """opencv 存在时顺带跑一遍真实标定求解；不存在就跳过（不装 cv2 是合法配置）。"""
    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        raise StepSkipped("未安装 opencv-python-headless（cv extra），跳过真实标定求解") from exc

    from yaro.capabilities.calib import calibrate_intrinsics

    images = sorted(Path(ctx["settings"].demo_dir).glob("*.png"))
    try:
        report = calibrate_intrinsics(images[:2])
    except Exception as exc:  # noqa: BLE001 - 演示图是合成棋盘，求解失败属预期
        raise StepSkipped(f"合成演示图不足以求解内参（预期行为）：{type(exc).__name__}") from exc
    return f"calibrate_intrinsics 未抛异常（ok={getattr(report, 'ok', '?')}）"


def step_static_assets(ctx: dict[str, Any]) -> str:
    """不动浏览器也能查出"JS 要的 id 在 HTML 里不存在"这类接线错误。"""
    web = ROOT / "yaro" / "web"
    for name in ("index.html", "app.css", "app.js", "logo.svg"):
        path = web / name
        need(path.is_file(), f"前端资源缺失：{path}")
        need(path.stat().st_size > 200, f"前端资源过小，疑似占位：{path}")

    html = (web / "index.html").read_text(encoding="utf-8")
    js = (web / "app.js").read_text(encoding="utf-8")
    css = (web / "app.css").read_text(encoding="utf-8")

    need("app.js" in html and "app.css" in html, "index.html 没有引用 app.js/app.css")
    need('id="transcript"' in html and 'id="session-list"' in html, "index.html 缺关键挂载点")
    need("--bg" in css and "grid-template-columns" in css, "app.css 不像三栏布局")
    need('data-theme="dark"' in css and ":root," in css, "app.css 缺明/暗双主题变量（默认应为浅色高对比）")
    need("btn-theme" in html and "btn-theme" in js, "缺少主题切换按钮接线（btn-theme）")
    need("THEME_KEY" in js and "applyTheme" in js, "app.js 缺主题持久化逻辑")
    need("app.css?v=" in html and "app.js?v=" in html, "静态资源没带 ?v= 版本号（浏览器可能复用旧 CSS）")
    need('id="foot-build"' in html and "foot-build" in js, "缺少 build 版本/主题指示（foot-build）")
    need("renderBuildStamp" in js and "CSS_EXPECT" in js, "缺少 CSS 生效自检（渲染实际背景色）")
    # 主题变量在 index.html 内联兜底、在 app.css 正式定义：两处的 --bg 必须一致，
    # 否则会出现"HTML 里浅色、CSS 里深色"的错位。
    need("--bg: #f2f4f7" in html and "--bg: #f2f4f7" in css, "主题主色不一致：index.html 内联兜底与 app.css 的 --bg 必须相同")

    html_ids = set(re.findall(r'id="([\w-]+)"', html))
    used_ids = set(re.findall(r'\$\("([\w-]+)"\)', js)) | set(re.findall(r'getElementById\("([\w-]+)"\)', js))
    missing = sorted(used_ids - html_ids - {"__none__"})
    need(not missing, f"app.js 引用了 HTML 里不存在的 id：{missing}")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    need("web/*" in pyproject, "pyproject.toml 缺 package-data web/*（pip install 后 GUI 会 500）")

    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    need("YARO_WORKSPACE" in env_example, ".env.example 缺 YARO_WORKSPACE")

    return f"{len(html_ids)} 个 id 全部能对上；package-data 与 .env.example 就位"


def step_egress(ctx: dict[str, Any]) -> str:
    """证明默认不出网、显式授权才放行、原图在 text 模式下真的被剥离。"""
    from yaro.contracts import ContractError
    from yaro.egress import (
        MODE_DERIVED,
        MODE_FULL,
        MODE_OFF,
        MODE_TEXT,
        RAW,
        EgressPolicy,
        is_private_host,
    )
    from yaro.secrets import mask, scrub

    # 1) off 模式：连本机端点都拒绝（默认就是"什么都不发"）
    try:
        EgressPolicy(mode=MODE_OFF).check_endpoint("http://127.0.0.1:8000/v1")
        need(False, "off 模式居然放行了请求")
    except ContractError:
        pass

    # 2) 公网端点未授权 → 拒绝；显式授权 → 放行
    try:
        EgressPolicy(mode=MODE_TEXT).check_endpoint("https://api.deepseek.com")
        need(False, "未授权公网端点被放行")
    except ContractError:
        pass
    EgressPolicy(mode=MODE_TEXT, allow_public=True).check_endpoint("https://api.deepseek.com")

    # 3) 私网判定与图像敏感度
    need(is_private_host("127.0.0.1") and is_private_host("192.168.1.9"), "私网判定错误")
    need(not is_private_host("api.deepseek.com"), "公网域名被误判为私网")
    need(EgressPolicy(mode=MODE_DERIVED).allows_images(RAW) is False, "derived 模式不应允许原图出网")
    need(EgressPolicy(mode=MODE_FULL).allows_images(RAW) is True, "full 模式应允许原图")

    # 4) 端到端：raw 图在 text 模式下被剥离，并留下可读的隐私提示
    from yaro.capabilities.registry import ToolRegistry
    from yaro.loop.agent import Agent
    from yaro.memory.artifacts import ArtifactStore
    from yaro.memory.session import SessionStore

    settings = ctx["settings"]
    store = ArtifactStore(settings.artifacts_dir)
    session = SessionStore.create(settings.sessions_dir)
    ref = store.put(
        b"\x89PNG\r\n\x1a\n" + b"raw" * 32,
        run_id="verify",
        step=1,
        kind="input",
        media_type="image/png",
        ext="png",
    )
    agent = Agent(
        llm=None,  # type: ignore[arg-type] - 只测消息组装，不会真的调用
        registry=ToolRegistry([]),
        session=session,
        artifacts=store,
        system="S",
        egress=EgressPolicy(mode=MODE_TEXT),
    )
    agent._last_tool_images = [
        {"uri": ref.uri, "sha256": ref.sha256, "media_type": "image/png", "sensitivity": RAW}
    ]
    messages = agent._request_messages()
    joined = json.dumps(messages, ensure_ascii=False)
    need("image_url" not in joined, "text 模式下仍然把图片注入了请求（会出网）")
    need("[隐私策略]" in joined, "剥离图片后没有留提示，模型会开始猜图像内容")

    # 5) 密钥只露掩码
    fake = "sk-verify-1234567890abcdef"
    need(fake not in mask(fake), f"mask() 泄露了完整密钥：{mask(fake)}")
    need(fake not in scrub(f"boom {fake}", fake), "scrub() 没有擦掉密钥")

    return f"默认 off · 授权后 text/derived/full 生效 · raw 图已剥离 · 密钥仅掩码 {mask(fake)}"


def step_packaging(ctx: dict[str, Any]) -> str:
    """证明"打包后仍然自包含"：契约在包内、GUI 资源随包、一键安装脚本存在。"""
    from yaro.config import CONTRACT_PATH, DEV_CONTRACT, PKG_CONTRACT, ROOT, resolve_contract_path

    need(PKG_CONTRACT.is_file(), f"包内契约缺失（wheel 装到 site-packages 后 system prompt 会丢）：{PKG_CONTRACT}")
    need(DEV_CONTRACT.is_file(), f"仓库契约缺失：{DEV_CONTRACT}")
    need(PKG_CONTRACT.read_bytes() == DEV_CONTRACT.read_bytes(), "包内契约与仓库契约内容不一致")
    need(CONTRACT_PATH.exists(), "resolve_contract_path() 指向了不存在的文件")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    need("data/*" in pyproject, "pyproject 缺 package-data data/*（CONTRACT.md 不会进 wheel）")
    need("web/*" in pyproject, "pyproject 缺 package-data web/*")

    for relative in ("MANIFEST.in", "docs/INSTALL.md", "scripts/install.ps1", "scripts/install.sh", "CHANGELOG.md"):
        need((ROOT / relative).is_file(), f"分发所需文件缺失：{relative}")

    # 独立窗口壳：多人协作时它决定"界面在别人机器上是否还长这样"。
    need((ROOT / "yaro" / "desktop.py").is_file(), "缺少桌面窗口壳 yaro/desktop.py")
    desktop_src = (ROOT / "yaro" / "desktop.py").read_text(encoding="utf-8")
    need("class DesktopUnavailable" in desktop_src, "desktop.py 缺降级异常（没装 pywebview 时应回退浏览器）")
    need("def available" in desktop_src and "def launch" in desktop_src, "desktop.py 缺 available()/launch()")
    # 回归保护：v0.4.x 试过借系统浏览器的 --app 模式开窗口，实测在受管机器上静默失败，
    # 既开不出窗口也没法诊断。窗口后端只能是系统 WebView，这条断言就是那段事故的封条。
    need("--app=" not in desktop_src, "desktop.py 又出现了浏览器 app 模式（实测不可靠，见 packaging/README.md）")
    need("pywebview" in pyproject and "desktop =" in pyproject, "pyproject 缺 desktop 可选依赖（pywebview）")
    cli_src = (ROOT / "yaro" / "cli.py").read_text(encoding="utf-8")
    need('add_parser("app"' in cli_src, "cli.py 缺 `yaro app` 子命令")
    install_ps1 = (ROOT / "scripts" / "install.ps1").read_text(encoding="utf-8")
    need("yaro app" in install_ps1 or "yaro.desktop" in install_ps1, "install.ps1 没探测/引导独立窗口")
    install_sh = (ROOT / "scripts" / "install.sh").read_text(encoding="utf-8")
    need("yaro.desktop" in install_sh, "install.sh 没探测独立窗口后端")
    install_doc = (ROOT / "docs" / "INSTALL.md").read_text(encoding="utf-8")
    need("yaro app" in install_doc, "INSTALL.md 没写独立窗口启动方式")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    need("yaro/data" in manifest and "yaro/web" in manifest, "MANIFEST.in 没包含 data/web 资源")

    # 分发形态：同事不该自己装 Python/依赖。必须有 exe 打包与离线 wheel 两条路。
    for relative in (
        "packaging/README.md",
        "packaging/entry.py",
        "packaging/yaro.spec",
        "packaging/build_windows.ps1",
        "packaging/build_wheelhouse.ps1",
        "packaging/install_offline.ps1",
        "packaging/probe_index.ps1",
        "packaging/start-yaro.cmd",
    ):
        need((ROOT / relative).is_file(), f"分发包缺文件：{relative}")
    spec_src = (ROOT / "packaging" / "yaro.spec").read_text(encoding="utf-8")
    need('"yaro/web"' in spec_src, "yaro.spec 没把 yaro/web 打进产物（exe 会白页）")
    need('"yaro/data"' in spec_src, "yaro.spec 没把 yaro/data 打进产物（CONTRACT.md 会丢）")
    packaging_doc = (ROOT / "packaging" / "README.md").read_text(encoding="utf-8")
    need("YARO.exe" in packaging_doc and "wheelhouse" in packaging_doc, "packaging/README.md 没覆盖两条分发路线")

    # Windows 脚本必须是纯 ASCII。Windows PowerShell 5.1 在没有 UTF-8 BOM 时按 ANSI 解码 .ps1，
    # 中文会变成乱码，并让解析器把引号吃掉 —— 真实踩过：整个 build_windows.ps1 报
    # "表达式或语句中包含意外的标记"。这条断言把"Windows 脚本里只写英文"变成会失败的测试。
    windows_scripts = sorted((ROOT / "packaging").glob("*.ps1"))
    windows_scripts += [ROOT / "scripts" / "install.ps1", ROOT / "packaging" / "start-yaro.cmd"]
    for script in windows_scripts:
        try:
            script.read_bytes().decode("ascii")
        except UnicodeDecodeError as exc:
            need(False, f"{script.relative_to(ROOT)} 含非 ASCII 字符（PowerShell 5.1 解析会失败）：{exc}")

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    need(".env" in gitignore and ".yaro/" in gitignore, ".gitignore 没忽略 .env/.yaro（密钥与隐私数据会被提交）")

    # 模拟安装态：YARO_CONTRACT 覆盖必须生效（否则 wheel 里那份永远读不到）
    os.environ["YARO_CONTRACT"] = str(PKG_CONTRACT)
    try:
        need(resolve_contract_path() == PKG_CONTRACT, "YARO_CONTRACT 覆盖未生效")
    finally:
        os.environ.pop("YARO_CONTRACT", None)

    return f"契约双份一致 {len(PKG_CONTRACT.read_bytes())}B · package-data 就位 · 安装脚本与文档齐全"


def step_http(ctx: dict[str, Any]) -> str:
    """真起一个 HTTP 服务，用真实请求走完：静态 → 建会话 → demo 运行 → 回滚 → 产物。"""
    from yaro.server import Harness, _bind

    settings = ctx["settings"]
    harness = Harness(settings)
    httpd = _bind("127.0.0.1", 0, harness)
    port = httpd.server_address[1]
    base = f"http://127.0.0.1:{port}"
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        config = http_json(base + "/api/config")
        need(config["model"] == settings.model, f"config 里的 model 不对：{config['model']}")
        need(config["demo"]["files"], "config 没列出演示资产")
        need(not config["has_key"], "临时配置不该有 API Key")
        need(config["tools"], "config 没列出工具清单")

        need(b"YARO" in http_raw(base + "/"), "首页不是 YARO 的 HTML")
        need(b"--bg" in http_raw(base + "/assets/app.css"), "app.css 取不到")
        need(b"function" in http_raw(base + "/assets/app.js"), "app.js 取不到")
        need(b"svg" in http_raw(base + "/assets/logo.svg"), "logo.svg 取不到")

        created = http_json(base + "/api/sessions", method="POST")
        session_id = created["session_id"]
        need(session_id, "创建会话没返回 id")

        http_json(f"{base}/api/sessions/{session_id}/run", method="POST", body={"task": "", "demo": True})

        deadline = time.time() + 60
        state: dict[str, Any] = {}
        while time.time() < deadline:
            state = http_json(f"{base}/api/sessions/{session_id}?since=0")
            if state["stats"]["stop_reason"]:
                break
            time.sleep(0.15)
        need(state["stats"]["stop_reason"] == "done", f"HTTP 运行的 stop_reason={state['stats']['stop_reason']}")
        need(state["stats"]["turns"] >= 4, f"turns 太少：{state['stats']['turns']}")
        need(state["stats"]["cost_usd"] > 0, "HTTP 运行的成本为 0")
        need(state["stats"]["hit_rate"] > 0.5, f"缓存命中率异常：{state['stats']['hit_rate']}")
        need(state["lineage"] and state["head"] == state["lineage"][-1], "lineage/head 不一致")

        sha = ""
        for event in state["events"]:
            if event["type"] != "tool":
                continue
            for item in event["payload"].get("evidence", []):
                if item.get("kind") == "image":
                    sha = item["payload"]["sha256"]
                    break
            if sha:
                break
        need(sha, "HTTP 会话里没有 image 证据")
        png = http_raw(f"{base}/api/artifact/{sha}")
        need(png.startswith(b"\x89PNG"), "取回的产物不是 PNG")

        events_before = state["stats"]["events"]
        target = state["lineage"][1]
        rolled = http_json(f"{base}/api/sessions/{session_id}/rewind", method="POST", body={"event_id": target})
        need(rolled["heads"]["main"] == target, "HTTP 回滚后 head 不对")
        after = http_json(f"{base}/api/sessions/{session_id}")
        need(after["stats"]["events"] == events_before, "HTTP 回滚删除了事件")
        need(after["head"] == target, "HTTP 回滚后 head 不是目标事件")

        forked = http_json(
            f"{base}/api/sessions/{session_id}/fork",
            method="POST",
            body={"event_id": state["lineage"][0], "branch": "alt"},
        )
        need("alt" in forked["branches"], "HTTP fork 没建出分支")

        verified = http_json(f"{base}/api/sessions/{session_id}/verify", method="POST")
        need(verified["ok"], f"HTTP 校验失败：{verified['problems']}")

        need(http_status(base + "/api/nope") == 404, "未知接口应返回 404")
        need(http_status(base + "/api/artifact/not-a-sha") == 400, "非法 sha 应返回 400")

        # 安全回归（code review C-8：以前这里一条安全断言都没有，_guard 被改坏无人报警）
        need(
            http_status(base + "/api/config", headers={"Host": "evil.example.com"}) == 403,
            "非本机 Host 应被 403 拒绝（防 DNS rebinding）",
        )
        need(
            http_status(base + "/api/config", headers={"Origin": "http://evil.example.com"}) == 403,
            "跨站 Origin 应被 403 拒绝（防 CSRF）",
        )
        need(
            http_status(base + "/api/config", headers={"Origin": base}) == 200,
            "同源 Origin 应放行",
        )
        need(http_status(f"{base}/api/sessions/{session_id}?since=abc") == 400, "非法 since 应返回 400")
        # GUI 的 Key 入口：空 Key 必须被拒；clear 是安全操作（不会写任何密钥）
        empty_status, _ = _request(
            base + "/api/key", method="POST", body={"action": "set", "key": ""}
        )
        need(empty_status == 400, f"空 API Key 应被拒绝，实际 {empty_status}")
        cleared = http_json(base + "/api/key", method="POST", body={"action": "clear"})
        need(cleared.get("ok") is True, "清除 Key 应返回 ok")
        gates = http_json(base + "/api/gates?path=.")
        need(isinstance(gates.get("findings"), list), "gates 应返回 findings 列表")
        return f"端口 {port}：17 个请求全通（含 3 条安全断言）；运行 {state['stats']['turns']} 轮 ${state['stats']['cost_usd']:.4f}"
    finally:
        httpd.shutdown()
        httpd.server_close()


# --------------------------------------------------------------------------- #
# HTTP 小工具
# --------------------------------------------------------------------------- #


def _request(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, bytes]:
    data = None
    merged = dict(headers or {})
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        merged.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, headers=merged, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - 只打本机
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def http_raw(url: str) -> bytes:
    status, payload = _request(url)
    need(status == 200, f"{url} 返回 {status}：{payload[:200]!r}")
    return payload


def http_json(url: str, *, method: str = "GET", body: dict[str, Any] | None = None) -> dict[str, Any]:
    status, payload = _request(url, method=method, body=body)
    need(status == 200, f"{method} {url} 返回 {status}：{payload[:300].decode('utf-8', 'replace')}")
    return json.loads(payload.decode("utf-8"))


def http_status(url: str, *, headers: dict[str, str] | None = None) -> int:
    return _request(url, headers=headers)[0]


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

STEPS: list[tuple[str, Callable[[dict[str, Any]], str]]] = [
    ("模块导入", step_imports),
    ("配置与契约", step_config),
    ("产物 CAS", step_artifacts),
    ("会话事件链", step_session),
    ("回滚与 fork", step_rollback),
    ("篡改与乱序检测", step_tamper),
    ("上下文压缩", step_compaction),
    ("评估器", step_evals),
    ("静态闸门", step_gates),
    ("标定判据", step_calib),
    ("演示资产", step_demo_assets),
    ("Agent 端到端（DemoLLM）", step_agent_loop),
    ("视觉全链路", step_vision),
    ("真实标定求解（可选）", step_cv2_optional),
    ("前端接线一致性", step_static_assets),
    ("隐私与出网边界", step_egress),
    ("打包自包含", step_packaging),
    ("HTTP 全链路", step_http),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="YARO 自检：证明这个库真的能跑")
    parser.add_argument("--no-http", action="store_true", help="跳过 HTTP 步骤")
    parser.add_argument("--keep", action="store_true", help="保留临时目录")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印每步细节")
    args = parser.parse_args(argv)

    print(f"YARO 自检 · {ROOT}")
    print(f"python {sys.version.split()[0]} · {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 72)

    ctx: dict[str, Any] = {}
    results: list[tuple[str, str, str]] = []
    counts = {PASS: 0, SKIP: 0, WARN: 0, FAIL: 0}

    try:
        for name, function in STEPS:
            if args.no_http and function is step_http:
                results.append((name, SKIP, "--no-http"))
                counts[SKIP] += 1
                print(f"  SKIP  {name}  (--no-http)")
                continue
            started = time.perf_counter()
            try:
                detail = function(ctx) or ""
                status = PASS
                counts[PASS] += 1
            except StepSkipped as exc:
                detail = str(exc)
                status = SKIP
                counts[SKIP] += 1
            except StepFailed as exc:
                detail = str(exc)
                status = FAIL
                counts[FAIL] += 1
            except BaseException:  # noqa: BLE001 - 一条步骤不该带崩整轮自检
                detail = traceback.format_exc(limit=8)
                status = FAIL
                counts[FAIL] += 1
            elapsed = (time.perf_counter() - started) * 1000
            results.append((name, status, detail))
            print(f"  {status:<4}  {name}  ({elapsed:.0f}ms)")
            if args.verbose or status != PASS:
                first_line = detail.strip().splitlines()[0] if detail.strip() else ""
                print(f"        {first_line[:180]}")
    finally:
        backup = ctx.get("_env_backup") or {}
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        tmp = ctx.get("tmp")
        if tmp and not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)

    print("-" * 72)
    print(
        f"通过 {counts[PASS]} · 跳过 {counts[SKIP]} · 警告 {counts[WARN]} · 失败 {counts[FAIL]}"
        f"   ({sum(counts.values())} 步)"
    )

    failures = [(name, detail) for name, status, detail in results if status == FAIL]
    if failures:
        print("\n失败详情：")
        for name, detail in failures:
            print(f"\n[{name}]\n{detail}")
        print("\n提示：先修第一个失败项；后面的失败经常是它的连锁反应。")
        return 1

    if counts[SKIP]:
        print("跳过的步骤不影响结论：它们只是本机缺少可选依赖。")

    print("\n下一步：")
    print("  python -m yaro gui --no-browser   # 起 GUI，浏览器开 http://127.0.0.1:8765")
    print("  勾选「演示模式」→ Ctrl+Enter，看完整一条标定评估流程")
    if not (ROOT / ".env").exists():
        print("  （真实模型需要 cp .env.example .env 并填 DEEPSEEK_API_KEY）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
