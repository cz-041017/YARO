"""命令行入口。

    yaro init                      初始化工作区（给同事的第一条命令）
    yaro gui                       启动本地 Web GUI
    yaro doctor                    自检：依赖、配置、沙箱、产物库、隐私边界
    yaro key set|status|clear      API Key 的本地安全存储（getpass，不进 shell 历史）
    yaro chain                     按固定顺序跑视觉数据链路（确定性，不经过模型）
    yaro dataset scan|audit|...    单步数据集操作
    yaro session list|verify|show  会话与事件链
    yaro gates <path...>           静态闸门扫描
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from .contracts import ContractError
from pathlib import Path
from typing import Any

from . import __version__
from .capabilities.sandbox import DockerSandbox
from .config import ROOT, Settings, system_prompt
from .egress import MODES, host_of, is_private_host, normalize_mode, summarize_policy
from .gates import format_findings, scan_files
from .memory.artifacts import ArtifactStore
from .memory.session import SessionStore
from .secrets import mask, status as secret_status


def _check(name: str, ok: bool, detail: str = "") -> dict[str, object]:
    return {"check": name, "ok": ok, "detail": detail}


# ---------------------------------------------------------------------- #
# doctor
# ---------------------------------------------------------------------- #


def doctor_report(settings: Settings) -> dict[str, object]:
    """自检结果（纯数据，CLI 与 GUI 共用同一份判断，避免两边结论不一致）。"""
    settings = settings.ensure()
    checks: list[dict[str, object]] = []
    contract = settings.contract_path

    checks.append(_check("python", sys.version_info >= (3, 10), f"{sys.version.split()[0]}"))
    checks.append(_check("contract", contract.exists(), str(contract)))

    try:
        import numpy  # noqa: F401  # type: ignore[import-not-found]

        checks.append(_check("numpy", True, "已安装"))
    except ImportError:
        checks.append(_check("numpy", False, "未安装（yaro[cv] 需要）"))

    try:
        import cv2  # noqa: F401  # type: ignore[import-not-found]

        checks.append(_check("opencv", True, "已安装（标定可求解）"))
    except ImportError:
        checks.append(_check("opencv", False, "未安装：判据仍可用，标定求解不可用"))

    sandbox = DockerSandbox(image=settings.sandbox_image, allow_net=settings.allow_net)
    available = sandbox.available()
    checks.append(
        _check(
            "docker",
            available,
            "可用" if available else "不可用：生产执行必须走沙箱，请安装 Docker",
        )
    )

    # 密钥：只说来源与掩码，永不打印原文。
    key = secret_status(settings.home)
    checks.append(
        _check(
            "api_key",
            bool(settings.api_key),
            (
                f"来源 {key['source']} · {key['masked']} · 更换：yaro key set"
                if settings.api_key
                else "未配置（演示模式可用）· 填写入口：yaro key set，或 .env 里 DEEPSEEK_API_KEY="
            ),
        )
    )
    checks.append(_check("model", True, f"主 {settings.model} / 规划 {settings.model_plan}"))

    # 隐私边界：这是企业/个人数据安全的核心开关，必须出现在自检里。
    policy = settings.egress_policy()
    public_target = not is_private_host(host_of(settings.base_url))
    egress_ok = not (policy.mode != "off" and public_target and not policy.allow_public)
    checks.append(
        _check(
            "egress",
            egress_ok,
            summarize_policy(policy, settings.base_url)
            + ("" if egress_ok else " · 配置矛盾：要出网必须显式 YARO_EGRESS_ALLOW_PUBLIC=1"),
        )
    )

    store = ArtifactStore(settings.artifacts_dir)
    checks.append(_check("artifacts", True, f"{len(store)} 个对象，{store.stats()['bytes']} 字节（去重后）"))

    sessions = sorted(path.name for path in settings.sessions_dir.glob("*") if path.is_dir())
    checks.append(_check("sessions", True, f"{len(sessions)} 个会话"))
    checks.append(_check("workspace", settings.workspace.exists(), str(settings.workspace)))

    # 桌面窗口只是显示层：可用最好，不可用也照常能跑（会回退 yaro gui），所以不进 fatal。
    try:
        from .desktop import diagnose as desktop_diagnose

        info = desktop_diagnose()
        if info["ok"]:
            checks.append(_check("desktop", True, f"独立窗口可用（{info['backend']}，pywebview {info['pywebview']}）"))
        else:
            checks.append(
                _check(
                    "desktop",
                    False,
                    f"独立窗口不可用：pywebview={info['pywebview']} webview2={info['webview2']}"
                    "（回退 yaro gui；诊断 yaro app --check）",
                )
            )
    except Exception as exc:  # noqa: BLE001 - 探测本身不该让 doctor 崩
        checks.append(_check("desktop", False, f"探测失败：{exc}"))

    # 陈旧安装是"敲了命令没反应"的经典原因：控制台脚本 `yaro` 走的可能是旧代码。
    try:
        from importlib.metadata import version as _dist_version

        from . import __version__ as _code_version

        if getattr(sys, "frozen", False):
            # 打包版自带代码，importlib.metadata 读到的是构建机上的旧 egg-info，无意义。
            checks.append(_check("install", True, f"代码 {_code_version}（打包版）"))
        else:
            try:
                dist = _dist_version("yaro")
            except Exception:  # noqa: BLE001
                dist = ""
            if dist and dist != _code_version:
                checks.append(
                    _check(
                        "install",
                        False,
                        f"安装的是 {dist}，代码是 {_code_version}："
                        f'运行 "{sys.executable}" -m pip install -e ".[full]" 消除歧义',
                    )
                )
            else:
                checks.append(_check("install", True, f"代码 {_code_version}" + (f" / 安装 {dist}" if dist else "")))
    except Exception as exc:  # noqa: BLE001
        checks.append(_check("install", False, f"版本检查失败：{exc}"))

    # 只有"致命项"参与 ok 判定：opencv/api_key/docker 缺失会降级，但不阻塞演示与判据。
    fatal = {"python", "contract", "workspace"}
    ok = all(check["ok"] for check in checks if check["check"] in fatal)
    return {
        "version": __version__,
        "root": str(ROOT),
        "home": str(settings.home),
        "workspace": str(settings.workspace),
        "contract": str(contract),
        "egress": policy.describe(),
        "key": key,
        "ok": ok,
        "checks": checks,
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    report = doctor_report(Settings.load())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


# ---------------------------------------------------------------------- #
# init
# ---------------------------------------------------------------------- #

_ENV_TEMPLATE = """# YARO 本地配置（由 `yaro init` 生成；不要提交到 git）
# 工作区：工具能看到的根目录。默认当前目录。
YARO_WORKSPACE={workspace}

# 出网策略：off（默认，不出网）| text | derived | full
YARO_EGRESS={egress}
# 只有显式设为 1，才允许把请求发到公网端点（如 api.deepseek.com）。
YARO_EGRESS_ALLOW_PUBLIC={allow_public}

# 模型：主模型必须支持 Vision；规划模型可以不支持。
YARO_MODEL={model}
YARO_MODEL_PLAN={model_plan}
DEEPSEEK_BASE_URL={base_url}

# API Key 推荐用 `yaro key set` 存到本地 0600 文件或系统钥匙串，
# 不要把明文写在这里。若必须写，注意 .env 已在 .gitignore 中。
# DEEPSEEK_API_KEY=
"""


def _write_env(path: Path, settings: Settings, *, force: bool) -> Path | None:
    if path.exists() and not force:
        return None
    text = _ENV_TEMPLATE.format(
        workspace=settings.workspace.as_posix(),
        egress=settings.egress,
        allow_public="1" if settings.egress_allow_public else "0",
        model=settings.model,
        model_plan=settings.model_plan,
        base_url=settings.base_url,
    )
    path.write_text(text, encoding="utf-8")
    try:
        os.chmod(path, 0o600)  # POSIX 生效；Windows 上是尽力而为
    except OSError:  # pragma: no cover
        pass
    return path


def cmd_init(args: argparse.Namespace) -> int:
    """给同事的第一步：建工作区、铺演示图、写 CONTRACT、给出下一步。

    幂等：重复执行不会覆盖已有 .env / CONTRACT.md，除非 ``--force``。
    """
    settings = Settings.load()
    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else settings.workspace
    egress = normalize_mode(args.egress) if args.egress else settings.egress
    settings = Settings(
        home=settings.home,
        workspace=workspace,
        model=settings.model,
        model_plan=settings.model_plan,
        api_key=settings.api_key,
        base_url=settings.base_url,
        sandbox_image=settings.sandbox_image,
        allow_net=settings.allow_net,
        allow_unsafe_local=settings.allow_unsafe_local,
        egress=egress,
        egress_allow_public=bool(args.allow_public) or settings.egress_allow_public,
        api_key_source=settings.api_key_source,
    ).ensure()

    workspace.mkdir(parents=True, exist_ok=True)
    demo = workspace / "demo"
    demo.mkdir(parents=True, exist_ok=True)

    # CONTRACT.md 放进工作区，团队可以改阈值/规矩，改完记得递增版本号。
    contract_dest = workspace / "contract" / "CONTRACT.md"
    contract_dest.parent.mkdir(parents=True, exist_ok=True)
    if args.force or not contract_dest.exists():
        contract_dest.write_text(system_prompt(), encoding="utf-8")

    env_path = _write_env(Path.cwd() / ".env", settings, force=args.force)

    from .demo import ensure_demo_assets

    assets = ensure_demo_assets(demo)

    status = secret_status(settings.home)
    report = {
        "workspace": str(workspace),
        "home": str(settings.home),
        "contract": str(contract_dest),
        "env": str(env_path) if env_path else "已存在（未覆盖，--force 可重建）",
        "demo_files": assets["files"],
        "egress": summarize_policy(settings.egress_policy(), settings.base_url),
        "key": status,
        "next": [
            "1) yaro key set            # 可选：安全存 API Key（不出现在 shell 历史）",
            "2) yaro doctor             # 自检：环境 / 沙箱 / 隐私边界",
            "3) yaro gui                # 打开界面，先在「演示模式」跑通全流程（零 token）",
        ],
    }
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"YARO 已初始化 → {workspace}")
        print(f"  home:     {settings.home}")
        print(f"  contract: {contract_dest}")
        print(f"  .env:     {report['env']}")
        print(f"  演示图:   {len(assets['files'])} 张（{assets['relative_dir']}/）")
        print(f"  隐私:     {report['egress']}")
        print(f"  API Key:  {status['masked']}（来源 {status['source']}）")
        if not status["configured"]:
            print("            填写入口：终端执行 `yaro key set`；或在程序窗口里点左栏 `api key` 一行填写；")
            print("            或在 .env 里写 DEEPSEEK_API_KEY=sk-...。")
            print("            （窗口里填的 Key 只发到 127.0.0.1，存进系统钥匙串或 ~/.yaro/secrets.json，")
            print("              不写进页面/事件/日志，界面只显示掩码。）")
        print("\n下一步：")
        for line in report["next"]:
            print("  " + line)
    return 0


# ---------------------------------------------------------------------- #
# key
# ---------------------------------------------------------------------- #


def cmd_key(args: argparse.Namespace) -> int:
    from .secrets import clear_api_key, keyring_clear, keyring_set, set_api_key

    settings = Settings.load()
    settings.ensure()

    if args.action == "status":
        print(json.dumps(secret_status(settings.home), ensure_ascii=False, indent=2))
        return 0

    if args.action == "clear":
        clear_api_key(settings.home)
        keyring_clear()
        print("已清除本地密钥（环境变量中的 DEEPSEEK_API_KEY 请自行 unset）")
        return 0

    # set：优先 getpass（TTY，不回显、不进 shell 历史），否则从 stdin 读一行
    value = ""
    if not args.stdin:
        try:
            import getpass

            value = getpass.getpass("请输入 DeepSeek API Key（输入不回显）: ")
        except (EOFError, KeyboardInterrupt):  # pragma: no cover - 交互中断
            print("\n已取消", file=sys.stderr)
            return 2
    else:
        value = sys.stdin.readline().strip()
    value = value.strip()
    if not value:
        print("密钥为空，未写入", file=sys.stderr)
        return 2

    if keyring_set(value):
        clear_api_key(settings.home)
        print(f"已写入系统钥匙串（{mask(value)}）")
    else:
        path = set_api_key(settings.home, value)
        print(f"已写入本地文件 {path}（权限 0600；{mask(value)}）")
    print("提示：该密钥不会出现在日志/事件/GUI 中；`yaro doctor` 只显示来源与掩码。")
    return 0


# ---------------------------------------------------------------------- #
# gui / session / gates / run
# ---------------------------------------------------------------------- #


def cmd_app(args: argparse.Namespace) -> int:
    """启动独立窗口（系统 WebView）。多人协作推荐用它：不受浏览器扩展/强制深色影响。"""
    if getattr(args, "check", False):
        from .desktop import diagnose

        info = diagnose()
        for key in (
            "ok",
            "backend",
            "pywebview",
            "webview2",
            "frozen",
            "python",
            "platform",
            "executable",
            "code_version",
            "dist_version",
            "package",
        ):
            if key in info:
                print(f"{key:12} {info[key]}")
        if info.get("stale_install"):
            print("")
            print(f"⚠ 版本不一致：代码是 {info['code_version']}，但环境里安装的 yaro 是 {info['dist_version']}。")
            print("  控制台脚本 `yaro` 可能仍在跑旧代码。")
            print("  修复：走源码时用 `python -m yaro ...`；或按 packaging/README.md 重新打包。")
        print("")
        print(f"结论：{info['hint']}")
        return 0 if info["ok"] else 1

    print("YARO 桌面模式启动中（首次加载窗口可能需要几秒）…", flush=True)
    settings = Settings.load().ensure()
    from .desktop import DesktopUnavailable, launch

    try:
        return launch(
            settings,
            host=args.host,
            port=args.port,
            gui=getattr(args, "gui", None),
        )
    except DesktopUnavailable as exc:
        print("窗口模式不可用 —— 已回退到浏览器模式（功能完全一样）。")
        print(f"  原因: {exc}")
        print("")
        print("  推荐做法（多人协作场景）：用打包好的 YARO.exe，内置 Python/依赖/窗口后端，")
        print("  同事不需要装 Python、不需要联网。构建方法见 packaging/README.md。")
        print('  或者本机装窗口后端：python -m pip install "pywebview>=5.0"')
        print("  诊断: python -m yaro app --check")
        from .server import serve

        serve(settings, host=args.host, port=args.port, open_browser=True)
        return 0
    except KeyboardInterrupt:
        print("已中断。")
        return 130
    except Exception:  # noqa: BLE001 - 任何意外都要让用户看到栈，而不是"没反应"
        import traceback

        print("桌面窗口启动失败，完整错误如下（把这段贴给维护者即可定位）：")
        traceback.print_exc()
        print("  诊断: python -m yaro app --check")
        print("  回退: python -m yaro gui")
        return 1


def cmd_gui(args: argparse.Namespace) -> int:
    """启动本地 Web GUI。"""
    from .server import serve

    serve(
        Settings.load(),
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )
    return 0


def cmd_session_list(args: argparse.Namespace) -> int:
    settings = Settings.load().ensure()
    rows = []
    for path in sorted(settings.sessions_dir.glob("*")):
        if not path.is_dir():
            continue
        try:
            store = SessionStore.open(settings.sessions_dir, path.name)
        except Exception as exc:  # noqa: BLE001 - 损坏的会话不应阻塞列表
            rows.append({"session": path.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        report = store.verify()
        rows.append(
            {
                "session": path.name,
                "events": report["events"],
                "branches": report["branches"],
                "verified": report["ok"],
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def cmd_session_verify(args: argparse.Namespace) -> int:
    settings = Settings.load().ensure()
    store = SessionStore.open(settings.sessions_dir, args.session_id)
    report = store.verify()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 2


def cmd_session_show(args: argparse.Namespace) -> int:
    settings = Settings.load().ensure()
    store = SessionStore.open(settings.sessions_dir, args.session_id)
    events = store.all_events()
    print(f"session={args.session_id} events={len(events)} heads={store.heads()}")
    for event in events[-args.tail :]:
        brief = json.dumps(event.payload, ensure_ascii=False)[:160]
        print(f"  #{event.seq:04d} {event.type:<10} {event.event_id[:8]}  {brief}")
    return 0


def cmd_gates(args: argparse.Namespace) -> int:
    result = scan_files(args.paths)
    print(result.summary())
    print(format_findings(result.findings))
    return 0 if result.ok else 2


def cmd_run(args: argparse.Namespace) -> int:
    """真实跑一轮需要 API Key、网络与放行的出网策略；这里只做诚实的前置检查。"""
    settings = Settings.load().ensure()
    policy = settings.egress_policy()
    if not settings.api_key:
        print("缺少 API Key：先运行 `yaro key set`（推荐）或配置环境变量", file=sys.stderr)
        return 2
    try:
        policy.check_endpoint(settings.base_url)
    except Exception as exc:  # noqa: BLE001
        print(f"出网策略拦截：{exc}", file=sys.stderr)
        return 3
    print(
        "P0 阶段请通过 Python API 驱动（见 README 示例）：\n"
        "  llm = DeepSeek(api_key=..., model=settings.model, egress=settings.egress_policy())\n"
        "  agent = Agent(llm=llm, registry=..., session=..., artifacts=..., egress=settings.egress_policy())\n"
        "  await agent.run(task)\n"
        "CLI 的完整 run 子命令在 P1 接入工具集后开放。"
    )
    return 0


# ---------------------------------------------------------------------- #
# 视觉链路：命令行直接调用同一批工具（不经过模型，确定性、可脚本化）
# ---------------------------------------------------------------------- #


def _vision_tool(name: str, root: Path, store: ArtifactStore, **kwargs: object) -> Any:
    from .vision.tools import build_vision_tools

    tools = {tool.spec.name: tool for tool in build_vision_tools(root, store, run_id="cli")}
    if name not in tools:
        raise SystemExit(f"未知的视觉工具：{name}")
    return asyncio.run(tools[name].run(**kwargs))


def _print_result(result: Any, *, verbose: bool) -> int:
    print(("OK  " if result.ok else "FAIL") + result.summary)
    for evidence in result.evidence:
        if evidence.kind == "table":
            print(f"\n[{evidence.caption}]")
            headers: list[str] = []
            for row in evidence.payload:
                for key in row:
                    if key not in headers:
                        headers.append(key)
            if headers:
                print("  " + " | ".join(str(h) for h in headers))
                for row in evidence.payload:
                    print("  " + " | ".join(str(row.get(h, "")) for h in headers))
        elif verbose:
            print(f"\n[{evidence.caption}] {evidence.to_text()[:400]}")
    if result.metrics:
        print("\nmetrics: " + json.dumps(result.metrics, ensure_ascii=False, sort_keys=True))
    if result.degraded:
        print("degraded: " + ", ".join(result.degraded))
    if result.hint:
        print("next: " + result.hint)
    return 0 if result.ok else 1


def _bootstrap_demo(settings: Any, dataset: str) -> None:
    """``demo/…`` 这类内置路径第一次用之前自动铺好演示资源。

    GUI 演示与 ``yaro chain`` 都依赖 ``demo/dataset``；如果新同事 clone 后
    没先跑 ``yaro init`` 就直接用，这里补一次（幂等、零成本），避免
    "目录不存在" 这种与用户意图无关的失败。
    """
    if not str(dataset).replace("\\", "/").startswith("demo"):
        return
    target = settings.workspace / dataset
    if target.exists():
        return
    from .demo import ensure_demo_assets

    assets = ensure_demo_assets(settings.demo_dir)
    if assets.get("dataset_error"):
        print(f"（演示数据生成失败：{assets['dataset_error']}）")
    else:
        print(f"（已生成演示数据：{assets['dataset']}）")


def cmd_chain(args: argparse.Namespace) -> int:
    """按固定顺序跑完整条数据链路：scan → audit → render → split（方案）→ train_plan。"""
    settings = Settings.load().ensure()
    store = ArtifactStore(settings.artifacts_dir)
    root = settings.workspace
    dataset = args.dataset
    _bootstrap_demo(settings, dataset)
    steps: list[tuple[str, dict[str, object]]] = [
        ("dataset_scan", {"dir": dataset}),
        ("dataset_audit", {"dir": dataset}),
        ("dataset_render", {"dir": dataset, "max_images": args.max_images}),
        ("dataset_split", {"dir": dataset, "apply": args.apply}),
        ("train_plan", {"dir": dataset, "backend": args.backend, "epochs": args.epochs}),
    ]
    failures = 0
    for index, (name, kwargs) in enumerate(steps, start=1):
        print(f"\n=== [{index}/{len(steps)}] {name} ===")
        result = _vision_tool(name, root, store, **kwargs)
        failures += 0 if _print_result(result, verbose=args.verbose) == 0 else 1
    print(f"\n链路完成：{len(steps) - failures}/{len(steps)} 步通过")
    if args.apply:
        print(f"切分文件已写入 {dataset}/splits/（seed 固定，可复现）")
    return 0 if failures == 0 else 1


def cmd_pipeline(args: argparse.Namespace) -> int:
    """真执行全链路：预标注 → 清洗体检 → 训练 → 调优 → 导出/测速。

    与 ``chain`` 的区别：chain 只做确定性数据整理（不训练）；pipeline 会真的起训练子进程、
    真的导出 ONNX、真的测延迟。训练默认要求 CUDA（``--cpu`` 显式降级冒烟）。
    """
    settings = Settings.load().ensure()
    store = ArtifactStore(settings.artifacts_dir)
    root = settings.workspace
    dataset = args.dataset
    _bootstrap_demo(settings, dataset)

    stages: list[tuple[str, dict[str, object]]] = []
    if not args.skip_prelabel:
        stages.append(
            (
                "prelabel_run",
                {
                    "dir": dataset,
                    "backend": args.backend,
                    "conf": args.conf,
                    "limit": args.prelabel_limit,
                    "apply": args.apply,
                },
            )
        )
    stages.append(("dataset_audit", {"dir": dataset}))
    stages.append(("dataset_split", {"dir": dataset, "apply": True}))
    if not args.skip_train:
        stages.append(
            (
                "train_run",
                {
                    "dir": dataset,
                    "epochs": args.epochs,
                    "imgsz": args.imgsz,
                    "batch": args.batch,
                    "lr": args.lr,
                    "device": "cpu" if args.cpu else "auto",
                    "max_minutes": args.max_minutes,
                },
            )
        )
    if not args.skip_tune and not args.skip_train:
        stages.append(
            (
                "tune_run",
                {
                    "dir": dataset,
                    "trials": args.trials,
                    "max_minutes_total": args.tune_minutes,
                },
            )
        )
    if not args.skip_deploy and not args.skip_train:
        stages.append(
            (
                "deploy_run",
                {
                    "weights": f"{dataset}/runs/train/best.pt",
                    "imgsz": args.deploy_imgsz,
                    "runs": args.bench_runs,
                    "latency_target_ms": args.latency_target,
                },
            )
        )

    failures = 0
    for index, (name, kwargs) in enumerate(stages, start=1):
        print(f"\n=== [{index}/{len(stages)}] {name} ===")
        result = _vision_tool(name, root, store, **kwargs)
        failures += 0 if _print_result(result, verbose=args.verbose) == 0 else 1
    print(f"\n全链路完成：{len(stages) - failures}/{len(stages)} 步通过")
    if args.skip_train:
        print("（已跳过训练/调优/部署：加 --train 才会真的跑模型）")
    return 0 if failures == 0 else 1


def cmd_dataset(args: argparse.Namespace) -> int:
    settings = Settings.load().ensure()
    store = ArtifactStore(settings.artifacts_dir)
    root = settings.workspace
    _bootstrap_demo(settings, args.dataset)
    action = args.action
    payload: dict[str, object] = {"dir": args.dataset}
    if action == "scan":
        payload = {"dir": args.dataset}
    elif action == "audit":
        payload = {"dir": args.dataset}
    elif action == "render":
        payload = {"dir": args.dataset, "max_images": args.max_images}
    elif action == "split":
        payload = {
            "dir": args.dataset,
            "train": args.train,
            "val": args.val,
            "test": args.test,
            "seed": args.seed,
            "apply": args.apply,
        }
    elif action == "dedupe":
        payload = {"dir": args.dataset, "apply": args.apply}
    elif action == "convert":
        payload = {"dir": args.dataset, "to": args.to}
    result = _vision_tool(f"dataset_{action}" if action != "convert" else "label_convert", root, store, **payload)
    return _print_result(result, verbose=args.verbose)



# ---------------------------------------------------------------------- #
# 解析
# ---------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yaro", description="YARO —— 机器视觉算法开发 harness")
    parser.add_argument("--version", action="version", version=f"yaro {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="初始化工作区（新同事从这里开始）")
    init.add_argument("--workspace", default=None, help="工作区路径，默认当前目录")
    init.add_argument("--egress", default=None, choices=MODES, help="出网策略，默认沿用配置（off）")
    init.add_argument("--allow-public", action="store_true", help="显式允许请求发往公网端点（谨慎）")
    init.add_argument("--force", action="store_true", help="覆盖已有 .env 与 contract/CONTRACT.md")
    init.add_argument("--json", action="store_true", help="输出 JSON")
    init.set_defaults(func=cmd_init)

    sub.add_parser("doctor", help="自检").set_defaults(func=cmd_doctor)

    gui = sub.add_parser("gui", help="启动本地 Web GUI")
    gui.add_argument("--host", default="127.0.0.1")
    gui.add_argument("--port", type=int, default=8765)
    gui.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    gui.set_defaults(func=cmd_gui)

    app = sub.add_parser("app", help="启动独立窗口（系统 WebView；不受浏览器扩展影响）")
    app.add_argument("--host", default="127.0.0.1")
    app.add_argument("--port", type=int, default=8765)
    app.add_argument("--gui", default=None, help="pywebview 后端：edgechromium / mshtml / gtk / cocoa")
    app.add_argument("--check", action="store_true", help="只诊断窗口环境，不启动窗口")
    app.set_defaults(func=cmd_app)

    key = sub.add_parser("key", help="API Key 的本地安全存储")
    key_sub = key.add_subparsers(dest="action", required=True)
    key_sub.add_parser("set", help="交互式写入 Key（不回显）").set_defaults(func=cmd_key, stdin=False)
    key_stdin = key_sub.add_parser("set-stdin", help="从 stdin 读一行 Key（脚本用）")
    key_stdin.set_defaults(func=cmd_key, stdin=True)
    key_sub.add_parser("status", help="显示来源与掩码").set_defaults(func=cmd_key, action="status")
    key_sub.add_parser("clear", help="清除本地 Key").set_defaults(func=cmd_key, action="clear")

    session = sub.add_parser("session", help="会话操作")
    session_sub = session.add_subparsers(dest="session_command", required=True)
    session_sub.add_parser("list", help="列出会话").set_defaults(func=cmd_session_list)
    verify = session_sub.add_parser("verify", help="校验事件链")
    verify.add_argument("session_id")
    verify.set_defaults(func=cmd_session_verify)
    show = session_sub.add_parser("show", help="查看最近事件")
    show.add_argument("session_id")
    show.add_argument("--tail", type=int, default=20)
    show.set_defaults(func=cmd_session_show)

    gates = sub.add_parser("gates", help="静态闸门扫描")
    gates.add_argument("paths", nargs="+")
    gates.set_defaults(func=cmd_gates)

    chain = sub.add_parser("chain", help="按固定顺序跑视觉数据链路（不经过模型）")
    chain.add_argument("--dataset", default="demo/dataset", help="数据集目录（相对工作区）")
    chain.add_argument("--backend", default="torchvision", choices=["torchvision", "ultralytics", "custom"])
    chain.add_argument("--epochs", type=int, default=10)
    chain.add_argument("--max-images", type=int, default=6, dest="max_images")
    chain.add_argument("--apply", action="store_true", help="真正写 splits/*.txt")
    chain.add_argument("--verbose", action="store_true")
    chain.set_defaults(func=cmd_chain)

    dataset_cmd = sub.add_parser("dataset", help="数据集操作：scan/audit/render/split/dedupe/convert")
    dataset_sub = dataset_cmd.add_subparsers(dest="action", required=True)
    for action, help_text in [
        ("scan", "扫描结构与类别分布"),
        ("audit", "体检：损坏/缺标/越界/重复/失衡"),
        ("render", "把标注画回图上并拼联系人表"),
        ("split", "分层切分（默认只出方案）"),
        ("dedupe", "按 sha256 找重复并隔离"),
        ("convert", "导出 COCO / VOC"),
    ]:
        node = dataset_sub.add_parser(action, help=help_text)
        node.add_argument("--dataset", default="demo/dataset", help="数据集目录（相对工作区）")
        node.add_argument("--apply", action="store_true", help="真正落盘（split/dedupe 生效）")
        node.add_argument("--verbose", action="store_true")
        if action == "render":
            node.add_argument("--max-images", type=int, default=6, dest="max_images")
        if action == "split":
            node.add_argument("--train", type=float, default=0.7)
            node.add_argument("--val", type=float, default=0.2)
            node.add_argument("--test", type=float, default=0.1)
            node.add_argument("--seed", type=int, default=7)
        if action == "convert":
            node.add_argument("--to", default="coco", choices=["coco", "voc"])
        node.set_defaults(func=cmd_dataset)

    pipeline = sub.add_parser("pipeline", help="真执行全链路：预标注 → 体检 → 切分 → 训练 → 调优 → 导出/测速")
    pipeline.add_argument("--dataset", default="demo/dataset", help="数据集目录（相对工作区）")
    pipeline.add_argument("--backend", default="auto", choices=["auto", "torchvision", "onnx", "contour"], help="预标注后端")
    pipeline.add_argument("--conf", type=float, default=0.35, help="预标注置信度阈值")
    pipeline.add_argument("--prelabel-limit", type=int, default=0, dest="prelabel_limit", help="预标注最多处理多少张（0=全部）")
    pipeline.add_argument("--apply", action="store_true", help="把 proposals 真正写入 labels/（默认只产 proposals）")
    pipeline.add_argument("--epochs", type=int, default=12)
    pipeline.add_argument("--imgsz", type=int, default=512)
    pipeline.add_argument("--batch", type=int, default=2)
    pipeline.add_argument("--lr", type=float, default=0.005)
    pipeline.add_argument("--max-minutes", type=float, default=60.0, dest="max_minutes", help="单次训练预算（分钟）")
    pipeline.add_argument("--cpu", action="store_true", help="允许 CPU 冒烟训练（默认只在 CUDA 上训练）")
    pipeline.add_argument("--trials", type=int, default=4, help="调优跑几组")
    pipeline.add_argument("--tune-minutes", type=float, default=90.0, dest="tune_minutes", help="调优总预算（分钟）")
    pipeline.add_argument("--deploy-imgsz", type=int, default=640, dest="deploy_imgsz")
    pipeline.add_argument("--bench-runs", type=int, default=20, dest="bench_runs")
    pipeline.add_argument("--latency-target", type=float, default=0.0, dest="latency_target", help="p95 目标毫秒（0=只报数不判定）")
    pipeline.add_argument("--skip-prelabel", action="store_true", dest="skip_prelabel")
    pipeline.add_argument("--skip-train", action="store_true", dest="skip_train")
    pipeline.add_argument("--skip-tune", action="store_true", dest="skip_tune")
    pipeline.add_argument("--skip-deploy", action="store_true", dest="skip_deploy")
    pipeline.add_argument("--verbose", action="store_true")
    pipeline.set_defaults(func=cmd_pipeline)

    run = sub.add_parser("run", help="（P1）跑一个任务")
    run.add_argument("task", nargs="?")
    run.set_defaults(func=cmd_run)
    return parser


def _normalize_key_args(argv: list[str] | None) -> list[str] | None:
    """``yaro key set --stdin`` 兼容写法 → ``yaro key set-stdin``。"""
    if not argv:
        return argv
    args = list(argv)
    if len(args) >= 3 and args[0] == "key" and args[1] == "set" and "--stdin" in args[2:]:
        args = ["key", "set-stdin"] + [item for item in args[2:] if item != "--stdin"]
    return args


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(_normalize_key_args(argv))
    try:
        return int(args.func(args) or 0)
    except ContractError as exc:
        # 配置类错误（模型名写错、未知出网档位…）用一行中文说清，别甩 traceback。
        print(f"配置错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
