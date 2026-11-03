"""桌面窗口：把本地 GUI 装进一个独立窗口，而不是交给"用户的浏览器"。

为什么需要这一层
----------------
浏览器里的页面会被三类外部因素改写，而且**每台机器都不一样**：

1. 浏览器扩展（Dark Reader / 夜间模式 / 护眼插件）会把浅色页面动态反转成深色；
2. 浏览器的"强制深色"实验开关（Chrome 的 ``#enable-force-dark``）；
3. 系统的对比度主题 / 深色模式。

同一份 CSS 在开发机上正常、在同事机器上却"黑得看不见"，问题根本不在 CSS。
窗口壳用系统 WebView 渲染同一份页面，没有第三方扩展，配色只由 YARO 决定。

为什么只用 WebView（不再用浏览器的 app 模式）
-------------------------------------------
早期版本试过调用系统 Edge/Chrome 的 ``--app <url>`` 来"借一个窗口"。实测在受管机器上
浏览器会静默退出（exit 0、无输出），既开不出窗口也无法可靠诊断——把"能不能开窗口"
押在用户浏览器的行为上，正是多人分发最不该做的事。现在只有一条路：
**系统 WebView（pywebview）**；没有就明确告诉用户走打包好的 ``YARO.exe`` 或浏览器模式。

分发策略（关键）
----------------
- 同事机器上**不应该**要求他们自己 ``pip install``：多数人没有可用的网络/代理。
- 正确形态是 `packaging/` 里两个产物：
  1. ``YARO.exe``（PyInstaller 打包，内含 Python + 依赖 + 前端资源，双击即用）；
  2. 离线 wheel 包（没网也能 ``pip install --no-index --find-links``）。
- 本模块只负责"起服务 + 开 WebView 窗口"，与打包方式无关：

  * 源码运行：``python -m yaro app``
  * 打包运行：``YARO.exe``（``sys.frozen`` 为真，入口见 ``packaging/entry.py``）

设计约束
--------
- 核心仍然零依赖；``pywebview`` 只在 ``[desktop]``/``[full]`` extras 里。
- 窗口只是本地 HTTP 服务的一个消费者；关窗口即优雅停服务。
- 不做任何权限提升：仍然只监听 127.0.0.1。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from .config import Settings

WINDOW_TITLE = "YARO · 视觉开发助手"
WINDOW_SIZE = (1440, 900)
WINDOW_MIN = (1080, 680)

INSTALL_HINT = (
    "安装窗口后端：python -m pip install \"pywebview>=5.0\"；"
    "或者使用打包好的 YARO.exe（内置全部依赖，不需要 Python）。"
)


class DesktopUnavailable(RuntimeError):
    """本机没有可用的窗口后端；调用方应回退到浏览器模式。"""


def _log(line: str) -> None:
    print(line, flush=True)


# ---------------------------------------------------------------------------
# 后端：系统 WebView（pywebview）
# ---------------------------------------------------------------------------


def _import_webview() -> Any:
    """导入 pywebview；任何异常都归一成 :class:`DesktopUnavailable`。"""
    try:
        import webview  # noqa: PLC0415  (可选依赖，按需导入)
    except Exception as exc:  # noqa: BLE001
        raise DesktopUnavailable(f"没有安装 pywebview（{exc.__class__.__name__}）") from exc
    return webview


def _pywebview_available() -> bool:
    try:
        _import_webview()
    except DesktopUnavailable:
        return False
    return True


def _pywebview_version() -> str:
    try:
        from importlib.metadata import version

        return version("pywebview")
    except Exception:  # noqa: BLE001
        return "未知"


def available() -> bool:
    """本机能不能开独立窗口（只检查 pywebview，不启动任何东西）。"""
    return _pywebview_available()


def _frozen() -> bool:
    """是不是 PyInstaller 打出来的可执行文件。"""
    return bool(getattr(sys, "frozen", False))


def _webview2_runtime() -> bool | None:
    """Windows：Edge WebView2 Runtime 是否装了。其它平台返回 None（不需要）。"""
    if sys.platform != "win32":
        return None
    import os

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "EdgeWebView" / "Application",
        Path("C:/Program Files (x86)/Microsoft/EdgeWebView/Application"),
        Path("C:/Program Files/Microsoft/EdgeWebView/Application"),
    ]
    for candidate in candidates:
        try:
            if candidate.is_dir() and any(candidate.iterdir()):
                return True
        except OSError:
            continue
    try:
        import winreg  # noqa: PLC0415

        for root in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                key = winreg.OpenKey(
                    root,
                    r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients"
                    r"\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}",
                )
            except OSError:
                continue
            try:
                value, _ = winreg.QueryValueEx(key, "pv")
            except OSError:
                continue
            finally:
                key.Close()
            if value:
                return True
    except Exception:  # noqa: BLE001
        return None
    return False


def _winforms_available() -> bool:
    """Windows：pywebview 6.x 的 EdgeChromium 后端用 pythonnet 托管 WebView2，需要
    .NET **Windows Desktop Runtime**（``System.Windows.Forms``）。只装了 .NET 基础运行时
    而没装 Desktop Runtime 时，``import clr`` 能成但 ``clr.AddReference('System.Windows.Forms')``
    抛 ``FileNotFoundException``，结果 ``webview.start()`` 静默开不出窗口。这里把整条链试一遍，
    失败即判定"桌面后端不可用"，让调用方回退浏览器模式，而不是开一个只有报错的控制台。"""
    if sys.platform != "win32":
        return True  # macOS / Linux 走各自后端（GTK/Cocoa），不走 WinForms
    import os

    try:
        try:
            import clr  # noqa: PLC0415
        except Exception:  # noqa: BLE001 - 默认 netfx 失败时切 coreclr（与 edgechromium.py 一致）
            os.environ["PYTHONNET_RUNTIME"] = "coreclr"
            import clr  # noqa: PLC0415
        clr.AddReference("System.Windows.Forms")  # noqa: PLC0415
        import System.Windows.Forms  # noqa: PLC0415, F401
        return True
    except Exception:  # noqa: BLE001
        return False


def diagnose() -> dict[str, Any]:
    """返回一组只读诊断字段，``yaro app --check`` 与 ``yaro doctor`` 都用它。"""
    from . import __version__

    code_version = __version__
    dist_version = "未安装"
    package = str(Path(__file__).resolve().parent)
    stale = False
    try:
        from importlib.metadata import version

        dist_version = version("yaro")
        stale = bool(dist_version) and dist_version != code_version
    except Exception:  # noqa: BLE001
        pass

    webview_ready = _pywebview_available()
    webview_version = _pywebview_version() if webview_ready else "未安装"
    backend = "pywebview" if webview_ready else "none"
    webview2 = _webview2_runtime()
    winforms = _winforms_available()
    ok = webview_ready and (webview2 is not False) and winforms

    if ok:
        hint = "窗口可用：yaro app"
    elif not webview_ready:
        hint = INSTALL_HINT
    elif webview2 is False:
        hint = (
            "缺少 Windows Edge WebView2 Runtime（Win10/11 一般自带）。"
            "装它：https://developer.microsoft.com/microsoft-edge/webview2/ ；"
            "或改用 `yaro gui`（浏览器标签页模式）。"
        )
    elif not winforms:
        hint = (
            "缺少 .NET Windows Desktop Runtime（pywebview 6.x 的窗口后端需要 System.Windows.Forms）。"
            "装它：https://dotnet.microsoft.com/download/dotnet （选 Windows Desktop Runtime）；"
            "或改用 `yaro gui`（浏览器标签页模式，功能一样）。"
        )
    else:
        hint = "窗口后端不可用；可改用 `yaro gui`（浏览器标签页模式）。"

    return {
        "ok": ok,
        "backend": backend,
        "pywebview": webview_version,
        "webview2": webview2,
        "winforms": winforms,
        "frozen": _frozen(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "executable": sys.executable,
        "package": package,
        "code_version": code_version,
        "dist_version": dist_version,
        "stale_install": stale,
        "privacy": "本地 127.0.0.1；窗口内不访问任何外部资源",
        "hint": hint,
    }


# ---------------------------------------------------------------------------
# 本地服务 + 窗口
# ---------------------------------------------------------------------------


def _start_server(settings: Settings, host: str, port: int) -> tuple[Any, str, Any]:
    """起本地 HTTP 服务（后台线程），返回 ``(httpd, url, harness)``。"""
    from .server import Harness, _bind

    harness = Harness(settings)
    httpd = _bind(host, port, harness)
    threading.Thread(target=httpd.serve_forever, daemon=True, name="yaro-http").start()
    host, port = httpd.server_address[:2]
    return httpd, f"http://{host}:{port}/", harness


def _open_webview(webview: Any, url: str, title: str, gui: str | None) -> None:
    """用 pywebview 开窗口。

    不做"失败就退回 mshtml"的静默降级（code review C-5）：mshtml 是 IE11 引擎，
    微软已停止安全更新，用它渲染模型输出等于把一个无补丁的浏览器内核塞进产品。
    宁可报错，让用户装 WebView2 Runtime 或用 ``yaro gui`` 走现代浏览器。
    """
    webview.create_window(
        title,
        url,
        width=WINDOW_SIZE[0],
        height=WINDOW_SIZE[1],
        min_size=WINDOW_MIN,
        confirm_close=False,
        text_select=True,
    )
    if gui:
        webview.start(gui=gui)
    else:
        webview.start()


def launch(
    settings: Settings | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    title: str = WINDOW_TITLE,
    gui: str | None = None,
) -> int:
    """起服务 + 开窗口，窗口关闭后返回；没有后端时抛 :class:`DesktopUnavailable`。"""
    _log("[1/3] 检查窗口后端 …")
    webview = _import_webview()  # 没有就抛，且不碰 settings
    version = _pywebview_version()
    _log(f"      pywebview {version}")
    if not _winforms_available():
        raise DesktopUnavailable(
            "pywebview 6.x 的窗口后端需要 .NET Windows Desktop Runtime（System.Windows.Forms），"
            "本机没有装，开不出原生窗口；已回退浏览器模式。"
        )

    settings = (settings or Settings.load()).ensure()

    _log("[2/3] 准备本地服务（首次会生成 demo 数据）…")
    httpd, url, harness = _start_server(settings, host, port)

    from .egress import summarize_policy

    _log(f"[3/3] 服务已就绪 → {url}")
    _log(f"      workspace: {settings.workspace}")
    _log(f"      home:      {settings.home}")
    _log(f"      隐私:      {summarize_policy(harness.egress, settings.base_url)}")
    _log("      打开窗口 …（关窗口即退出）")
    try:
        _open_webview(webview, url, title, gui)
    finally:
        try:
            httpd.shutdown()
        finally:
            httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(launch())
