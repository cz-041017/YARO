"""桌面窗口壳的离线测试：不真的开窗口，只验证降级路径、诊断字段与打包入口。

之所以不测"窗口真的打开了"：CI/无显示环境起不了 WebView，而且这里最该保护的是
"没装 pywebview 时不能崩，要能回退浏览器模式"——那才是给同事分发时会踩的坑。
"""

from __future__ import annotations

import pytest

from yaro import desktop


def test_desktop_unavailable_is_runtime_error() -> None:
    assert issubclass(desktop.DesktopUnavailable, RuntimeError)


def _no_webview(monkeypatch: pytest.MonkeyPatch) -> None:
    """模拟"没装 pywebview"的机器（分发时最常见的一种）。"""

    def boom():
        raise desktop.DesktopUnavailable("模拟：没装 pywebview")

    monkeypatch.setattr(desktop, "_import_webview", boom)


def test_available_returns_bool() -> None:
    assert isinstance(desktop.available(), bool)


def test_available_false_without_webview(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_webview(monkeypatch)
    assert desktop.available() is False


def test_launch_raises_before_touching_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_webview(monkeypatch)
    with pytest.raises(desktop.DesktopUnavailable):
        desktop.launch(None, port=0)


def test_window_constants_are_sane() -> None:
    assert desktop.WINDOW_SIZE[0] >= 1024 and desktop.WINDOW_SIZE[1] >= 640
    assert desktop.WINDOW_MIN[0] <= desktop.WINDOW_SIZE[0]
    assert desktop.WINDOW_MIN[1] <= desktop.WINDOW_SIZE[1]
    assert "YARO" in desktop.WINDOW_TITLE


def test_diagnose_shape() -> None:
    info = desktop.diagnose()
    for key in (
        "ok",
        "backend",
        "pywebview",
        "webview2",
        "winforms",
        "frozen",
        "python",
        "platform",
        "executable",
        "code_version",
        "dist_version",
        "stale_install",
        "package",
        "hint",
    ):
        assert key in info, f"diagnose 缺字段：{key}"
    assert isinstance(info["ok"], bool)
    assert isinstance(info["stale_install"], bool)
    assert isinstance(info["frozen"], bool)
    assert info["backend"] in {"pywebview", "none"}
    assert isinstance(info["hint"], str) and info["hint"]


def test_no_browser_app_mode_left() -> None:
    """回归保护：曾经用系统浏览器的 --app 模式兜底，实测不可靠，已移除。

    如果以后有人（或我）又想"零依赖地借浏览器窗口"，这个测试会拦下来，
    并提醒读 packaging/README.md 里那段事故记录。
    """
    assert not hasattr(desktop, "find_chromium")
    assert not hasattr(desktop, "_run_chromium_app")


def test_winforms_available_returns_bool() -> None:
    assert isinstance(desktop._winforms_available(), bool)


def test_winforms_available_non_windows_is_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(desktop.sys, "platform", "linux")
    assert desktop._winforms_available() is True


def test_launch_falls_back_when_winforms_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """回归：pywebview 装了但缺 .NET Desktop Runtime（System.Windows.Forms）时，
    launch 必须抛 DesktopUnavailable 让 CLI 回退浏览器模式，而不是在 webview.start()
    处开一个空白/报错的控制台。"""

    def fake_webview():
        return object()

    monkeypatch.setattr(desktop, "_import_webview", fake_webview)
    monkeypatch.setattr(desktop, "_winforms_available", lambda: False)
    with pytest.raises(desktop.DesktopUnavailable):
        desktop.launch(None, port=0)
