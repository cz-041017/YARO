# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置：产出 dist/YARO/YARO.exe（onedir，内含 Python + 依赖 + 前端）。

为什么用 onedir 而不是 onefile：
- onedir 启动快（onefile 每次都要解压到临时目录，WebView 启动会明显变慢）；
- 杀软对 onefile 自解压更敏感，企业环境更容易被拦；
- 出问题时目录里能直接看到日志与资源，好排查。

产物目录里必须有 yaro/web 与 yaro/data —— 否则 `yaro app` 打开是白页、
CONTRACT.md 也会丢（历史 bug，别退化）。
"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).parent  # noqa: F821  (SPECPATH 由 PyInstaller 注入)

hiddenimports = collect_submodules("webview")
# LangGraph 运行时用 importlib 做动态导入（checkpoint 序列化器、pregel 运行时、
# MCP 适配器），PyInstaller 的静态分析抓不全，必须显式收集子模块，否则打包后的
# `yaro gui`/`yaro app` 会在 `import yaro.graph.runner` 处 ImportError。
for _mod in ("langgraph", "langchain_core", "langchain_openai", "langchain_mcp_adapters", "mcp"):
    try:
        hiddenimports += collect_submodules(_mod)
    except Exception:  # pragma: no cover - 构建机上缺失某包时继续，让静态分析兜底
        pass

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[
        (str(ROOT / "yaro" / "web"), "yaro/web"),
        (str(ROOT / "yaro" / "data"), "yaro/data"),
        (str(ROOT / "contract"), "contract"),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["pytest", "tests", "tkinter.test"],
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="YARO",
    debug=False,
    strip=False,
    upx=False,
    # console=True：双击时保留一个控制台窗口。
    # 这是故意的——窗口后端出问题时，用户至少能看到一行中文原因（而不是"双击没反应"）。
    # 想要纯窗口形态，把这里改成 False 并重新打包即可（日志会改写到 <home>/logs/）。
    console=True,
    icon=None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="YARO",
)
