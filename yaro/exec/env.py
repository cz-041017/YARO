"""探测本机能跑什么：torch / CUDA / onnxruntime / OpenCV / 版本。

用途：在真正执行训练或预标注之前先问"这台机器具备什么能力"，
把"缺 torch"变成一句可执行的安装建议，而不是一个 ImportError 栈。
零依赖：import 探测全部用 importlib.util.find_spec + 子进程取版本。
"""

from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

# 训练/预标注真正需要的包 -> 安装说明
_EXTRAS = {
    "torch": "torch",
    "torchvision": "torchvision",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "cv2": "opencv-python-headless",
    "numpy": "numpy",
    "PIL": "pillow",
}


def python_executable() -> str:
    return sys.executable


def which(name: str) -> str | None:
    from shutil import which as _which

    return _which(name)


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _version_of(module: str) -> str:
    """用子进程取版本，避免在当前进程 import 重库（torch 启动很慢）。"""
    code = (
        "import json,sys;"
        f"m=__import__({module!r});"
        "print(json.dumps({'v': getattr(m,'__version__','?')}))"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if out.returncode == 0 and out.stdout.strip():
            return str(json.loads(out.stdout.strip().splitlines()[-1]).get("v", "?"))
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    return "?"


def _cuda_info() -> dict[str, Any]:
    """torch 可用时问它要 CUDA 信息；否则返回不可用。"""
    if not _has("torch"):
        return {"available": False, "devices": [], "note": "未安装 torch"}
    code = (
        "import json;"
        "import torch;"
        "print(json.dumps({"
        "'torch': torch.__version__,"
        "'cuda': bool(torch.cuda.is_available()),"
        "'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],"
        "'mps': bool(getattr(torch.backends,'mps',None) and torch.backends.mps.is_available())"
        "}))"
    )
    try:
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if out.returncode == 0 and out.stdout.strip():
            data = json.loads(out.stdout.strip().splitlines()[-1])
            return {
                "available": bool(data.get("cuda")) or bool(data.get("mps")),
                "torch": data.get("torch", "?"),
                "devices": data.get("devices", []),
                "mps": data.get("mps", False),
                "note": "CUDA" if data.get("cuda") else ("MPS" if data.get("mps") else "仅 CPU"),
            }
    except (subprocess.SubprocessError, ValueError, OSError):
        pass
    return {"available": False, "devices": [], "note": "torch 探测失败"}


def capabilities() -> dict[str, Any]:
    """返回本机能力清单（可 JSON 序列化，直接进 doctor / 事件）。"""
    packages = {}
    for module, pip_name in _EXTRAS.items():
        present = _has(module)
        packages[module] = {
            "present": present,
            "pip": pip_name,
            "version": _version_of(module) if present else "",
        }
    cuda = _cuda_info()
    missing_train = [p["pip"] for m, p in packages.items() if not p["present"] and m in ("torch", "torchvision")]
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "cuda": cuda,
        "can_train": bool(packages["torch"]["present"] and packages["torchvision"]["present"]),
        "can_prelabel_torch": bool(packages["torch"]["present"] and packages["torchvision"]["present"]),
        "can_prelabel_onnx": bool(packages["onnxruntime"]["present"]),
        "can_prelabel_contour": bool(packages["cv2"]["present"]),
        "can_export_onnx": bool(packages["torch"]["present"] and packages["onnx"]["present"]),
        "missing_for_train": missing_train,
    }


def suggest_install(*, train: bool = True, with_cv: bool = False) -> str:
    """给出与当前解释器绑定的安装命令（不要用裸 pip）。"""
    names: list[str] = []
    if train:
        names += ["torch", "torchvision"]
    names += ["onnx", "onnxruntime"]
    if with_cv:
        names += ["numpy", "opencv-python-headless"]
    joined = " ".join(dict.fromkeys(names))
    return f'"{sys.executable}" -m pip install {joined}'


def describe(caps: dict[str, Any] | None = None) -> str:
    """一行中文摘要，用于 CLI / 工具 hint。"""
    caps = caps or capabilities()
    if caps["can_train"]:
        dev = caps["cuda"].get("note", "?")
        return f"可训练（torch {caps['packages']['torch']['version']}，{dev}）"
    missing = "、".join(caps["missing_for_train"]) or "torch"
    return f"不可训练：缺 {missing}；安装：{suggest_install()}"


def ensure_root(root: Path | str) -> Path:
    path = Path(root).resolve()
    if not path.exists():
        raise FileNotFoundError(f"工作区不存在：{path}")
    return path
