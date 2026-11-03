"""子进程执行器：跑 harness 自己生成的脚本，带超时、流式输出、进程树清理。

只做一件事：把 argv 跑起来，把输出收集回来。不做解析（解析留给 vision.plan 的
parse_train_log）、不做预算判断（留给调用方）、不落盘（产物登记由上层做）。
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

LineHandler = Callable[[str, str], None]  # (stream: "stdout"|"stderr", line)


@dataclass(frozen=True)
class RunSpec:
    """一次受控执行。argv 必须是调用方构造好的模板命令，不接受模型随意拼接。"""

    argv: list[str]
    cwd: Path | str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 1800.0
    label: str = "run"


@dataclass
class RunOutcome:
    label: str
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool = False
    error: str = ""
    lines: int = 0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.error

    def tail(self, limit: int = 4000) -> str:
        text = (self.stdout + ("\n" if self.stdout and self.stderr else "") + self.stderr).strip()
        if len(text) <= limit:
            return text
        return "…（前 %d 字符已截断）\n%s" % (len(text) - limit, text[-limit:])

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "argv": self.argv,
            "returncode": self.returncode,
            "duration_s": round(self.duration_s, 2),
            "timed_out": self.timed_out,
            "ok": self.ok,
            "error": self.error,
            "lines": self.lines,
            "tail": self.tail(1200),
        }


_SENSITIVE_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")
# 子进程只继承白名单前缀：训练脚本由 harness 生成，但它仍然可能 print(os.environ)，
# 所以 API Key / 云凭证一律不带进去（需要额外变量走 RunSpec.env 显式传）。
_SAFE_ENV_PREFIXES = (
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP",
    "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
    "LANG", "LC_", "PYTHON", "VIRTUAL_ENV", "CONDA", "PIP_",
    "CUDA", "NVIDIA", "OMP_", "MKL", "KMP", "NUMBA", "TORCH",
    "HF_", "TRANSFORMERS", "XDG_", "SSL_", "REQUESTS_", "HTTP_", "HTTPS_", "NO_PROXY",
)


def _child_env(overrides: dict[str, str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for name, value in os.environ.items():
        upper = name.upper()
        if any(hint in upper for hint in _SENSITIVE_HINTS):
            continue
        if not any(upper == prefix or upper.startswith(prefix) for prefix in _SAFE_ENV_PREFIXES):
            continue
        env[name] = value
    env.update({str(k): str(v) for k, v in overrides.items()})
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _kill_tree(pid: int) -> None:
    """Windows 上 kill 父进程不会带走子进程（DataLoader worker 等），要整棵树。"""
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                timeout=30,
            )
            return
        except (subprocess.SubprocessError, OSError):
            pass
    try:
        if os.name == "nt":
            os.kill(pid, 9)
        else:
            os.killpg(pid, 9)  # Popen 用了 start_new_session，pid 即进程组号
    except OSError:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def run(spec: RunSpec, *, on_line: LineHandler | None = None, max_lines: int = 20000) -> RunOutcome:
    """同步执行（调用方在 async 工具里用 asyncio.to_thread 包一层即可）。"""
    argv = [str(a) for a in spec.argv]
    cwd = str(spec.cwd) if spec.cwd else None
    env = _child_env({str(k): str(v) for k, v in spec.env.items()})
    started = time.monotonic()
    # 环形缓冲：保留**末尾** max_lines 行。失败原因（Traceback / OOM）几乎总在尾部，
    # 之前"只收前 N 行"会让最重要的错误信息一个字节都进不来。
    out_buf: deque[str] = deque(maxlen=max_lines)
    err_buf: deque[str] = deque(maxlen=max_lines)
    counters = {"stdout": 0, "stderr": 0}
    lock = threading.Lock()

    def pump(stream_name: str, pipe, sink: deque[str]) -> None:
        try:
            for raw in iter(pipe.readline, b""):
                try:
                    line = raw.decode("utf-8", "replace").rstrip("\r\n")
                except (UnicodeDecodeError, AttributeError):
                    line = str(raw)
                with lock:
                    counters[stream_name] += 1
                    sink.append(line)
                if on_line is not None:
                    try:
                        on_line(stream_name, line)
                    except Exception:  # 回调坏了不能影响执行
                        pass
        except (ValueError, OSError):
            pass
        finally:
            try:
                pipe.close()
            except (ValueError, OSError):
                pass

    try:
        proc = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=(os.name != "nt"),  # POSIX：独立进程组，超时可整组杀
        )
    except FileNotFoundError as exc:
        return RunOutcome(
            label=spec.label,
            argv=argv,
            returncode=127,
            stdout="",
            stderr="",
            duration_s=0.0,
            error=f"找不到可执行文件：{argv[0] if argv else ''}（{exc}）",
        )
    except OSError as exc:
        return RunOutcome(
            label=spec.label,
            argv=argv,
            returncode=126,
            stdout="",
            stderr="",
            duration_s=0.0,
            error=f"无法启动进程：{exc}",
        )

    threads = [
        threading.Thread(target=pump, args=("stdout", proc.stdout, out_buf), daemon=True),
        threading.Thread(target=pump, args=("stderr", proc.stderr, err_buf), daemon=True),
    ]
    for thread in threads:
        thread.start()

    timed_out = False
    try:
        proc.wait(timeout=spec.timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc.pid)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
    finally:
        for thread in threads:
            thread.join(timeout=10)

    duration = time.monotonic() - started
    stdout_text = "\n".join(out_buf)
    stderr_text = "\n".join(err_buf)
    truncated = counters["stdout"] + counters["stderr"] > (len(out_buf) + len(err_buf))
    if truncated:
        # 只留尾部，明确告诉调用方前面被丢了（不要让模型以为输出只有这么点）。
        stdout_text = "[输出被截断：只保留末尾 %d 行]\n%s" % (len(out_buf), stdout_text)
    if timed_out:
        returncode = -9
    else:
        returncode = int(proc.returncode if proc.returncode is not None else -9)
    return RunOutcome(
        label=spec.label,
        argv=argv,
        returncode=returncode,
        stdout=stdout_text,
        stderr=stderr_text,
        duration_s=duration,
        timed_out=timed_out,
        error="执行超时，已终止进程树" if timed_out else "",
        lines=counters["stdout"] + counters["stderr"],
    )


def run_python(
    code: str,
    *,
    cwd: Path | str | None = None,
    timeout_s: float = 1800.0,
    label: str = "python",
    python: str | None = None,
    on_line: LineHandler | None = None,
) -> RunOutcome:
    """用当前解释器执行一段脚本（写临时文件而不是 -c，便于在错误里看到行号）。"""
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", delete=False, encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(code)
        path = handle.name
    try:
        return run(
            RunSpec(
                argv=[python or sys.executable, path],
                cwd=cwd,
                timeout_s=timeout_s,
                label=label,
            ),
            on_line=on_line,
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
