"""代码执行隔离。

上游 OpenVLHarness 的 coding tool 直接在宿主机以用户权限跑生成的 Python（README 自己标了
[!WARNING] "执行 timeout 不是沙箱"）。这在企业里不可交付——模型生成的一行 ``shutil.rmtree``
就能毁掉一个工作区。

所以 YARO 的执行默认走 Docker：``--network none``、``--read-only``、非 root、CPU/内存/进程数配额。
本地执行只能通过 ``LocalUnsafeSandbox`` 显式开启（``YARO_ALLOW_UNSAFE_LOCAL=1``），并在事件里留痕。
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from ..contracts import ContractError

_MAX_STREAM = 8192  # 每路输出只保留末尾 8KB：模型读长日志没用，还会烧上下文


@dataclass
class ExecResult:
    ok: bool
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    truncated: bool = False
    artifacts: list[str] = field(default_factory=list)
    note: str = ""

    def to_text(self) -> str:
        parts = [f"exit_code={self.exit_code} duration={self.duration_s:.1f}s" + (" TIMEOUT" if self.timed_out else "")]
        if self.stdout:
            parts.append(f"--- stdout ---\n{self.stdout}")
        if self.stderr:
            parts.append(f"--- stderr ---\n{self.stderr}")
        if self.truncated:
            parts.append("(输出已截断，只保留末尾)")
        if self.note:
            parts.append(self.note)
        return "\n".join(parts)


def _tail(text: str) -> tuple[str, bool]:
    if len(text) <= _MAX_STREAM:
        return text, False
    return "…(前段省略)…\n" + text[-_MAX_STREAM:], True


class DockerSandbox:
    """Docker 隔离执行器。依赖宿主已安装 docker CLI。"""

    def __init__(
        self,
        *,
        image: str = "python:3.12-slim",
        allow_net: bool = False,
        cpus: float = 2.0,
        memory_gb: float = 8.0,
        timeout_s: float = 300.0,
        pids_limit: int = 256,
    ) -> None:
        self.image = image
        self.allow_net = allow_net
        self.cpus = cpus
        self.memory_gb = memory_gb
        self.timeout_s = timeout_s
        self.pids_limit = pids_limit

    def available(self) -> bool:
        return shutil.which("docker") is not None

    def _argv(self, mounts: Sequence[tuple[Path, str]]) -> list[str]:
        argv = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--read-only",
            "--tmpfs", "/tmp:rw,size=512m",
            "--user", "65534:65534",
            "--cpus", str(self.cpus),
            "--memory", f"{self.memory_gb:g}g",
            "--memory-swap", f"{self.memory_gb:g}g",
            "--pids-limit", str(self.pids_limit),
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
        ]
        argv += ["--network", "bridge" if self.allow_net else "none"]
        for host_path, container_path in mounts:
            argv += ["-v", f"{Path(host_path).resolve()}:{container_path}:ro"]
        argv += ["--workdir", "/tmp", self.image, "python", "-"]
        return argv

    def run(
        self,
        code: str,
        *,
        mounts: Sequence[tuple[Path, str]] = (),
        timeout_s: float | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        if not self.available():
            raise ContractError(
                "docker 不可用。生产执行必须走沙箱；如需本机临时跑通，请显式设置 YARO_ALLOW_UNSAFE_LOCAL=1 并使用 LocalUnsafeSandbox。"
            )
        timeout = timeout_s or self.timeout_s
        argv = self._argv(mounts)
        if env:
            index = argv.index(self.image)
            env_args: list[str] = []
            for key, value in env.items():
                env_args += ["-e", f"{key}={value}"]
            argv = argv[:index] + env_args + argv[index:]
        started = time.monotonic()
        try:
            completed = subprocess.run(
                argv,
                input=code.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.monotonic() - started
            stdout, cut_out = _tail((exc.stdout or b"").decode("utf-8", "replace"))
            stderr, cut_err = _tail((exc.stderr or b"").decode("utf-8", "replace"))
            return ExecResult(
                ok=False,
                exit_code=-1,
                stdout=stdout,
                stderr=stderr,
                duration_s=duration,
                timed_out=True,
                truncated=cut_out or cut_err,
                note=f"超过 {timeout:.0f}s 被强制终止；请降低单次运行规模或拆成可缓存的步骤",
            )

        duration = time.monotonic() - started
        stdout, cut_out = _tail(completed.stdout.decode("utf-8", "replace"))
        stderr, cut_err = _tail(completed.stderr.decode("utf-8", "replace"))
        return ExecResult(
            ok=completed.returncode == 0,
            exit_code=int(completed.returncode),
            stdout=stdout,
            stderr=stderr,
            duration_s=duration,
            truncated=cut_out or cut_err,
        )


class LocalUnsafeSandbox:
    """**不安全**的本机执行，仅供无 Docker 的开发环境临时验证。

    必须在设置 ``YARO_ALLOW_UNSAFE_LOCAL=1`` 时才能构造；每次执行都会在 note 里标注，
    上层需要把这条标注写进 ``events.jsonl``，让审计能看出"这次跑在宿主上"。
    """

    def __init__(self, *, allow_unsafe_local: bool = False, timeout_s: float = 300.0) -> None:
        if not allow_unsafe_local:
            raise ContractError(
                "拒绝在本机执行模型生成的代码。"
                "要么使用 DockerSandbox，要么显式设置 YARO_ALLOW_UNSAFE_LOCAL=1（仅限本地开发）。"
            )
        self.timeout_s = timeout_s

    def available(self) -> bool:
        return shutil.which("python") is not None or shutil.which("python3") is not None

    def run(self, code: str, *, timeout_s: float | None = None, **_ignored: object) -> ExecResult:
        timeout = timeout_s or self.timeout_s
        python = shutil.which("python") or shutil.which("python3") or "python"
        started = time.monotonic()
        try:
            completed = subprocess.run(
                [python, "-"],
                input=code.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(
                ok=False,
                exit_code=-1,
                duration_s=time.monotonic() - started,
                timed_out=True,
                note="超时终止（本机执行，无隔离）",
            )
        stdout, cut_out = _tail(completed.stdout.decode("utf-8", "replace"))
        stderr, cut_err = _tail(completed.stderr.decode("utf-8", "replace"))
        return ExecResult(
            ok=completed.returncode == 0,
            exit_code=int(completed.returncode),
            stdout=stdout,
            stderr=stderr,
            duration_s=time.monotonic() - started,
            truncated=cut_out or cut_err,
            note="UNSAFE: 本次执行在宿主机上进行，无文件系统/网络隔离",
        )
