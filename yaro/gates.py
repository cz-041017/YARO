"""闸门：行动前的 Guides 与行动后的 Sensors。

这里实现的是**最便宜**的那道闸：静态扫描。它不替代人审、不替代回归集，
只负责在"提交一个 run 之前"挡住几类已知会翻车的写法。

设计原则：每条规则都要能给出"为什么 + 怎么改"，否则工程师会直接关掉闸门。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

# 主链路禁止引入的许可证（企业交付底线）。键是 import 名或分发名。
FORBIDDEN_LICENSES: dict[str, str] = {
    "ultralytics": "AGPL-3.0 —— 主链路不得引入；需要 YOLO 请走独立进程 + 法务评估",
    "yolov5": "GPL-3.0 —— 同上",
    "yolo_world": "GPL-3.0 —— 同上",
    "mmdet": "依赖树复杂，须先出 SBOM 再定",
}

DANGEROUS_CALLS: list[tuple[str, str, str]] = [
    (r"\beval\s*\(", "eval() 在生产代码里没有正当用途", "改用显式分支或 ast.literal_eval"),
    (r"\bexec\s*\(", "exec() 让代码不可静态审计", "把动态逻辑显式化"),
    (r"os\.system\s*\(", "shell 注入面", "改用 subprocess.run(list, shell=False)"),
    (r"subprocess\.[a-z_]+\([^)]*shell\s*=\s*True", "shell=True 会拼接不可信输入", "传 list 参数、保持 shell=False"),
    (r"pickle\.loads?\s*\(", "反序列化不可信 pickle 可执行任意代码", "改用 json / np.load(allow_pickle=False) 或 safetensors"),
    (r"cv2\.imwrite\s*\(", "直接写图片容易原地覆盖输入数据", "写新产物进 CAS，只移动指针"),
    (r"requests\.|urllib\.request|socket\.socket", "沙箱默认禁网，联网必须显式声明", "把网络依赖移到 capability 层并登记"),
]

# 无 seed 的随机：会让"同样的输入跑出不同结果"，破坏可复现性
_RANDOM_PAT = re.compile(r"\b(random\.|np\.random\.|torch\.rand)")
_SEED_PAT = re.compile(r"\bseed\s*\(|manual_seed\s*\(|np\.random\.default_rng\s*\(")

_ABS_PATH_PAT = re.compile(r'["\'](?:[A-Za-z]:\\\\|[A-Za-z]:\\|/home/|/Users/|/mnt/)')

_IMPORT_PAT = re.compile(r"^\s*(?:from\s+([\w\.]+)\s+import|import\s+([\w\.,\s]+))", re.MULTILINE)


@dataclass(frozen=True)
class Finding:
    severity: str  # "error" | "warn"
    rule: str
    message: str
    fix: str
    file: str = ""
    line: int = 0

    def format(self) -> str:
        location = f"{self.file}:{self.line}" if self.file else "-"
        return f"[{self.severity.upper()}] {self.rule} @ {location}：{self.message} → {self.fix}"


@dataclass
class GateResult:
    ok: bool
    findings: list[Finding] = field(default_factory=list)

    @property
    def errors(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.severity == "error"]

    def summary(self) -> str:
        if self.ok:
            return f"静态闸门通过（{len(self.findings)} 条提示）"
        return f"静态闸门拦截：{len(self.errors)} 个 error，{len(self.findings) - len(self.errors)} 个 warn"


def scan_source(code: str, *, filename: str = "<run>", line_offset_safe: bool = True) -> GateResult:
    """扫描一段源码。返回全部发现（不 fail-fast），便于一次修完。"""
    findings: list[Finding] = []
    lines = code.splitlines()

    def line_of(pattern: str | re.Pattern[str]) -> int:
        for index, text in enumerate(lines, start=1):
            if (re.search(pattern, text) if isinstance(pattern, str) else pattern.search(text)):
                return index
        return 0

    for module, reason in FORBIDDEN_LICENSES.items():
        if re.search(rf"(?:^|\W)(?:import|from)\s+{re.escape(module)}\b", code):
            findings.append(
                Finding("error", "license", f"引入 {module}：{reason}", "移出主链路或走独立进程 + 法务评估", filename, 0)
            )

    for pattern, message, fix in DANGEROUS_CALLS:
        if re.search(pattern, code):
            findings.append(
                Finding(
                    "error" if "eval" in pattern or "exec" in pattern or "pickle" in pattern else "warn",
                    "unsafe_call",
                    message,
                    fix,
                    filename,
                    line_of(pattern) if line_offset_safe else 0,
                )
            )

    if _RANDOM_PAT.search(code) and not _SEED_PAT.search(code):
        findings.append(
            Finding("warn", "reproducibility", "用了随机但没设 seed，同样输入会跑出不同结果", "在入口处固定 random/np/torch 的 seed 并记录进 run 元数据", filename, line_of(_RANDOM_PAT))
        )

    if _ABS_PATH_PAT.search(code):
        findings.append(
            Finding("error", "hardcoded_path", "硬编码绝对路径，换机器就崩", "路径从 mounts/env 注入，代码里只接受相对路径", filename, line_of(_ABS_PATH_PAT))
        )

    if re.search(r"\bprint\s*\(", code) and len(lines) > 60:
        findings.append(
            Finding("warn", "noise", "长脚本里大量 print 会挤掉证据预算", "只 print 聚合后的指标行；明细写产物文件", filename, 0)
        )

    return GateResult(ok=not any(finding.severity == "error" for finding in findings), findings=findings)


_SKIP_DIRS = frozenset({".git", ".venv", ".yaro", "node_modules", "__pycache__", "dist", "build"})
_SCAN_SUFFIXES = frozenset({".py", ".ps1", ".sh", ".cmd", ".bat", ".js", ".ts"})


def expand_paths(paths: Iterable[Path | str], *, max_files: int = 400) -> list[Path]:
    """把目录展开成待扫文件。

    W-7：以前目录被当成"文件不存在"跳过且 ok=True，而 GUI 的
    ``/api/gates?path=.`` 传的正是一个目录 —— 闸门恒通过，等于没有闸门。
    """
    expanded: list[Path] = []
    for path in paths:
        file_path = Path(path)
        if file_path.is_dir():
            for child in sorted(file_path.rglob("*")):
                if len(expanded) >= max_files:
                    break
                if child.is_file() and child.suffix.lower() in _SCAN_SUFFIXES:
                    if any(part in _SKIP_DIRS for part in child.parts):
                        continue
                    expanded.append(child)
        else:
            expanded.append(file_path)
    return expanded[:max_files]


def scan_files(paths: Iterable[Path | str]) -> GateResult:
    findings: list[Finding] = []
    for file_path in expand_paths(paths):
        if not file_path.is_file():
            findings.append(Finding("warn", "io", "文件不存在，已跳过", "检查路径", str(file_path), 0))
            continue
        try:
            code = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        result = scan_source(code, filename=str(file_path))
        findings.extend(result.findings)
    return GateResult(ok=not any(finding.severity == "error" for finding in findings), findings=findings)


def format_findings(findings: Sequence[Finding], *, limit: int = 20) -> str:
    if not findings:
        return "无发现"
    ordered = sorted(findings, key=lambda finding: (finding.severity != "error", finding.rule))
    return "\n".join(finding.format() for finding in ordered[:limit])
