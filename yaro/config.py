"""运行时配置：路径、模型、沙箱、出网策略。

零第三方依赖：`.env` 用 30 行解析器读，不引入 python-dotenv；密钥解析走
``yaro.secrets``（环境变量 → 钥匙串 → 本地 0600 文件 → .env）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .contracts import ensure_known_model

PACKAGE_ROOT = Path(__file__).resolve().parent  # yaro/
ROOT = PACKAGE_ROOT.parent  # 仓库根（开发态）；装成 wheel 后就是 site-packages

# CONTRACT.md 有两个来源：
# 1) 开发态：仓库根 contract/CONTRACT.md（人类编辑的那份）
# 2) 安装态：包内 yaro/data/CONTRACT.md（wheel 里必须带上，否则 pip 安装后丢失）
# 两者内容必须一致，`python verify.py` 会比对；优先用开发态那份。
DEV_CONTRACT = ROOT / "contract" / "CONTRACT.md"
PKG_CONTRACT = PACKAGE_ROOT / "data" / "CONTRACT.md"


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def resolve_contract_path() -> Path:
    override = _env("YARO_CONTRACT")
    if override:
        return Path(override).expanduser()
    for candidate in (DEV_CONTRACT, PKG_CONTRACT):
        if candidate.exists():
            return candidate
    return PKG_CONTRACT


CONTRACT_PATH = resolve_contract_path()


def load_dotenv(path: Path | None = None) -> None:
    """极简 .env 加载：KEY=VALUE，忽略注释与空行，不覆盖已有环境变量。

    查找顺序：显式传入 > 当前目录/.env（用户项目）> 仓库根/.env（开发态）。
    装成 wheel 后 ``ROOT`` 是 site-packages，所以必须支持当前目录这一档。
    """
    candidates = [path] if path is not None else [Path.cwd() / ".env", ROOT / ".env"]
    for candidate in candidates:
        p = Path(candidate)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())
        return


@dataclass(frozen=True)
class Settings:
    home: Path
    workspace: Path
    model: str
    model_plan: str
    api_key: str
    base_url: str
    sandbox_image: str
    allow_net: bool
    allow_unsafe_local: bool
    # ---- v0.2 新增（都带默认值，避免旧测试构造失败）----
    egress: str = "off"
    egress_allow_public: bool = False
    api_key_source: str = "none"

    @classmethod
    def load(cls, *, dotenv: bool = True) -> "Settings":
        from .secrets import resolve_api_key

        # 先记下 .env 注入前的状态，才能如实报告 key 的来源（env / dotenv）。
        pre_env_key = _env("DEEPSEEK_API_KEY")
        if dotenv:
            load_dotenv()

        # home 必须是绝对路径：相对路径会在"换个目录再启动"时生出一份新的 .yaro，
        # 密钥、会话、产物就都"丢"了（其实是分裂成两份）。
        raw_home = Path(_env("YARO_HOME", ".yaro")).expanduser()
        home = (raw_home if raw_home.is_absolute() else (Path.cwd() / raw_home)).resolve()
        workspace = _env("YARO_WORKSPACE")
        api_key, key_source = resolve_api_key(home)
        if key_source == "env" and not pre_env_key:
            key_source = "dotenv"
        if key_source == "dotenv":  # .env 里的值已经进了环境变量
            api_key = _env("DEEPSEEK_API_KEY")

        model = _env("YARO_MODEL", "deepseek-flash")
        model_plan = _env("YARO_MODEL_PLAN", "deepseek-v4-pro")
        # 配置写错就要在启动时炸掉（fail fast），不要跑到第一次记账才发现。
        # 价格表同时是"认识的型号清单"，见 contracts.ensure_known_model。
        ensure_known_model(model)
        ensure_known_model(model_plan)

        return cls(
            home=home,
            # 工具能看到的根目录。默认当前目录，因为"我在哪个项目里就操作哪个项目"最符合直觉。
            workspace=Path(workspace).resolve() if workspace else Path.cwd(),
            model=model,
            model_plan=model_plan,
            api_key=api_key,
            base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            sandbox_image=_env("YARO_SANDBOX_IMAGE", "python:3.12-slim"),
            allow_net=_env("YARO_ALLOW_NET", "0") == "1",
            allow_unsafe_local=_env("YARO_ALLOW_UNSAFE_LOCAL", "0") == "1",
            # 默认 off：不显式授权，一个字节都不许出网。
            egress=_env("YARO_EGRESS", "off").lower(),
            egress_allow_public=_env("YARO_EGRESS_ALLOW_PUBLIC", "0") == "1",
            api_key_source=key_source,
        )

    # ------------------------------------------------------------------ #
    # 路径
    # ------------------------------------------------------------------ #

    @property
    def demo_dir(self) -> Path:
        """离线演示图放在工作区里，这样工具用相对路径就能读到（不搞特殊目录）。"""
        return self.workspace / "demo"

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def artifacts_dir(self) -> Path:
        return self.home / "artifacts"

    @property
    def runs_dir(self) -> Path:
        return self.home / "runs"

    @property
    def secrets_path(self) -> Path:
        from .secrets import secrets_file

        return secrets_file(self.home)

    @property
    def contract_path(self) -> Path:
        return resolve_contract_path()

    def ensure(self) -> "Settings":
        for path in (self.home, self.sessions_dir, self.artifacts_dir, self.runs_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self

    # ------------------------------------------------------------------ #
    # 出网策略
    # ------------------------------------------------------------------ #

    def egress_policy(self):
        """构造本次运行生效的不可变策略（路径/密钥自动进脱敏表）。"""
        from .egress import build_policy

        return build_policy(
            self.egress,
            allow_public=self.egress_allow_public,
            redact_paths=(str(self.workspace), str(self.home)),
            secrets=(self.api_key,),
        )


def system_prompt(contract_path: Path | None = None) -> str:
    """system prompt = CONTRACT.md 原文。

    这是 prompt 的**稳定前缀**：字节不变 → DeepSeek 前缀缓存可命中（价差 50×）。
    因此不要在调用侧拼接时间戳/随机 id；有任何改动请递增 CONTRACT.md 内的版本号。
    """
    path = Path(contract_path) if contract_path else resolve_contract_path()
    text = path.read_text(encoding="utf-8") if path.exists() else "# YARO-CONTRACT (missing)\n"
    return text.rstrip() + "\n"
