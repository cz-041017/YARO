"""密钥保护：不落日志、不落事件、不进 shell 历史、不回显在界面上。

读取顺序（先到先用）：

1. 环境变量 ``DEEPSEEK_API_KEY``（CI / 临时会话首选）
2. 操作系统钥匙串（装了 ``keyring`` 才有：Windows Credential Manager /
   macOS Keychain / Secret Service）
3. ``<home>/secrets.json``（权限 0600，``yaro key set`` 用 getpass 写入，
   明文不过 shell 历史）
4. 仓库 ``.env``（兼容历史用法，不推荐长期使用）

任何对外展示都只走 :func:`mask`；任何可能进日志/事件的文本都要过
:func:`scrub`。
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any

KEYRING_SERVICE = "yaro"
KEYRING_USERNAME = "deepseek-api-key"
SECRETS_FILENAME = "secrets.json"


def mask(value: str | None) -> str:
    """只露头尾，永不返回完整密钥。"""
    if not value:
        return "（未配置）"
    if len(value) <= 10:
        return "***"
    return f"{value[:4]}…{value[-4:]}"


def secrets_file(home: Path | str) -> Path:
    return Path(home) / SECRETS_FILENAME


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _chmod_600(path: Path) -> None:
    """尽力收紧权限：Windows 上 os.chmod 只影响只读位，POSIX 上才是真实权限。"""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:  # pragma: no cover - 某些文件系统不支持
        pass


def _write_json(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _chmod_600(tmp)
    os.replace(tmp, path)
    _chmod_600(path)
    return path


# ---------------------------------------------------------------------- #
# 钥匙串（可选依赖：装了就优先用）
# ---------------------------------------------------------------------- #


def keyring_available() -> bool:
    try:
        import keyring  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - 后端损坏也当作不可用
        return False
    try:
        return keyring.get_keyring() is not None
    except Exception:  # noqa: BLE001
        return False


def keyring_get() -> str:
    if not keyring_available():
        return ""
    try:
        import keyring  # type: ignore[import-not-found]

        return (keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME) or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def keyring_set(value: str) -> bool:
    if not keyring_available():
        return False
    try:
        import keyring  # type: ignore[import-not-found]

        keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, value)
        return True
    except Exception:  # noqa: BLE001
        return False


def keyring_clear() -> bool:
    if not keyring_available():
        return False
    try:
        import keyring  # type: ignore[import-not-found]
        from keyring.errors import PasswordDeleteError

        try:
            keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except PasswordDeleteError:
            return True
        return True
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------- #
# 本地文件（0600）
# ---------------------------------------------------------------------- #


def load_local(home: Path | str) -> str:
    data = _read_json(secrets_file(home))
    return str(data.get("deepseek_api_key") or "").strip()


def set_api_key(home: Path | str, value: str) -> Path:
    value = (value or "").strip()
    if not value:
        raise ValueError("密钥不能为空")
    path = secrets_file(home)
    data = _read_json(path)
    data["deepseek_api_key"] = value
    return _write_json(path, data)


def clear_api_key(home: Path | str) -> Path:
    path = secrets_file(home)
    data = _read_json(path)
    data.pop("deepseek_api_key", None)
    data["_note"] = "yaro key clear：本地文件里已无 api key；环境变量/钥匙串请另行清理"
    return _write_json(path, data)


# ---------------------------------------------------------------------- #
# 解析与擦除
# ---------------------------------------------------------------------- #


def resolve_api_key(home: Path | str | None) -> tuple[str, str]:
    """返回 ``(key, source)``；source ∈ env / keyring / file / dotenv / none。"""
    env_value = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if env_value:
        return env_value, "env"
    value = keyring_get()
    if value:
        return value, "keyring"
    if home is not None:
        value = load_local(home)
        if value:
            return value, "file"
    dotenv_value = (os.environ.get("YARO_DOTENV_API_KEY") or "").strip()
    if dotenv_value:
        return dotenv_value, "dotenv"
    return "", "none"


def scrub(text: str, *secrets: str | None) -> str:
    """把密钥从任意文本里擦掉（日志、异常、事件、GUI 报错都必须先过它）。"""
    out = text or ""
    for secret in secrets:
        if secret and len(secret) >= 8:
            out = out.replace(secret, "***")
    return out


def scrub_obj(value: Any, *secrets: str | None) -> Any:
    if isinstance(value, str):
        return scrub(value, *secrets)
    if isinstance(value, list):
        return [scrub_obj(item, *secrets) for item in value]
    if isinstance(value, dict):
        return {scrub_obj(key, *secrets): scrub_obj(item, *secrets) for key, item in value.items()}
    return value


def status(home: Path | str | None) -> dict[str, Any]:
    """给 doctor/GUI 用的状态（不含密钥原文）。"""
    key, source = resolve_api_key(home)
    return {
        "configured": bool(key),
        "source": source,
        "masked": mask(key),
        "keyring": keyring_available(),
        "file": str(secrets_file(home)) if home is not None else "",
    }


__all__ = [
    "KEYRING_SERVICE",
    "KEYRING_USERNAME",
    "SECRETS_FILENAME",
    "clear_api_key",
    "keyring_available",
    "keyring_clear",
    "keyring_get",
    "keyring_set",
    "load_local",
    "mask",
    "resolve_api_key",
    "scrub",
    "scrub_obj",
    "secrets_file",
    "set_api_key",
    "status",
]
