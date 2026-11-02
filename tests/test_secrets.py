"""密钥保护测试：掩码、本地 0600 文件、来源优先级、擦除、状态不泄露原文。"""

from __future__ import annotations

import json

from yaro import secrets as sec

FAKE_KEY = "sk-test-1234567890abcdef"


def test_mask_never_returns_full_key():
    assert sec.mask("") == "（未配置）"
    assert sec.mask("short") == "***"
    masked = sec.mask(FAKE_KEY)
    assert FAKE_KEY not in masked
    assert masked.startswith("sk-t") and masked.endswith("cdef")


def test_local_file_roundtrip(tmp_path):
    path = sec.set_api_key(tmp_path, FAKE_KEY)
    assert path.exists()
    assert sec.load_local(tmp_path) == FAKE_KEY
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["deepseek_api_key"] == FAKE_KEY

    sec.clear_api_key(tmp_path)
    assert sec.load_local(tmp_path) == ""
    # 清理后文件仍在（保留 _note 审计痕迹），但不再含明文密钥
    assert FAKE_KEY not in (tmp_path / sec.SECRETS_FILENAME).read_text(encoding="utf-8")


def test_empty_key_rejected(tmp_path):
    try:
        sec.set_api_key(tmp_path, "   ")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("空密钥应被拒绝")


def test_resolve_priority_env_over_file(tmp_path, monkeypatch):
    sec.set_api_key(tmp_path, "sk-from-file-000000")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-from-env-111111")
    monkeypatch.setattr(sec, "keyring_get", lambda: "")
    key, source = sec.resolve_api_key(tmp_path)
    assert key == "sk-from-env-111111"
    assert source == "env"


def test_resolve_file_when_no_env(tmp_path, monkeypatch):
    sec.set_api_key(tmp_path, "sk-from-file-222222")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("YARO_DOTENV_API_KEY", raising=False)
    monkeypatch.setattr(sec, "keyring_get", lambda: "")
    key, source = sec.resolve_api_key(tmp_path)
    assert key == "sk-from-file-222222"
    assert source == "file"


def test_resolve_keyring_before_file(tmp_path, monkeypatch):
    sec.set_api_key(tmp_path, "sk-from-file-333333")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("YARO_DOTENV_API_KEY", raising=False)
    monkeypatch.setattr(sec, "keyring_get", lambda: "sk-from-keyring-444444")
    key, source = sec.resolve_api_key(tmp_path)
    assert key == "sk-from-keyring-444444"
    assert source == "keyring"


def test_scrub_removes_secret_everywhere():
    text = f"failed with {FAKE_KEY} while calling"
    assert FAKE_KEY not in sec.scrub(text, FAKE_KEY)
    payload = {"detail": f"key={FAKE_KEY}", "items": [FAKE_KEY], "n": 3}
    cleaned = sec.scrub_obj(payload, FAKE_KEY)
    assert FAKE_KEY not in json.dumps(cleaned, ensure_ascii=False)
    assert cleaned["n"] == 3


def test_status_hides_raw_key(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(sec, "keyring_get", lambda: "")
    sec.set_api_key(tmp_path, FAKE_KEY)
    status = sec.status(tmp_path)
    assert status["configured"] is True
    assert status["source"] == "file"
    assert FAKE_KEY not in json.dumps(status, ensure_ascii=False)


def test_keyring_probe_is_boolean():
    assert isinstance(sec.keyring_available(), bool)
