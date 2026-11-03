"""审计账本 AuditLedger 的离线单测。"""

from __future__ import annotations

import json

from yaro.audit import GENESIS, AuditEvent, AuditLedger


def test_append_builds_digest_chain(tmp_path) -> None:
    ledger = AuditLedger.create(tmp_path / "events.jsonl")
    e1 = ledger.append("label", {"count": 3})
    e2 = ledger.append("train", {"ok": True})

    assert e1.prev_digest == GENESIS
    assert e2.prev_digest == e1.digest
    report = ledger.verify()
    assert report["ok"] is True
    assert report["events"] == 2
    assert report["head_digest"] == e2.digest


def test_reopen_preserves_chain(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = AuditLedger.create(path)
    ledger.append("label", {"count": 1})
    ledger.append("analyze", {"stop": "done"})

    reopened = AuditLedger.open(path)
    assert len(reopened) == 2
    assert reopened.verify()["ok"] is True
    assert reopened.head_digest() == ledger.head_digest()


def test_tamper_detected(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = AuditLedger.create(path)
    ledger.append("label", {"count": 1})

    # 篡改 payload 但不重算 digest → verify 必须报错
    event = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    event["payload"]["count"] = 999
    path.write_text(json.dumps(event, ensure_ascii=False) + "\n", encoding="utf-8")

    reopened = AuditLedger.open(path)
    assert reopened.verify()["ok"] is False


def test_corrupt_line_isolated(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = AuditLedger.create(path)
    ledger.append("label", {"count": 1})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"broken": \n')  # 半行 JSON（模拟崩溃）
    ledger.append("train", {"ok": True})

    reopened = AuditLedger.open(path)
    assert len(reopened) == 2  # 坏行被隔离，label + train 还在
    assert reopened.verify()["ok"] is True  # train.prev_digest 仍接 label
    assert path.with_suffix(".corrupt.jsonl").exists()


def test_from_dict_roundtrip() -> None:
    e = AuditEvent(seq=1, event_id="e1", ts=1.0, type="x", payload={"a": 1}, prev_digest=GENESIS).seal()
    e2 = AuditEvent.from_dict(e.to_dict())
    assert e2.to_dict() == e.to_dict()


def test_empty_ledger_verify_ok(tmp_path) -> None:
    ledger = AuditLedger.create(tmp_path / "events.jsonl")
    report = ledger.verify()
    assert report["ok"] is True
    assert report["events"] == 0
    assert report["head_digest"] == GENESIS
