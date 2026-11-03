"""内容寻址产物库的单元测试。"""

from __future__ import annotations

from yaro.memory.artifacts import ArtifactStore


def test_put_is_content_addressed_and_deduped(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    first = store.put(b"hello", run_id="r1", step=1, kind="text", media_type="text/plain")
    second = store.put(b"hello", run_id="r1", step=2, kind="text", media_type="text/plain")

    assert first.sha256 == second.sha256
    assert first.uri != second.uri  # step 不同 → 地址不同，但指向同一份字节
    assert len(store) == 1
    assert store.stats()["bytes"] == 5


def test_put_same_content_twice_does_not_duplicate_file(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    store.put(b"abc", run_id="r1", step=1, kind="mask", media_type="image/png")
    store.put(b"abc", run_id="r1", step=1, kind="mask", media_type="image/png")
    files = list((tmp_path / "artifacts" / "objects").rglob("*.*"))
    assert len([path for path in files if path.is_file()]) == 1


def test_get_and_verify(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put(b"\x89PNG fake", run_id="r1", step=3, kind="render", media_type="image/png")
    assert store.get(ref) == b"\x89PNG fake"
    assert store.verify(ref) is True


def test_verify_detects_corruption(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put(b"original", run_id="r1", step=1, kind="text", media_type="text/plain")
    path = store.path(ref)
    path.write_bytes(b"corrupted")
    assert store.verify(ref) is False


def test_reopen_rebuilds_index(tmp_path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put(b"persist me", run_id="r1", step=1, kind="text", media_type="text/plain")
    reopened = ArtifactStore(tmp_path / "artifacts")
    assert len(reopened) == 1
    assert reopened.get(ref) == b"persist me"


def test_put_file_infers_extension(tmp_path) -> None:
    source = tmp_path / "k.json"
    source.write_text('{"fx": 1000}', encoding="utf-8")
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put_file(source, run_id="r1", step=1, kind="intrinsics", media_type="application/json")
    assert ref.uri.endswith(".json")
    assert ref.size == source.stat().st_size
