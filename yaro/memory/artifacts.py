"""内容寻址产物库（CAS）。

不变式：**同样的字节只存一份，永不原地覆盖**。模型/被删产物都只是指针的移动。

布局::

    <root>/objects/<sha[:2]>/<sha>.<ext>
    <root>/index.jsonl           # 每个 sha 一行元数据，追加写
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from ..contracts import ArtifactRef, artifact_uri, object_key, sha256_hex

_EXT_BY_MEDIA = {
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/webp": "webp",
    "application/json": "json",
    "text/csv": "csv",
    "text/plain": "txt",
    "application/npy": "npy",
}


@dataclass(frozen=True)
class _Entry:
    sha256: str
    ext: str
    media_type: str
    size: int


class ArtifactStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.objects_dir = self.root / "objects"
        self.index_path = self.root / "index.jsonl"
        self._entries: dict[str, _Entry] = {}
        self.objects_dir.mkdir(parents=True, exist_ok=True)
        self._load_index()

    # ------------------------------------------------------------------ #
    # 索引
    # ------------------------------------------------------------------ #

    def _load_index(self) -> None:
        if not self.index_path.exists():
            return
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                entry = _Entry(
                    sha256=str(data["sha256"]),
                    ext=str(data["ext"]),
                    media_type=str(data["media_type"]),
                    size=int(data["size"]),
                )
            except (KeyError, ValueError, TypeError):
                continue  # 索引损坏的一行不该拖垮整个库；verify() 会单独报告
            self._entries[entry.sha256] = entry

    def _append_index(self, entry: _Entry) -> None:
        line = json.dumps(
            {"sha256": entry.sha256, "ext": entry.ext, "media_type": entry.media_type, "size": entry.size},
            ensure_ascii=False,
        )
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #

    def put(
        self,
        data: bytes,
        *,
        run_id: str,
        step: int,
        kind: str,
        media_type: str = "application/octet-stream",
        ext: str | None = None,
    ) -> ArtifactRef:
        sha = sha256_hex(data)
        ext = (ext or _EXT_BY_MEDIA.get(media_type) or "bin").lstrip(".")
        existing = self._entries.get(sha)
        if existing is not None:
            return ArtifactRef(
                uri=artifact_uri(run_id, step, kind, sha, existing.ext),
                sha256=sha,
                media_type=existing.media_type,
                size=existing.size,
            )

        path = self.objects_dir / object_key(sha, ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():  # 幂等：同 sha 重复写入直接跳过
            fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-")
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_name, path)
            except BaseException:
                Path(tmp_name).unlink(missing_ok=True)
                raise

        entry = _Entry(sha256=sha, ext=ext, media_type=media_type, size=len(data))
        self._entries[sha] = entry
        self._append_index(entry)
        return ArtifactRef(
            uri=artifact_uri(run_id, step, kind, sha, ext),
            sha256=sha,
            media_type=media_type,
            size=len(data),
        )

    def put_file(
        self,
        source: Path | str,
        *,
        run_id: str,
        step: int,
        kind: str,
        media_type: str = "application/octet-stream",
        ext: str | None = None,
    ) -> ArtifactRef:
        source = Path(source)
        suffix = ext or source.suffix.lstrip(".") or None
        return self.put(
            source.read_bytes(),
            run_id=run_id,
            step=step,
            kind=kind,
            media_type=media_type,
            ext=suffix,
        )

    def put_stream(
        self,
        handle: BinaryIO,
        *,
        run_id: str,
        step: int,
        kind: str,
        media_type: str = "application/octet-stream",
        ext: str | None = None,
    ) -> ArtifactRef:
        return self.put(
            handle.read(),
            run_id=run_id,
            step=step,
            kind=kind,
            media_type=media_type,
            ext=ext,
        )

    # ------------------------------------------------------------------ #
    # 读取与校验
    # ------------------------------------------------------------------ #

    def get(self, ref: ArtifactRef) -> bytes:
        return self.path(ref).read_bytes()

    def path(self, ref: ArtifactRef) -> Path:
        entry = self._entries.get(ref.sha256)
        ext = entry.ext if entry is not None else (ref.uri.rsplit(".", 1)[-1] if "." in ref.uri else "bin")
        return self.objects_dir / object_key(ref.sha256, ext)

    def verify(self, ref: ArtifactRef) -> bool:
        path = self.path(ref)
        if not path.exists():
            return False
        return sha256_hex(path.read_bytes()) == ref.sha256

    def exists(self, sha: str) -> bool:
        return sha in self._entries

    def entries(self) -> dict[str, dict[str, Any]]:
        """按 sha256 索引的产物元数据。

        server 的 ``/api/artifacts`` 列表与 ``/api/artifacts/<sha>`` 取字节都按
        sha 查找，因此这里返回 dict 而不是可迭代的元数据行。
        """
        return {
            entry.sha256: {
                "sha256": entry.sha256,
                "ext": entry.ext,
                "media_type": entry.media_type,
                "size": entry.size,
            }
            for entry in self._entries.values()
        }

    def stats(self) -> dict[str, int]:
        """去重后的实际占用——这是"内容寻址"最直观的收益观测点。"""
        return {
            "objects": len(self._entries),
            "bytes": sum(entry.size for entry in self._entries.values()),
        }

    def __len__(self) -> int:
        return len(self._entries)
