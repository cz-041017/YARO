"""Skills：过程性记忆（SKILL.md 渐进披露）。

参照 Claude Code 的 skills 机制与 arXiv:2609.00006 对 11 个 harness 的实证结论：

- 采用率 9/11 > MCP 8/11；canonical 路径 ``.agents/skills/`` 被 6 个系统接受。
- 9 个采用者里 8 个**只预加载元数据、正文按需取**（渐进披露）——这是关键：把技能
  正文全塞进 system prompt 会烧 token 且稀释注意力，正确的做法是"先给目录，命中再取正文"。

本模块零依赖：手写一个极小的 YAML frontmatter 解析器（只支持标量 / 引号标量 /
内联 ``[a, b]`` / 块状 ``key:`` + ``  - item`` 列表），不引入 PyYAML。

格式示例（``.agents/skills/dataset-audit/SKILL.md``）：:

    ---
    name: dataset-audit
    description: 体检数据集，找脏数据（截断图/缺标注/越界框/重复图）
    when_to_use:
      - 数据集需要体检时
      - 收到一批新图片要检查时
    allowed-tools: [dataset_scan, dataset_audit, dataset_render]
    license: MIT
    version: 1
    ---
    # 数据集体检
    ...正文，命中时才读...
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# --------------------------------------------------------------------------- #
# 极简 YAML frontmatter 解析
# --------------------------------------------------------------------------- #

def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """解析 ``---\\n...\\n---`` 头。无 frontmatter 返回 (空 dict, 原文)。"""
    if not text:
        return {}, text
    # 容忍 BOM 与前导空白（不改变"无 frontmatter"时的返回原文）。
    probe = text.lstrip("\ufeff \t\r\n")
    if not probe.startswith("---"):
        return {}, text
    start = text.index("---")
    rest = text[start + 3 :].lstrip("\r\n")
    idx = rest.find("\n---")
    if idx == -1:
        # 没有闭合分隔符：整段都是正文，当作无 frontmatter。
        return {}, text
    front = rest[:idx]
    body = rest[idx + 4 :].lstrip("\r\n")
    return _parse_meta(front), body


def _split_inline_list(value: str) -> list[str]:
    """按逗号切内联列表，尊重引号（引号内逗号不切）。"""
    items: list[str] = []
    cur = ""
    in_quote = False
    quote = ""
    for ch in value:
        if ch in "\"'":
            if in_quote and ch == quote:
                in_quote = False
            elif not in_quote:
                in_quote, quote = True, ch
            else:
                cur += ch
        elif ch == "," and not in_quote:
            if cur.strip():
                items.append(cur.strip().strip("\"'"))
            cur = ""
        else:
            cur += ch
    if cur.strip():
        items.append(cur.strip().strip("\"'"))
    return items


def _parse_meta(front: str) -> dict[str, Any]:
    """解析 frontmatter 的极小 YAML 子集。"""
    meta: dict[str, Any] = {}
    list_key: str | None = None
    list_buf: list[str] | None = None

    def _flush() -> None:
        nonlocal list_key, list_buf
        if list_key is not None and list_buf is not None:
            meta[list_key] = list_buf
        list_key, list_buf = None, None

    for raw in front.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            item = line[2:].strip().strip("\"'")
            if list_key is not None and list_buf is not None:
                list_buf.append(item)
            continue
        if ":" not in line:
            continue
        _flush()
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            list_key, list_buf = key, []
        elif value.startswith("[") and value.endswith("]"):
            meta[key] = _split_inline_list(value[1:-1])
        else:
            meta[key] = value.strip("\"'")
    _flush()
    return meta


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    return (str(value),)


# --------------------------------------------------------------------------- #
# Skill 与注册表
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    when_to_use: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    license: str = ""
    version: str = ""

    def body(self) -> str:
        """正文（渐进披露：只在命中时读）。"""
        return self.path.read_text(encoding="utf-8", errors="replace")


def _skill_from_md(path: Path) -> Skill | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    meta, _ = parse_frontmatter(text)
    name = meta.get("name")
    if not name:
        return None
    return Skill(
        name=str(name),
        description=str(meta.get("description", "")),
        path=path,
        when_to_use=_as_tuple(meta.get("when_to_use")),
        allowed_tools=_as_tuple(meta.get("allowed-tools") or meta.get("allowed_tools")),
        license=str(meta.get("license", "")),
        version=str(meta.get("version", "")),
    )


def discover_skills(*dirs: str | Path) -> list[Skill]:
    """扫描若干目录下的 ``**/*.md``，按 name 去重（先出现的优先），返回按 name 排序。"""
    found: dict[str, Skill] = {}
    for d in dirs:
        if not d or not Path(d).is_dir():
            continue
        for p in sorted(Path(d).rglob("*.md")):
            try:
                skill = _skill_from_md(p)
            except OSError:
                continue  # 目录同名 / 权限拒绝等，单文件隔离不拖垮整目录
            if skill is not None:
                found.setdefault(skill.name, skill)
    return sorted(found.values(), key=lambda s: s.name)


class SkillRegistry:
    """技能索引：目录元数据常驻、正文按需取。"""

    def __init__(self, skills: Iterable[Skill] = ()):
        self._skills: dict[str, Skill] = {s.name: s for s in skills}
        self._bodies: dict[str, str] = {}

    def add(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def names(self) -> list[str]:
        return sorted(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def index(self) -> list[dict[str, Any]]:
        """元数据目录（给上下文用，不含正文）。"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "when_to_use": list(s.when_to_use),
            }
            for s in sorted(self._skills.values(), key=lambda s: s.name)
        ]

    def body(self, name: str) -> str | None:
        skill = self._skills.get(name)
        if skill is None:
            return None
        if name not in self._bodies:
            self._bodies[name] = skill.body()
        return self._bodies[name]

    def render_index(self, max_chars: int = 1500) -> str:
        """压缩目录文本（system prompt 用），按整行截断、绝不在行中间切断。"""
        lines = []
        used = 0
        for s in sorted(self._skills.values(), key=lambda s: s.name):
            line = f"- {s.name}: {s.description}"
            cost = len(line) + (1 if lines else 0)  # 换行符
            if used + cost > max_chars:
                break
            lines.append(line)
            used += cost
        return "\n".join(lines)

    def match(self, query: str) -> list[Skill]:
        """按关键词命中 name / description / when_to_use（简单子串匹配，非向量检索）。

        忽略长度 < 2 的词（"a"/"的" 这类单字会命中全部）。
        """
        q = query.lower()
        if not q:
            return []
        terms = [t for t in q.split() if len(t) >= 2]
        if not terms:
            return []
        hits = []
        for s in sorted(self._skills.values(), key=lambda s: s.name):
            hay = " ".join([s.name, s.description, *s.when_to_use]).lower()
            if any(term in hay for term in terms):
                hits.append(s)
        return hits
