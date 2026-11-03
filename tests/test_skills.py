"""skills 模块（SKILL.md 渐进披露）的离线单测。"""

from __future__ import annotations

from pathlib import Path

import pytest

from yaro.skills import (
    Skill,
    SkillRegistry,
    discover_skills,
    parse_frontmatter,
)


SAMPLE = """---
name: dataset-audit
description: 体检数据集找脏数据
when_to_use:
  - 数据集需要体检时
  - 收到一批新图片
allowed-tools: [dataset_scan, dataset_audit]
license: MIT
version: 1
---
# 数据集体检

正文内容，命中时才读。
"""


def test_parse_frontmatter_scalar_and_lists() -> None:
    meta, body = parse_frontmatter(SAMPLE)
    assert meta["name"] == "dataset-audit"
    assert meta["description"] == "体检数据集找脏数据"
    assert meta["when_to_use"] == ["数据集需要体检时", "收到一批新图片"]
    assert meta["allowed-tools"] == ["dataset_scan", "dataset_audit"]
    assert meta["license"] == "MIT"
    assert meta["version"] == "1"
    assert body.startswith("# 数据集体检")
    assert "正文内容" in body


def test_parse_frontmatter_absent() -> None:
    meta, body = parse_frontmatter("just a markdown file\nno frontmatter")
    assert meta == {}
    assert "no frontmatter" in body


def test_parse_frontmatter_quoted() -> None:
    meta, _ = parse_frontmatter('---\nname: "x"\ndescription: \'y\'\n---\nbody')
    assert meta["name"] == "x"
    assert meta["description"] == "y"


def test_discover_skills(tmp_path: Path) -> None:
    (tmp_path / "dataset-audit").mkdir()
    (tmp_path / "dataset-audit" / "SKILL.md").write_text(SAMPLE, encoding="utf-8")
    # 无 frontmatter 的普通 md 不算 skill
    (tmp_path / "notes.md").write_text("# notes\nnot a skill", encoding="utf-8")

    skills = discover_skills(tmp_path)
    assert [s.name for s in skills] == ["dataset-audit"]
    assert skills[0].description == "体检数据集找脏数据"
    assert skills[0].when_to_use == ("数据集需要体检时", "收到一批新图片")
    assert skills[0].allowed_tools == ("dataset_scan", "dataset_audit")


def test_discover_skills_dedupes_by_name(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    (tmp_path / "a" / "SKILL.md").write_text(
        "---\nname: s\ndescription: first\n---\nA", encoding="utf-8"
    )
    (tmp_path / "b" / "SKILL.md").write_text(
        "---\nname: s\ndescription: second\n---\nB", encoding="utf-8"
    )
    skills = discover_skills(tmp_path / "a", tmp_path / "b")
    assert len(skills) == 1  # 先出现（a）优先
    assert skills[0].description == "first"


def test_registry_progressive_disclosure(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(SAMPLE, encoding="utf-8")
    skill = discover_skills(tmp_path)[0]
    reg = SkillRegistry([skill])

    assert "dataset-audit" in reg
    assert reg.get("dataset-audit") is skill
    assert reg.index()[0]["name"] == "dataset-audit"
    # 渐进披露：正文按需读，第一次读后缓存
    body = reg.body("dataset-audit")
    assert body is not None and "正文内容" in body
    assert reg.body("missing") is None


def test_render_index_and_match(tmp_path: Path) -> None:
    (tmp_path / "SKILL.md").write_text(SAMPLE, encoding="utf-8")
    reg = SkillRegistry(discover_skills(tmp_path))

    idx = reg.render_index()
    assert "dataset-audit" in idx
    assert "体检数据集" in idx

    assert [s.name for s in reg.match("数据集")] == ["dataset-audit"]
    assert [s.name for s in reg.match("audit")] == ["dataset-audit"]
    assert reg.match("不存在的词") == []
    assert reg.match("") == []
    # 单字查询不命中全部（len < 2 被忽略）
    assert reg.match("a") == []


def test_crlf_frontmatter_body() -> None:
    meta, body = parse_frontmatter(
        "---\r\nname: x\r\ndescription: d\r\n---\r\n# 标题\r\n正文"
    )
    assert meta["name"] == "x"
    assert body.startswith("# 标题")  # 不能残留 \r\n
    assert "正文" in body


def test_leading_bom_and_whitespace() -> None:
    meta, _ = parse_frontmatter("\ufeff  \n---\nname: x\n---\nbody")
    assert meta["name"] == "x"


def test_colon_in_value() -> None:
    meta, _ = parse_frontmatter("---\nname: x\ndescription: 用: 冒号\n---\nbody")
    assert meta["description"] == "用: 冒号"


def test_inline_list_with_quoted_comma() -> None:
    meta, _ = parse_frontmatter('---\nname: x\nwhen_to_use: ["a, b", "c"]\n---\nbody')
    assert meta["when_to_use"] == ["a, b", "c"]


def test_malformed_no_closing_delimiter() -> None:
    meta, body = parse_frontmatter("---\nname: x\nno closing")
    assert meta == {}
    assert "no closing" in body


def test_discover_skips_dir_named_md(tmp_path: Path) -> None:
    # 一个叫 x.md 的目录（rglob 会匹配到它），read_text 会抛 IsADirectoryError
    (tmp_path / "x.md").mkdir()
    (tmp_path / "SKILL.md").write_text(SAMPLE, encoding="utf-8")
    skills = discover_skills(tmp_path)
    assert [s.name for s in skills] == ["dataset-audit"]
