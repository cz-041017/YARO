"""MCP 对内消费的测试：包装逻辑（纯）+ 真实 stdio 服务器连接（端到端）。"""

from __future__ import annotations

import sys
from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from yaro.config import parse_mcp_servers
from yaro.contracts import ContractError, ToolResult
from yaro.mcp import (
    McpServer,
    McpTool,
    _build_connections,
    _extract_text,
    _pydantic_schema_to_openai,
    _tool_spec,
    mcp_tools,
    servers_from_config,
)


# --------------------------------------------------------------------------- #
# 假工具（不依赖 MCP 连接，直接测包装逻辑）
# --------------------------------------------------------------------------- #


class _EchoInput(BaseModel):
    text: str = Field(description="要回显的文本")
    count: int | None = Field(default=None, description="可选次数")


class _EchoTool(BaseTool):
    name: str = "echo"
    description: str = "把文本回显给模型"
    args_schema: type[BaseModel] = _EchoInput

    def _run(self, text: str, count: int | None = None) -> str:
        return f"echo: {text}"


class _BoomTool(BaseTool):
    name: str = "boom"
    description: str = "必炸"
    args_schema: type[BaseModel] = _EchoInput

    def _run(self, **kwargs: Any) -> str:
        raise RuntimeError("kaboom")


def _make_tool() -> McpTool:
    return McpTool(_EchoTool())


# --------------------------------------------------------------------------- #
# 纯逻辑
# --------------------------------------------------------------------------- #


def test_pydantic_schema_strips_defs_title_and_anyof_null():
    schema = {
        "$defs": {"Inner": {"type": "object", "title": "Inner", "properties": {"x": {"type": "integer"}}}},
        "type": "object",
        "title": "Root",
        "properties": {
            "opt": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
            "ref": {"$ref": "#/$defs/Inner"},
        },
    }
    out = _pydantic_schema_to_openai(schema)
    assert "title" not in out and "$defs" not in out
    assert out["properties"]["opt"] == {"type": "integer"}  # anyOf [T, null] → T
    assert out["properties"]["ref"]["properties"]["x"] == {"type": "integer"}  # $ref 内联


def test_tool_spec_from_args():
    spec = _tool_spec(_EchoTool())
    assert spec.name == "echo"
    assert spec.description == "把文本回显给模型"
    params = spec.parameters
    assert params["type"] == "object"
    assert "text" in params["properties"]
    assert params["properties"]["text"]["type"] == "string"


def test_extract_text_handles_shapes():
    assert _extract_text("hello") == "hello"
    assert _extract_text({"a": 1}) == '{"a": 1}'

    class _Text:
        text = "abc"

    class _Image(dict):
        pass

    img = _Image({"type": "image"})
    assert _extract_text([_Text(), img]) == "abc\n[image]"
    assert _extract_text(123) == "123"


def test_mcp_tool_run_ok_and_error():
    ok = McpTool(_EchoTool())
    result = await_result(ok.run(text="hi", count=None))
    assert result.ok and result.summary == "echo: hi"

    bad = McpTool(_BoomTool())
    result = await_result(bad.run(text="x"))
    assert not result.ok
    assert "RuntimeError: kaboom" in result.summary


def await_result(coro):
    import asyncio

    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# 连接层配置校验
# --------------------------------------------------------------------------- #


def test_build_connections_validates():
    with pytest.raises(ValueError, match="没给 command"):
        _build_connections([McpServer(name="x", transport="stdio")])
    with pytest.raises(ValueError, match="没给 url"):
        _build_connections([McpServer(name="x", transport="sse")])
    with pytest.raises(ValueError, match="未知 MCP transport"):
        _build_connections([McpServer(name="x", transport="carrier-pigeon")])
    conns = _build_connections(
        [McpServer(name="s", transport="stdio", command="python", args=["x.py"], env={"K": "V"})]
    )
    assert conns["s"] == {"transport": "stdio", "command": "python", "args": ["x.py"], "env": {"K": "V"}}


def test_servers_from_config_parses_dict_and_passthrough():
    """config 的 YARO_MCP_SERVERS dict → McpServer；对象直通；非法类型报错。"""
    parsed = servers_from_config(
        [
            {"name": " math ", "transport": "stdio", "command": "python", "args": ["s.py"], "env": {"K": "1"}, "url": ""},
            {"name": "remote", "transport": "sse", "url": "http://x/y"},
        ]
    )
    assert isinstance(parsed[0], McpServer)
    assert parsed[0].name == "math"  # strip
    assert parsed[0].command == "python"
    assert parsed[0].args == ["s.py"]  # 值强制 str
    assert parsed[0].env == {"K": "1"}
    assert parsed[1].transport == "sse" and parsed[1].url == "http://x/y"

    # McpServer 对象直通（代码里直接构造不重复包）
    obj = McpServer(name="o", command="cmd")
    assert servers_from_config([obj]) == [obj]

    # 空 → 空
    assert servers_from_config([]) == []
    assert servers_from_config(None) == []

    with pytest.raises(ValueError, match="必须是对象或 McpServer"):
        servers_from_config(["not-an-object"])


def test_servers_from_config_type_confusion_raises_valueerror():
    """M-2/L-2 回归：args/env 字段类型混淆必须抛 ValueError（可被 server.py 降级），
    而不是 AttributeError/TypeError 穿透炸掉整次运行。"""
    with pytest.raises(ValueError, match="args 必须是列表"):
        servers_from_config([{"name": "m", "command": "python", "args": "script.py"}])
    with pytest.raises(ValueError, match="args 必须是列表"):
        servers_from_config([{"name": "m", "command": "python", "args": {"a": 1}}])
    with pytest.raises(ValueError, match="env 必须是对象"):
        servers_from_config([{"name": "m", "command": "python", "env": "FOO=bar"}])
    with pytest.raises(ValueError, match="env 必须是对象"):
        servers_from_config([{"name": "m", "command": "python", "env": ["FOO=bar"]}])


def test_parse_mcp_servers_validation():
    """config.parse_mcp_servers 只做 JSON 层校验；字段级校验留给 servers_from_config。"""
    assert parse_mcp_servers("") == ()
    assert parse_mcp_servers(None) == ()
    parsed = parse_mcp_servers('[{"name":"m","command":"python"}]')
    assert parsed == ({"name": "m", "command": "python"},)

    with pytest.raises(ContractError, match="不是合法 JSON"):
        parse_mcp_servers("{not json")
    with pytest.raises(ContractError, match="必须是 JSON 数组"):
        parse_mcp_servers('{"name":"m"}')
    with pytest.raises(ContractError, match="必须是对象"):
        parse_mcp_servers('["just-a-string"]')


# --------------------------------------------------------------------------- #
# 端到端：真实 FastMCP stdio 服务器
# --------------------------------------------------------------------------- #

_SERVER_CODE = '''\
from mcp.server.fastmcp import FastMCP

app = FastMCP("math")


@app.tool()
def add(a: int, b: int) -> int:
    """两个整数相加。"""
    return a + b


if __name__ == "__main__":
    app.run(transport="stdio")
'''


async def _test_stdio_real_server(tmp_path):
    server_py = tmp_path / "mcp_server.py"
    server_py.write_text(_SERVER_CODE, encoding="utf-8")
    servers = [McpServer(name="math", transport="stdio", command=sys.executable, args=[str(server_py)])]

    async with mcp_tools(servers) as tools:
        assert len(tools) == 1
        tool = tools[0]
        assert tool.spec.name == "add"
        assert tool.spec.description == "两个整数相加。"
        result = await tool.run(a=2, b=3)
        assert result.ok
        assert "5" in result.summary


def test_mcp_tools_stdio_real_server(tmp_path):
    import asyncio

    asyncio.run(_test_stdio_real_server(tmp_path))
