"""MCP 对内消费：把外部 MCP 服务器（社区视觉工具）的工具收进 YARO 的 ToolRegistry。

决策（m03681）：MCP 只做「对内消费」——YARO 作为 MCP client，连接社区视觉工具暴露的
MCP server，把它们的工具当一等工具用；不对外暴露 YARO 自身工具（那是单独的 MCP server，
属 P2）。

用法（server.py 引擎内）：:

    from yaro.mcp import mcp_tools

    async with mcp_tools(settings.mcp_servers) as extra:
        registry = ToolRegistry(build_default_tools(...) + extra)
        await run_agent(...)

``mcp_tools`` 是 async context manager，只为统一调用方 API + 延迟 import（无 MCP
服务器时一个字节的 MCP 适配器都不 import）。langchain-mcp-adapters 0.1+ 的语义是
**每次工具调用新建一个会话**（见 ``MultiServerMCPClient.get_tools`` 的 docstring），
所以工具对象是自包含的、没有需要手动管理的会话生命周期；把工具带出 ``async with``
块外再调用也安全（每次调用各自起会话）。
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

from .contracts import ToolResult, ToolSpec

if TYPE_CHECKING:  # 仅类型检查；运行时注解是字符串，不 import langchain（保持模块零第三方依赖）
    from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class McpServer:
    """一个 MCP 服务器配置。``transport`` 支持 ``"stdio"``（本地子进程）与 ``"sse"``（远程）。"""

    name: str
    transport: str = "stdio"
    command: str = ""            # stdio: 可执行文件
    args: list[str] = field(default_factory=list)   # stdio: 启动参数
    env: dict[str, str] = field(default_factory=dict)  # stdio: 注入子进程的环境
    url: str = ""                # sse: 服务器 URL


def _build_connections(servers: list[McpServer]) -> dict[str, dict[str, Any]]:
    """把 McpServer 列表转成 MultiServerMCPClient 需要的 connections dict。"""
    connections: dict[str, dict[str, Any]] = {}
    for s in servers:
        if not s.name:
            raise ValueError("MCP 服务器必须有 name")
        if s.transport == "stdio":
            if not s.command:
                raise ValueError(f"MCP 服务器 {s.name!r} 用 stdio transport 但没给 command")
            conn: dict[str, Any] = {
                "transport": "stdio",
                "command": s.command,
                "args": list(s.args),
            }
            if s.env:
                conn["env"] = dict(s.env)
            connections[s.name] = conn
        elif s.transport == "sse":
            if not s.url:
                raise ValueError(f"MCP 服务器 {s.name!r} 用 sse transport 但没给 url")
            connections[s.name] = {"transport": "sse", "url": s.url}
        else:
            raise ValueError(
                f"未知 MCP transport {s.transport!r}（服务器 {s.name!r}，支持 stdio/sse）"
            )
    return connections


def _pydantic_schema_to_openai(schema: dict[str, Any]) -> dict[str, Any]:
    """把 pydantic v2 的 model_json_schema() 转成 OpenAI function calling 能吃的 JSON Schema。

    处理三件 OpenAI 不认的事：``$defs`` 引用、可选字段的 ``anyOf [T, null]``、以及
    pydantic 塞进去的 ``title`` 装饰字段。
    """
    defs = schema.get("$defs", {}) if isinstance(schema, dict) else {}

    def clean(node: Any) -> Any:
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for key, val in node.items():
                if key in ("title", "examples", "$defs"):
                    continue
                if key == "$ref":
                    name = str(val).split("/")[-1]
                    out.update(clean(defs.get(name, {"type": "object"})))
                elif key == "anyOf":
                    non_null = [x for x in val if not (isinstance(x, dict) and x.get("type") == "null")]
                    if len(non_null) == 1:
                        out.update(clean(non_null[0]))
                    else:
                        out[key] = [clean(x) for x in non_null]
                else:
                    out[key] = clean(val)
            return out
        if isinstance(node, list):
            return [clean(x) for x in node]
        return node

    return clean(schema)


def _tool_spec(tool: BaseTool) -> ToolSpec:
    """把 LangChain/MCP 工具转成 YARO ToolSpec。"""
    # 优先用 .args（LangChain 已整理的扁平 dict），退化到 args_schema。
    if getattr(tool, "args", None):
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": dict(tool.args),
            "additionalProperties": False,
        }
    elif getattr(tool, "args_schema", None) is not None:
        parameters = _pydantic_schema_to_openai(tool.args_schema.model_json_schema())
    else:
        parameters = {"type": "object", "properties": {}}
    return ToolSpec(
        name=tool.name,
        description=(tool.description or "").strip(),
        parameters=parameters,
    )


def _extract_text(value: Any) -> str:
    """把 MCP/LangChain 工具的返回值压成一段给模型看的文本。

    覆盖 str / dict（结构化输出）/ MCP ContentBlock 列表（TextContent/ImageContent）
    / langchain BaseMessage。图片块只留占位符——图的 uri 要由调用方走 ArtifactRef，
    这里不解析像素。
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "image":
                parts.append("[image]")
                continue
            text = _extract_text(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)[:4000]
    if hasattr(value, "content"):
        return _extract_text(getattr(value, "content"))
    if hasattr(value, "text"):
        return _extract_text(getattr(value, "text"))
    return str(value)


def _to_result(value: Any) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    text = _extract_text(value)
    return ToolResult(ok=True, summary=text[:4000] or "(工具无文本输出)")


class McpTool:
    """把 LangChain MCP tool 包装成 YARO Tool（``run`` 返回 ToolResult，不抛裸异常）。"""

    def __init__(self, tool: BaseTool):
        self.spec = _tool_spec(tool)
        self._tool = tool

    async def run(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self._tool.ainvoke(kwargs)
            return _to_result(result)
        except Exception as exc:  # noqa: BLE001 - 工具边界兜底，不让单个 MCP 工具炸掉整轮
            return ToolResult(ok=False, summary=f"{type(exc).__name__}: {exc}")


@asynccontextmanager
async def mcp_tools(servers: list[McpServer]) -> AsyncIterator[list[McpTool]]:
    """连接所有 MCP 服务器，yield 包装好的工具列表。

    空列表直接 yield []（不 import MCP 适配器，demo/无 MCP 环境不付出连接成本）。
    """
    if not servers:
        yield []
        return

    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(_build_connections(servers))
    raw_tools = await client.get_tools()
    yield [McpTool(t) for t in raw_tools]
