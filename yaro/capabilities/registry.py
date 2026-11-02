"""工具注册表。

两条规矩：

1. 工具抛出的任何异常都必须被兜成 ``ToolResult(ok=False)``，否则一轮崩掉整个会话；
2. 模型给出的 arguments 是字符串，解析失败要给出**可读**的错误（并附带 schema 摘要），
   这样模型下一步能自己改正。
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from ..contracts import ContractError, Tool, ToolResult, ToolSpec, function_schema


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        spec = getattr(tool, "spec", None)
        if not isinstance(spec, ToolSpec):
            raise ContractError(f"{type(tool).__name__} 缺少合法的 ToolSpec")
        if spec.name in self._tools:
            raise ContractError(f"工具重名：{spec.name}")
        self._tools[spec.name] = tool

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        """按 name 排序输出，保证每次请求的 tools 数组字节稳定（前缀缓存要求）。"""
        return [function_schema(self._tools[name].spec) for name in sorted(self._tools)]

    async def call(self, name: str, arguments: str | dict[str, Any]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                summary=f"未知工具 {name!r}；可用工具：{', '.join(sorted(self._tools))}",
                hint="请从可用工具中选择，不要臆造工具名",
            )

        if isinstance(arguments, str):
            try:
                kwargs = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                return ToolResult(
                    ok=False,
                    summary=f"参数不是合法 JSON：{exc.msg}（位置 {exc.pos}）",
                    hint=f"本工具的参数 schema：{json.dumps(tool.spec.parameters, ensure_ascii=False)}",
                )
        else:
            kwargs = arguments

        if not isinstance(kwargs, dict):
            return ToolResult(ok=False, summary=f"参数必须是 JSON object，实际是 {type(kwargs).__name__}")

        try:
            result = await tool.run(**kwargs)
        except TypeError as exc:
            return ToolResult(
                ok=False,
                summary=f"参数不匹配：{exc}",
                hint=f"本工具的参数 schema：{json.dumps(tool.spec.parameters, ensure_ascii=False)}",
            )
        except BaseException as exc:  # noqa: BLE001 - 工具边界必须兜住一切
            return ToolResult(
                ok=False,
                summary=f"{type(exc).__name__}: {exc}",
                degraded=[type(exc).__name__],
                hint="这个失败已记录。请换一种方式或先检查前置条件，不要原样重试。",
            )

        if not isinstance(result, ToolResult):
            return ToolResult(
                ok=False,
                summary=f"工具 {name!r} 返回了非法类型 {type(result).__name__}，应为 ToolResult",
            )
        return result
