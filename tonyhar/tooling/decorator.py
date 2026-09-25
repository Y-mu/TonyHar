"""使用函数和类型提示声明工具。"""

import inspect
import re
import types
from collections.abc import Callable
from functools import update_wrapper
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from .base import BaseTool, ToolPolicy


def _annotation_to_schema(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin is Annotated:
        schema = _annotation_to_schema(arguments[0])
        for metadata in arguments[1:]:
            if isinstance(metadata, str):
                schema["description"] = metadata
            elif isinstance(metadata, dict):
                schema.update(metadata)
        return schema

    if annotation is Any:
        return {}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    if origin is Literal:
        values = list(arguments)
        schema = {"enum": values}
        if values:
            schema.update(_annotation_to_schema(type(values[0])))
        return schema

    if origin in (Union, types.UnionType):
        variants = [
            {"type": "null"}
            if argument is type(None)
            else _annotation_to_schema(argument)
            for argument in arguments
        ]
        return {"anyOf": variants}

    if origin in (list, set, tuple):
        item_type = arguments[0] if arguments else Any
        return {
            "type": "array",
            "items": _annotation_to_schema(item_type),
        }

    if origin is dict:
        value_type = arguments[1] if len(arguments) == 2 else Any
        return {
            "type": "object",
            "additionalProperties": _annotation_to_schema(value_type),
        }

    raise TypeError(f"暂不支持工具参数类型: {annotation!r}")


def _argument_descriptions(docstring: str) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    in_args = False
    for line in docstring.splitlines():
        stripped = line.strip()
        if stripped in {"Args:", "Arguments:", "参数:"}:
            in_args = True
            continue
        if not in_args:
            continue
        if stripped and not line.startswith((" ", "\t")):
            break
        match = re.match(r"^\s*([A-Za-z_]\w*)\s*:\s*(.+)$", line)
        if match:
            descriptions[match.group(1)] = match.group(2).strip()
    return descriptions


class FunctionTool(BaseTool):
    """由普通 Python 函数生成的工具适配器。"""

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        policy: ToolPolicy | None = None,
    ):
        if not inspect.iscoroutinefunction(function):
            raise TypeError("@tool 只接受 async def 工具函数")
        signature = inspect.signature(function)
        type_hints = get_type_hints(function, include_extras=True)
        docstring = inspect.getdoc(function) or ""
        resolved_description = description or docstring
        if not resolved_description:
            raise ValueError("工具函数必须提供 docstring 或 description")

        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []
        argument_docs = _argument_descriptions(docstring)

        for parameter_name, parameter in signature.parameters.items():
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.POSITIONAL_ONLY,
            }:
                raise TypeError("工具函数不支持位置专用参数、*args 或 **kwargs")
            if parameter_name not in type_hints:
                raise TypeError(
                    f"工具参数 '{parameter_name}' 必须提供类型提示"
                )

            schema = _annotation_to_schema(type_hints[parameter_name])
            if parameter_name in argument_docs:
                schema.setdefault("description", argument_docs[parameter_name])
            if parameter.default is inspect.Parameter.empty:
                required.append(parameter_name)
            else:
                schema["default"] = parameter.default
            properties[parameter_name] = schema

        self.name = name or function.__name__
        self.description = resolved_description
        self.parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
        self.policy = policy or ToolPolicy()
        self._function = function
        update_wrapper(self, function)

    async def execute(self, **kwargs) -> Any:
        return await self._function(**kwargs)


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    policy: ToolPolicy | None = None,
):
    """把带类型提示和 docstring 的函数转换为 FunctionTool。"""

    def decorator(target: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(
            target,
            name=name,
            description=description,
            policy=policy,
        )

    if function is None:
        return decorator
    return decorator(function)
