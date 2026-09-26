"""工具定义和 ``@tool`` 语法糖。"""

import inspect
import re
import types
from collections.abc import Callable
from functools import update_wrapper
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

from .base import BaseTool, ToolDefinition, ToolPolicy


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
    if annotation is type(None):
        return {"type": "null"}
    if origin is Literal:
        values = list(arguments)
        schema = {"enum": values}
        if values:
            schema.update(_annotation_to_schema(type(values[0])))
        return schema
    if origin in (Union, types.UnionType):
        return {
            "anyOf": [
                _annotation_to_schema(argument)
                for argument in arguments
            ],
        }
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


def _schema_from_args_schema(args_schema: Any) -> dict[str, Any]:
    if isinstance(args_schema, dict):
        return dict(args_schema)
    model_json_schema = getattr(args_schema, "model_json_schema", None)
    if callable(model_json_schema):
        return model_json_schema()
    schema = getattr(args_schema, "schema", None)
    if callable(schema):
        return schema()
    raise TypeError("args_schema 必须是 dict 或提供 model_json_schema() 的模型")


class FunctionTool(BaseTool):
    """由异步函数生成的 ``BaseTool`` 实现。"""

    def __init__(
        self,
        function: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        policy: ToolPolicy | None = None,
        parameters: dict[str, Any] | None = None,
        args_schema: Any | None = None,
    ) -> None:
        if not inspect.iscoroutinefunction(function):
            raise TypeError("@tool 只接受 async def 工具函数")
        signature = inspect.signature(function)
        type_hints = get_type_hints(function, include_extras=True)
        docstring = inspect.getdoc(function) or ""
        resolved_description = description or docstring
        if not resolved_description:
            raise ValueError("工具函数必须提供 docstring 或 description")

        if args_schema is not None:
            resolved_parameters = _schema_from_args_schema(args_schema)
        elif parameters is not None:
            resolved_parameters = dict(parameters)
        else:
            properties: dict[str, dict[str, Any]] = {}
            required: list[str] = []
            argument_docs = _argument_descriptions(docstring)
            for parameter_name, parameter in signature.parameters.items():
                if parameter.kind in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                    inspect.Parameter.POSITIONAL_ONLY,
                }:
                    raise TypeError(
                        "工具函数不支持位置专用参数、*args 或 **kwargs"
                    )
                if parameter_name == "runtime":
                    continue
                if parameter_name not in type_hints:
                    raise TypeError(
                        f"工具参数 '{parameter_name}' 必须提供类型提示"
                    )
                schema = _annotation_to_schema(type_hints[parameter_name])
                if parameter_name in argument_docs:
                    schema.setdefault(
                        "description",
                        argument_docs[parameter_name],
                    )
                if parameter.default is inspect.Parameter.empty:
                    required.append(parameter_name)
                else:
                    schema["default"] = parameter.default
                properties[parameter_name] = schema
            resolved_parameters = {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }

        self.name = name or function.__name__
        self.description = resolved_description
        self.parameters = resolved_parameters
        self.policy = policy or ToolPolicy()
        self._function = function
        self.__tool_definition__ = ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
            policy=self.policy,
        )
        update_wrapper(self, function)

    async def execute(self, **kwargs: Any) -> Any:
        return await self._function(**kwargs)


def _decorate_class(
    target: type[BaseTool],
    *,
    name: str | None,
    description: str | None,
    policy: ToolPolicy | None,
) -> type[BaseTool]:
    if not inspect.isclass(target) or not issubclass(target, BaseTool):
        raise TypeError("@tool 只能修饰 BaseTool 子类或 async def 函数")

    declared = target.__dict__
    resolved_name = name or declared.get("name") or target.__name__
    resolved_description = (
        description
        or declared.get("description")
        or inspect.getdoc(target)
    )
    parameters = declared.get("parameters")
    resolved_policy = policy or declared.get("policy") or ToolPolicy()
    if not resolved_description:
        raise ValueError("工具必须提供 description 或类 docstring")
    if not isinstance(parameters, dict):
        raise ValueError("工具必须提供 parameters JSON Schema")

    target.name = str(resolved_name).strip()
    target.description = str(resolved_description)
    target.policy = resolved_policy
    target.__tool_definition__ = ToolDefinition(
        name=target.name,
        description=target.description,
        parameters=target.parameters,
        policy=target.policy,
    )
    return target


def tool(
    target: type[BaseTool] | Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    policy: ToolPolicy | None = None,
    parameters: dict[str, Any] | None = None,
    args_schema: Any | None = None,
):
    """将类或异步函数转换为完整的 ``BaseTool`` 定义。

    类工具必须实现 ``BaseTool`` 并声明 ``parameters``；函数工具通过签名、
    docstring、JSON Schema 或 Pydantic 模型生成参数定义。装饰器不会修改全局
    Manager，工具实例由 ``ToolManager`` 管理。
    """

    def decorator(value: type[BaseTool] | Callable[..., Any]):
        if inspect.isclass(value):
            return _decorate_class(
                value,
                name=name,
                description=description,
                policy=policy,
            )
        return FunctionTool(
            value,
            name=name,
            description=description,
            policy=policy,
            parameters=parameters,
            args_schema=args_schema,
        )

    if target is None:
        return decorator
    return decorator(target)
