import re
from typing import Any, Iterable, Optional, Type, Union, get_args

from pydantic import BaseModel
from typing_extensions import get_origin


def recursively_search_base_model_dependencies(
    source_cls: Type[BaseModel],
) -> set[Type[BaseModel]]:
    def _optional_mro(_cls: Type[Any]) -> Optional[tuple[type, ...]]:
        try:
            return _cls.__mro__
        except:  # noqa: E722
            ...

    deps = set()

    def dfs(source_cls: Type[BaseModel]):
        _source_mro = _optional_mro(source_cls)
        if _source_mro is None:
            return

        if BaseModel in _source_mro:
            deps.add(source_cls)
        else:
            return

        for _, field_info in source_cls.model_fields.items():
            is_union_type = get_origin(field_info.annotation) is Union
            if is_union_type:
                type_args = get_args(field_info.annotation)
                for _type in type_args:
                    dfs(_type)
                continue
            is_iterable = get_origin(field_info.annotation) in (
                Iterable,
                list,
                tuple,
                set,
            )
            if is_iterable:
                type_args = get_args(field_info.annotation)
                for _type in type_args:
                    dfs(_type)
                continue

            if field_info.annotation is not None:
                dfs(field_info.annotation)

    dfs(source_cls)
    return deps


def inject_locals_decorator(func):
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if "session" not in globals():
            globals()["session"] = {}
        globals()["session"][func.__name__] = locals()
        return result

    return wrapper


def inject_decorator_for_source_code(lines: list[str], decorator_name: str) -> str:
    # Prepare the decorator line
    decorator_line = f"@{decorator_name}\n"

    # Create a new list for the modified source code
    modified_lines = []

    # Find all function definitions and inject the decorator above each one
    func_pattern = re.compile(
        r"^\s*def\s+\w+\s*\("
    )  # Pattern to match any function definition
    for line in lines:
        if func_pattern.match(line):
            # Insert the decorator line above the function definition
            modified_lines.append(decorator_line)
        modified_lines.append(line)
    return "\n".join(modified_lines)
