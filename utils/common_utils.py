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
