from abc import ABC, abstractmethod
import inspect
from typing import (
    Generic,
    Iterable,
    Optional,
    Type,
    TypeVar,
    Union,
    get_origin,
    get_args,
)
from loguru import logger
from pydantic import BaseModel

from smart_base_model.llm.large_language_model_base import (
    LargeLanguageModelBase,
    MessageDict,
)
from smart_base_model.prompts.model_prompt import BASE_PROMPT

T = TypeVar("T")


class SmartBaseModel(BaseModel, Generic[T]):
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def _search_base_model_dependencies(
        cls, source_cls: Type["SmartBaseModel"]
    ) -> set[Type[BaseModel]]:
        """
        Recursively searches for all BaseModel dependencies of the given source class.

        This method traverses the fields of the source class and its base classes to find all
        BaseModel types that are used as field annotations, either directly or as part of a
        Union or Iterable type. The resulting set of BaseModel types represents the complete
        set of dependencies for the source class.

        Args:
            source_cls (Type[SmartBaseModel]): The source class to search for dependencies.

        Returns:
            set[Type[BaseModel]]: The set of all BaseModel dependencies for the source class.
        """

        base_model_dependencies: set[Type[BaseModel]] = set([source_cls])
        try:
            for attr, field_info in source_cls.model_fields.items():
                if field_info.annotation is None:
                    continue
                is_union_type = get_origin(field_info.annotation) is Union
                if is_union_type:
                    arg_types = get_args(field_info.annotation)
                    for _type in arg_types:
                        if issubclass(_type.__class__, (BaseModel)):
                            base_model_dependencies.add(_type)
                    continue
                dependencies = field_info.annotation.__mro__
                if BaseModel in dependencies:
                    base_model_dependencies.add(dependencies[0])
                py_container = get_origin(field_info.annotation)
                if py_container is not None and issubclass(
                    py_container, (list, set, Iterable)
                ):
                    arg_types = get_args(field_info.annotation)
                    for type in arg_types:
                        type_dependencies = type.__mro__
                        if BaseModel in type_dependencies:
                            base_model_dependencies.add(type)
        except (TypeError, AttributeError) as error:
            logger.error(f"Error in {field_info}")
            logger.exception(error)
            raise error

        return base_model_dependencies

    @classmethod
    def _get_model_with_source_code(cls) -> tuple[Type[BaseModel], str]:
        model_cls = cls.__mro__[0]
        model_classes = cls._search_base_model_dependencies(model_cls)
        all_source_code: set[str] = set()
        for _cls in model_classes:
            source_code = inspect.getsource(_cls)
            all_source_code.add(source_code)
        return (model_cls, "\n".join(all_source_code))

    @classmethod
    def model_ask_json(cls, prompt: str, llm: LargeLanguageModelBase) -> str:
        _self_model_cls, self_source_code = cls._get_model_with_source_code()

        system_prompt = BASE_PROMPT % (self_source_code, prompt)
        messages: list[MessageDict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        logger.info(f"[CLASS MODELLING] Taget model: \n{self_source_code}")
        response = llm.chat(messages)

        logger.info(f"[MODEL RESPONSE] Response json: \n{response}")

        return response

    @classmethod
    def model_ask(cls, prompt: str, llm: LargeLanguageModelBase) -> Optional[T]:
        try:
            _self_model_cls, self_source_code = cls._get_model_with_source_code()
            json_response = cls.model_ask_json(prompt, llm)
            return cls.model_validate_json(json_response)  # type: ignore
        except Exception as error:
            logger.info(f"\n{self_source_code}")
            logger.info(json_response)
            logger.exception(error)
