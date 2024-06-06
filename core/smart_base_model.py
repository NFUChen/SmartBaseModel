from abc import ABC, abstractmethod
import inspect
from typing import Type
from pydantic import BaseModel

from smart_base_model.llm.large_language_model_base import LargeLanguageModelBase


class SmartBaseModel(ABC, BaseModel):
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def _get_model_with_source_code(cls) -> tuple[Type[BaseModel], str]:
        model_cls = cls.__mro__[0]
        self_source_code = inspect.getsource(model_cls)
        return (model_cls, self_source_code)

    @classmethod
    @abstractmethod
    def model_ask(cls, prompt: str, llm: LargeLanguageModelBase) -> BaseModel: ...

    @classmethod
    @abstractmethod
    def model_ask_json(cls, prompt: str, llm: LargeLanguageModelBase) -> str: ...
