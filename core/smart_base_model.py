import inspect
from typing import (
    Generic,
    Optional,
    Type,
    TypeVar,
)
from loguru import logger
from pydantic import BaseModel

from smart_base_model.llm.large_language_model_base import (
    LargeLanguageModelBase,
    MessageDict,
)
from smart_base_model.prompts.model_prompt import BASE_PROMPT
from smart_base_model.utils import common_utils


T = TypeVar("T")


class SmartBaseModel(BaseModel, Generic[T]):
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def _get_model_with_source_code(cls) -> tuple[Type[BaseModel], str]:
        model_cls = cls.__mro__[0]
        model_classes = common_utils.recursively_search_base_model_dependencies(
            model_cls
        )
        all_source_code: set[str] = set()
        for _cls in model_classes:
            source_code = inspect.getsource(_cls)
            all_source_code.add(source_code)
        return (model_cls, "\n".join(all_source_code))

    @classmethod
    def model_ask_json(cls, prompt: str, llm: LargeLanguageModelBase) -> Optional[str]:
        try:
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
        except Exception as error:
            logger.exception(error)
            return

    @classmethod
    def model_ask(cls, prompt: str, llm: LargeLanguageModelBase) -> Optional[T]:
        try:
            _self_model_cls, self_source_code = cls._get_model_with_source_code()
            json_response = cls.model_ask_json(prompt, llm)
            if json_response is None:
                return
            return cls.model_validate_json(json_response)  # type: ignore
        except Exception as error:
            logger.info(f"\n{self_source_code}")
            logger.info(json_response)
            logger.exception(error)
