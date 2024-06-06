from loguru import logger

from smart_base_model.core.smart_base_model import SmartBaseModel
from smart_base_model.llm.llm_impls.openai_large_language_model import (
    MessageDict,
    OpenaiModel,
)
from smart_base_model.prompts.model_prompt import BASE_PROMPT
from typing import Generic, Optional, TypeVar

T = TypeVar("T")

class OpenaiSmartBaseModel(SmartBaseModel, Generic[T]):
    """
    Provides an implementation of the `SmartBaseModel` interface using the OpenAI language model.
    The `OpenaiSmartBaseModel` class is a generic implementation of the `SmartBaseModel` interface that uses the OpenAI language model to generate responses to prompts. 
    It includes methods for generating JSON responses and validating the responses against a generic type `T`.
    The `model_ask_json` method takes a prompt and an `OpenaiModel` instance, and returns the JSON response from the OpenAI model. 
    The `model_ask` method takes a prompt and an `OpenaiModel` instance, and returns the validated response of type `T`.
    Both methods handle exceptions and log relevant information, such as the source code of the model and the JSON response, in case of errors.
    """
        
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def model_ask_json(cls, prompt: str, model: OpenaiModel) -> str:
        _self_model_cls, self_source_code = cls._get_model_with_source_code()

        system_prompt = BASE_PROMPT % (self_source_code, prompt)
        messages: list[MessageDict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        logger.info(f"[CLASS MODELLING]Taget model: \n{self_source_code}")
        response = model.chat(messages)

        logger.info(f"[OPENAI RESPONSE] Response json: \n{response}")

        return response


    @classmethod
    def model_ask(cls, prompt: str, model: OpenaiModel) -> Optional[T]:
        try:
            _self_model_cls, self_source_code = cls._get_model_with_source_code()
            json_response = cls.model_ask_json(prompt, model)
            return cls.model_validate_json(json_response)  # type: ignore
        except Exception as error:
            logger.info(f"\n{self_source_code}")
            logger.info(json_response)
            logger.exception(error)
