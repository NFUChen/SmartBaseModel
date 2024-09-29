import inspect
from enum import Enum
from typing import ClassVar, Generic, Optional, Type, TypeVar
from uuid import UUID, uuid4

from loguru import logger
from pydantic import BaseModel
from typing_extensions import TypedDict

from smart_base_model.core.smart_base_model.prompts.model_prompts import (
    BASE_PROMPT,
    ERROR_CORRECTION_PROMPT,
)
from smart_base_model.llm.large_language_model_base import (
    LargeLanguageModelBase,
    MessageDict,
    StreamChunkMessageDict,
)
from smart_base_model.messaging.behavior_subject import BehaviorSubject
from smart_base_model.utils import common_utils

T = TypeVar("T")


class ScratchPad(BaseModel):
    prompt: str
    schema_reference: str
    current_response: str
    error: str

    def as_text(self) -> str:
        return f"""
            Original Prompt: {self.prompt}
            Schema: 
                {self.schema_reference}
            Response: 
                {self.current_response}
            Error:
                {self.error}
        """


class MessageSubjectResponse(TypedDict):
    id: str
    chunk_message: StreamChunkMessageDict


class SmartBaseModel(BaseModel, Generic[T]):
    """
    A base model that interacts with large language models (LLMs) to generate responses based on a given prompt. 
    This class handles exceptions and supports recursive attempts to retrieve valid outputs.

    Args:
        prompt (str): The prompt to be sent to the LLM.
        llm (LargeLanguageModelBase): The instance of the LLM to use for generating the response.

    Returns:
        Optional[T]: The validated response from the LLM or None if an exception occurs.

    This abstract base class provides a common interface for different types of LLMs. 
    Subclasses must implement the abstract methods: `async_ask`, `async_chat`, `ask`, and `chat`.

    Example:
        Define a SmartBaseModel subclass for a `Person`:

        ```python
        class Person(SmartBaseModel["Person"]):
            name: str
            age: int
            address: str
        ```

        Create an instance of an LLM (e.g., OpenAIModel):

        ```python
        model = OpenAIModel(
            {
                "api_key": "your-api-key-here",
                "model_name": "gpt-3.5-turbo",
                "mode": "json",
            }
        )
        ```

        Use the model to create a `Person` instance:

        ```python
        person = Person.model_ask(
            "My name is William Chen, I am 28 years old, and I live in New York.",
            llm=model
        )
        ```

        This will create a `Person` object using information extracted from the prompt, leveraging the LLM for natural language understanding.

    Attributes:
        _MAX_ATTEMPT (ClassVar[int]): Maximum number of attempts to fetch a valid response.
        message_subject (ClassVar[BehaviorSubject[MessageSubjectResponse]]): Subject to handle message responses during the LLM query process.

    Methods:
        _get_model_with_source_code(cls) -> tuple[Type[BaseModel], str]:
            Retrieves the source code of the model and its dependencies.
        
        model_ask_json(cls, prompt: str, llm: LargeLanguageModelBase[MessageDict], response_id: UUID = uuid4()) -> Optional[str]:
            Sends a prompt to the LLM and retrieves the response in JSON format.
        
        model_ask(cls, prompt: str, llm: LargeLanguageModelBase, response_id: UUID = uuid4()) -> Optional[T]:
            Recursively attempts to generate a response for the given prompt, handling any exceptions that may occur.
    """

    
    
    _MAX_ATTEMPT: ClassVar[int] = 5
    message_subject: ClassVar[BehaviorSubject[MessageSubjectResponse]] = (
        BehaviorSubject[MessageSubjectResponse]()
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def _get_model_with_source_code(cls) -> tuple[Type[BaseModel], str]:
        model_cls = cls.__mro__[0]
        model_classes = common_utils.recursively_search_base_model_dependencies(
            source_cls=model_cls, include_classes=[Enum]
        )
        all_source_code: set[str] = set()
        for _cls in model_classes:
            source_code = inspect.getsource(_cls)
            all_source_code.add(source_code)
        return (model_cls, "\n".join(all_source_code))

    @classmethod
    def model_ask_json(
        cls,
        prompt: str,
        llm: LargeLanguageModelBase[MessageDict],
        response_id: UUID = uuid4(),
    ) -> Optional[str]:
        try:
            _self_model_cls, self_source_code = cls._get_model_with_source_code()

            system_prompt = BASE_PROMPT % (self_source_code, prompt)
            messages: list[MessageDict] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            logger.info(f"[CLASS MODELLING] Target model: \n{self_source_code}")
            for chunk in llm.async_chat(messages):
                cls.message_subject.next(
                    {"id": str(response_id), "chunk_message": chunk}
                )
            cls.message_subject.next({"id": str(response_id), "chunk_message": chunk})
            logger.info(f"[MODEL RESPONSE] Response json: \n{chunk['content']}")
            return chunk["content"]
        except Exception as error:
            logger.exception(error)
            return

    @classmethod
    def model_ask(
        cls, prompt: str, llm: LargeLanguageModelBase, response_id: UUID = uuid4()
    ) -> Optional[T]:
        """
        Recursively attempts to generate a response from the large language model (LLM) for the given prompt, handling any exceptions that may occur.

        Args:
            prompt (str): The prompt to be sent to the LLM.
            llm (LargeLanguageModelBase): The large language model instance to use for generating the response.

        Returns:
            Optional[T]: The validated response from the LLM, or None if an exception occurs.
        """
        _self_model_cls, self_source_code = cls._get_model_with_source_code()
        scratch_pad: ScratchPad = ScratchPad(
            prompt=prompt,
            error="",
            current_response="",
            schema_reference=self_source_code,
        )
        current_attempt = 0

        def model_ask_wrapper(scratch_pad: ScratchPad) -> Optional[T]:
            nonlocal current_attempt
            try:
                json_response = cls.model_ask_json(
                    scratch_pad.as_text(), llm, response_id
                )
                if json_response is None:
                    return
                return cls.model_validate_json(json_response)  # type: ignore
            except Exception as error:
                logger.info(f"\n{self_source_code}")
                logger.info(json_response)
                logger.exception(error)

                scratch_pad.error = ERROR_CORRECTION_PROMPT.format(error=error)
                if json_response is not None:
                    scratch_pad.current_response = json_response
                logger.warning(
                    f"[ERROR ATTEMPT] Attempt[{current_attempt}]: Current scratch pad with error:\n {scratch_pad.as_text()}"
                )
                current_attempt += 1
                if current_attempt > cls._MAX_ATTEMPT:
                    logger.critical(
                        "[EXCEED MAX ATTEMPT] Exit model_ask loop for preventing recursively query on model..."
                    )
                    return
                return model_ask_wrapper(scratch_pad)

        return model_ask_wrapper(scratch_pad)
