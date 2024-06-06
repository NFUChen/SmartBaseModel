from pprint import pformat
from typing import Callable, Iterable, Literal, TypedDict, cast

from loguru import logger
from smart_base_model.llm.large_language_model_base import LargeLanguageModelBase, ModelType

from openai import OpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
)


class OpenaiModelConfig(TypedDict):
    api_key: str
    model_name: str
    mode: Literal["text"] | Literal["json"]


class MessageDict(TypedDict):
    role: Literal["user"] | Literal["assistant"] | Literal["system"]
    content: str


class OpenaiModel(LargeLanguageModelBase[MessageDict]):
    """
    Implements an OpenAI-based large language model (LLM) for use in the SmartBaseModel framework.
    
    The `OpenaiModel` class provides an interface to interact with the OpenAI API to generate text completions. 
    It supports both streaming and non-streaming modes, and allows for the use of a system prompt to provide context for the model.
    
    The class has the following methods:
    
    - `__init__`: Initializes the OpenAI model with the provided configuration and system prompt.
    - `_create_chat_func`: Creates a callable function that can be used to generate text completions from the OpenAI API.
    - `async_chat`: Generates a text completion in streaming mode and emits the partial responses.
    - `async_ask`: Generates a text completion for a single prompt in streaming mode.
    - `chat`: Generates a text completion in non-streaming mode and returns the full response.
    - `ask`: Generates a text completion for a single prompt in non-streaming mode.
    """
        
    MODEL_TYPE = ModelType.OPENAI

    def __init__(
        self, model_config: OpenaiModelConfig, system_prompt: str = ""
    ) -> None:
        self.system_prompt_dict: MessageDict = {
            "role": "system",
            "content": system_prompt,
        }
        self.api_key = model_config["api_key"]
        self.model_name = model_config["model_name"]
        self.mode = model_config["mode"]

    def _create_chat_func(
        self,
        model_name: str,
        messages: list[MessageDict],
        is_stream: bool,
    ) -> Callable[[], ChatCompletion | Iterable[ChatCompletionChunk]]:
        client = OpenAI(api_key=self.api_key)

        _format = {
            "json": {"type": "json_object"},
            "text": {"type": "text"},
        }

        return lambda: client.chat.completions.create(
            response_format=_format[self.mode], # type: ignore
            messages=messages,  # type: ignore
            model=model_name,
            stream=is_stream,
        )

    def async_chat(self, prompts: list[MessageDict]) -> None:
        messages: list[MessageDict] = [self.system_prompt_dict, *prompts]
        stream_func = self._create_chat_func(self.model_name, messages, True)
        current_message = ""
        for chunk in stream_func():  # type: ignore
            chunk: ChatCompletionChunk
            message = chunk.choices[0].delta.content
            if message is None:
                continue
            current_message += message
            self.message_subject.next(current_message)

    def async_ask(self, prompt: str) -> None:
        self.async_chat([{"role": "user", "content": prompt}])

    def chat(self, prompts: list[MessageDict]) -> str:
        messages: list[MessageDict] = [self.system_prompt_dict, *prompts]
        response_body = self._create_chat_func(self.model_name, messages, False)()
        logger.debug(f"\n{pformat(response_body.model_dump())}") #type: ignore
        human_readable_response = cast(
            str,
            response_body
            .choices[0]  # type: ignore
            .message.content,
        )  # type: ignore
        
        return human_readable_response

    def ask(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])
