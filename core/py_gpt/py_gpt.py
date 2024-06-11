from typing import Iterable, Type, TypeVar
from smart_base_model.core.python_code_interpreter.command_executor import CommandExecutor
from smart_base_model.core.python_code_interpreter.python_code_interpreter import (
    PythonCodeInterpreter,
)
from smart_base_model.core.python_code_interpreter.python_source import PythonSource
from smart_base_model.llm.extensions.langchain.langchain_tool_chain_model import LangChainToolChainModel
from smart_base_model.llm.large_language_model_base import (
    LargeLanguageModelBase,
    MessageDict,
    StreamChunkMessageDict,
)
from smart_base_model.prompts.py_gpt_prompts import PY_GPT_SYSTEM_PROMPT


class PyGPT(LargeLanguageModelBase[MessageDict]):
    def __init__(
        self,
        smart_model_llm: LargeLanguageModelBase[MessageDict],
        gpt_llm: LargeLanguageModelBase[MessageDict],
        interpreter_cls: Type[PythonCodeInterpreter],
        py_source_cls: Type[PythonSource],
        command_executor: CommandExecutor
    ) -> None:
        self.smart_model_llm = smart_model_llm
        self.py_source_cls = py_source_cls
        self.interpreter_cls = interpreter_cls
        system_prompt = PY_GPT_SYSTEM_PROMPT
        self.gpt_llm = gpt_llm
        self.gpt_llm.set_system_prompt(system_prompt)
        self.executor = command_executor
        
    def async_ask(self, prompt: str) -> Iterable[StreamChunkMessageDict]:
        ...

    def async_chat(self, prompts: list[MessageDict]) -> Iterable[StreamChunkMessageDict]:
        optional_py_source  = self.py_source_cls.model_ask(prompts[-1]['content'], self.smart_model_llm)
        if optional_py_source is None:
            raise TypeError("[UNABLE TO MODEL PY SOURCE] Unable to model python source...")
        code_interpreter = self.interpreter_cls(optional_py_source, self.executor)
        response = code_interpreter.execute_python_source()
        self.gpt_llm.async_chat([*prompts, {"role": "user", "content": ""}])

    def ask(self, prompt: str) -> str: ...

    def chat(self, prompts: list[MessageDict]) -> str: ...
