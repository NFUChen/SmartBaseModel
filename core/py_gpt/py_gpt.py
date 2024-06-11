import inspect
from typing import Iterable, Type
from smart_base_model.core.py_gpt.python_code_interpreter.command_executor import (
    CommandExecutor,
)
from smart_base_model.core.py_gpt.python_code_interpreter.python_code_interpreter import (
    PythonCodeInterpreter,
)
from smart_base_model.core.py_gpt.python_code_interpreter.python_source import (
    PythonSource,
)
from smart_base_model.core.smart_base_model.smart_base_model import (
    MessageSubjectResponse,
)
from smart_base_model.llm.large_language_model_base import (
    LargeLanguageModelBase,
    MessageDict,
    StreamChunkMessageDict,
)
from smart_base_model.messaging.behavior_subject import BehaviorSubject
from smart_base_model.core.py_gpt.prompts.py_gpt_prompts import PY_GPT_SYSTEM_PROMPT


class PythonInterpreterError(BaseException):
    def __init__(self, source_code: str, stderr: str) -> None:
        self.source_code = source_code
        self.stderr = stderr

    def __str__(self) -> str:
        return f"[INTERPRETER FAILED] Fail to execute following source code: \n{self.source_code} \n STDERR: \n{self.stderr}"


class PyGPT(LargeLanguageModelBase[MessageDict]):
    def __init__(
        self,
        smart_model_llm: LargeLanguageModelBase[MessageDict],
        gpt_llm: LargeLanguageModelBase[MessageDict],
        interpreter_cls: Type[PythonCodeInterpreter],
        py_source_cls: Type[PythonSource],
        command_executor: CommandExecutor,
    ) -> None:
        self.smart_model_llm = smart_model_llm
        self.py_source_cls = py_source_cls
        self.interpreter_cls = interpreter_cls
        self.system_unformatted_prompt = PY_GPT_SYSTEM_PROMPT
        self.gpt_llm = gpt_llm
        self.executor = command_executor

    def get_message_subject(self) -> BehaviorSubject[MessageSubjectResponse]:
        return self.py_source_cls.message_subject

    def async_ask(self, prompt: str) -> Iterable[StreamChunkMessageDict]:
        return self.async_chat([{"role": "user", "content": prompt}])

    def async_chat(
        self, prompts: list[MessageDict]
    ) -> Iterable[StreamChunkMessageDict]:
        optional_py_source = self.py_source_cls.model_ask(
            prompts[-1]["content"], self.smart_model_llm
        )
        if optional_py_source is None:
            raise TypeError(
                "[UNABLE TO MODEL PY SOURCE] Unable to model python source..."
            )
        code_interpreter = self.interpreter_cls(optional_py_source, self.executor)
        response = code_interpreter.execute_python_source()

        if not response.is_successful:
            raise PythonInterpreterError(optional_py_source.code, response.stderr or "")

        response_source_code = inspect.getsource(response.__class__)
        self.gpt_llm.set_system_prompt(
            self.system_unformatted_prompt
            % (response_source_code, response.model_dump_json())
        )

        return self.gpt_llm.async_chat(prompts)

    def ask(self, prompt: str) -> str: ...

    def chat(self, prompts: list[MessageDict]) -> str: ...
