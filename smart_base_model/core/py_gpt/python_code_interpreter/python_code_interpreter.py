import inspect
import json
import os
import re
import uuid
from pprint import pformat
from typing import Any

from loguru import logger
from openai import BaseModel
from pydantic import computed_field

import smart_base_model.utils.common_utils as common_utils
from smart_base_model.core.py_gpt.python_code_interpreter.command_executor import (
    CommandExecutor,
)
from smart_base_model.core.py_gpt.python_code_interpreter.python_source import (
    PythonSource,
)
from smart_base_model.core.py_gpt.python_code_interpreter.template import CODE_TEMPLATE

FilePath = str


class InterpreterResponse(BaseModel):
    """
    Represents the response from executing a Python code snippet using the Python interpreter.

    Attributes:
        response_id (str): A unique identifier for the response.
        source (PythonSource): The source of the Python code that was executed.
        code_executed (str): The Python code that was executed.
        stdout (Optional[str]): The standard output resulting from the code execution, if any.
        stderr (Optional[str]): The standard error resulting from the code execution, if any.
        session (dict[str, Any]): A dictionary containing any session data captured during the code execution.

    Properties:
        is_successful (bool): Indicates whether the code execution was successful. Returns True if stderr is None or empty, otherwise False.
    """

    response_id: str
    source: PythonSource
    code_executed: str
    stdout: str = ""
    stderr: str = ""
    session: dict[str, Any]

    @computed_field
    @property
    def is_successful(self) -> bool:
        """
        Determine if the code execution was successful.

        Returns:
            bool: True if there were no errors (stderr is None or empty), False otherwise.
        """
        return len(self.stderr) == 0


class PythonCodeInterpreter:
    """
    Provides an implementation of a Python code interpreter that can execute Python code and return the results.
    The `PythonCodeInterpreter` class is responsible for handling the execution of Python code, including injecting a decorator to capture local variables, creating and deleting temporary files, and parsing the output of the executed code.
    The `execute_python_source` method is the main entry point for executing Python code. It takes a `PythonSource` object, which represents the Python code to be executed, and returns an `InterpreterResponse` object containing the results of the execution, including the stdout, stderr, and any session data captured during the execution.
    """

    PY_COMMAND = "python3 -u {source_file}"

    def __init__(self, source: PythonSource, command_executor: CommandExecutor) -> None:
        self.source = source
        self.executor = command_executor

    def handle_init_python_source(self, source: PythonSource) -> str:
        init_source_code = inspect.getsource(common_utils.inject_locals_decorator)
        injected_code = common_utils.inject_decorator_for_source_code(
            source.code.split("\n"), common_utils.inject_locals_decorator.__name__
        )
        return CODE_TEMPLATE % (init_source_code, injected_code)

    def _create_temp_file(self, file_name: str, content: str) -> FilePath:
        file_path = f"/tmp/{file_name}"
        with open(file_path, "w") as temp_file:
            temp_file.write(content)
            logger.info(f"[FILE CREATION] Create temp file: {file_path}")
            return file_path

    def _delete_temp_file(self, file_path: str) -> None:
        if os.path.exists(file_path):
            logger.info(f"[FILE REMOVAL] Remove temp file: {file_path}")
            os.remove(file_path)

    def __parse_session_stdout(self, stdout: str) -> dict[str, Any]:
        pattern = r"<session>(.*?)</session>"

        # Search for the pattern
        match = re.search(pattern, stdout)
        # Check if the pattern was found and extract the content
        if match:
            content = match.group()
            return json.loads(
                content.replace("<session>", "").replace("</session>", "")
            )
        return {}

    def execute_python_source(self) -> InterpreterResponse:
        final_source_code = self.handle_init_python_source(self.source)
        logger.debug(final_source_code)
        file_name = str(uuid.uuid4())
        file_path = self._create_temp_file(file_name, final_source_code)

        cmd = self.PY_COMMAND.format(source_file=file_path)
        logger.info(f"[PYTHON EXECUTION] Execute python script: {cmd}")
        self.executor.execute(cmd, is_async_execution=False)

        self._delete_temp_file(file_path)
        stdout = "\n".join(self.executor.stdout_queue)
        stderr = "\n".join(self.executor.stderr_queue)
        session = self.__parse_session_stdout(stdout)
        logger.info(pformat(session))

        return InterpreterResponse(
            response_id=file_name,
            source=self.source,
            code_executed=final_source_code,
            stdout=stdout,
            stderr=stderr,
            session=session,
        )
