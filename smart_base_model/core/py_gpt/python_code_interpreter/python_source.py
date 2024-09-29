from smart_base_model.core.smart_base_model.smart_base_model import SmartBaseModel


class PythonSource(SmartBaseModel["PythonSource"]):
    """
    Represents a data class that encapsulates valid Python code.

    Attributes:
    - `intent`: A string representing the intended purpose or functionality of the Python code.
    - `code`: A string containing valid Python code that can be executed in a sandbox environment.

    **IMPORTANT**
    - Ensure proper Python indentation for the code you write.

    """

    code: str
    intent: str
