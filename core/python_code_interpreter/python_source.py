
from smart_base_model.core.smart_base_model.smart_base_model import SmartBaseModel


class PythonSource(SmartBaseModel["PythonSource"]):
    """
    Represents a data class that contains valid Python code.
    The `intent` attribute represents the intended purpose or meaning of the Python code.
    The `code` attribute is annotated as a string containing valid Python code that can be executed inside a sandbox environment
    **IMPORTANT**
        Please also consider python indentation of the function you wrote
    """

    code: str
    intent: str
