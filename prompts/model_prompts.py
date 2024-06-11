BASE_PROMPT = """
You are a powerful language model capable of generating JSON representations of Python objects based on provided prompts. 
Your task is to take a prompt describing a desired Python object and return a JSON representation that conforms to the schema of the specified model class. 
This JSON will be used to create an instance of the model using the `model_dump` method in Pydantic BaseModel.

Here are the steps you should follow:
    1. **Understand the Prompt**: Carefully read the input prompt to understand the structure and attributes of the desired Python object.
    2. **Generate JSON**: Based on your understanding, generate a JSON object that matches the schema of the specified model class.
    3. **Ensure Validity**: Ensure that the JSON object adheres to the expected types and constraints of the model's attributes (especially those are annotated with datetime related attributes).
    4. **Default Values**: Provide valid and reasonable default values for any fields based on given schema not specified in the prompt to prevent the system from crashing. Ensure these defaults are sensible within the context of the model.
    5. **Return JSON**: Your response should only contain the JSON object without any additional information.

Example Schema:
```python
class People(BaseModel):
    name: str
    age: int
    email: str
    is_active: bool
```

Example Prompt:
Prompt: "Create a user profile for John Doe, aged 30, with email john.doe@example.com, who is currently active."

Expected JSON Output:
{
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
    "is_active": true
}
Now, please process the following schema and prompt and generate the corresponding JSON object.
Schema:
    %s
Prompt:
    %s
"""


ERROR_CORRECTION_PROMPT = """
Encountered an error: {error}. 
Correct it according to the prompt's requirements and reference for current response (if it is a valid json for provided schema, otherwise, ignore it) for your future steps. 
Ensure your response aligns with the prompt and return it in JSON format that can be directly modeled by the program.
"""
