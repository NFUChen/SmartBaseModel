## Package Overview

The `smart_base_model` package provides a base model for interacting with large language models (LLMs). It facilitates the generation of responses based on user-defined prompts and includes robust error handling mechanisms for better reliability.

## Features

- **Recursive Response Generation**: Attempts to fetch valid responses through multiple retries in case of errors.
- **Extensible Design**: Allows subclasses to define specific data models, making it easy to adapt to various use cases.
- **Integration with LLMs**: Works seamlessly with different large language model implementations.

## Installation

You can install the package using pip:

```
pip3 install git+https://github.com/NFUChen/smart_base_model.git
```

## Usage

### Defining a Model

You can define your own data model by subclassing `SmartBaseModel`. For example, to create a `Person` model:

```python
from smart_base_model import SmartBaseModel

class Person(SmartBaseModel["Person"]):
    name: str
    age: int
    address: str
```

### Creating an LLM Instance

Create an instance of a large language model, such as `OpenAIModel`:

```python
from smart_base_model import OpenAIModel

model = OpenAIModel(
    {
        "api_key": "your-api-key-here",
        "model_name": "gpt-3.5-turbo",
        "mode": "json",
    }
)
```

### Generating a Model Instance

Use the model to generate an instance of your defined class:

```python
person = Person.model_ask(
    "My name is William Chen, I am 28 years old, and I live in New York.",
    llm=model
)
```

This will create a `Person` object populated with the data extracted from the provided prompt.
