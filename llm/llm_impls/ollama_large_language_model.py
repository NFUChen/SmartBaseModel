from typing import Any, Literal, TypedDict
from smart_base_model.llm.large_language_model_base import LargeLanguageModelBase
import ollama


class OllamaModelConfig(TypedDict):
    host: str
    port: int
    model_name: Literal[
        "llama3",
        "llama3:70b",
        "phi3",
        "phi3:medium",
        "gemma:2b",
        "gemma:7b",
        "mistral",
        "moondream",
        "neural-chat",
        "starling-lm",
        "codellama",
        "llama2-uncensored",
        "llava",
        "solar",
    ]


class OllamaModel(LargeLanguageModelBase[ollama.Message]):
    def __init__(
        self, model_config: OllamaModelConfig, system_prompt: str = ""
    ) -> None:
        self.system_prompt_dict: ollama.Message = {
            "role": "system",
            "content": system_prompt,
        }
        self.model_config = model_config
        url = f'http://{model_config["host"]}:{model_config["port"]}'

        self.model_name = model_config["model_name"]

        self.client = ollama.Client(url)

    def ask(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    def chat(self, prompts: list[ollama.Message]) -> Any:
        messages: list[ollama.Message] = [self.system_prompt_dict, *prompts]
        return self.client.chat(model=self.model_name, messages=messages)
