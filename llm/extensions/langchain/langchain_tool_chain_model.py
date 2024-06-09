import os
from typing import Iterable, Literal, Type, TypeVar

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.utils import AddableDict
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from loguru import logger

from smart_base_model.llm.large_language_model_base import (
    LargeLanguageModelBase, MessageDict, StreamChunkMessageDict)
from smart_base_model.llm.llm_impls.ollama_large_language_model import \
    OllamaModelConfig
from smart_base_model.llm.llm_impls.openai_large_language_model import \
    OpenAIModelConfig

T = TypeVar("T")
StreamModeT = Literal["chunk", "full"]


class LangChainToolChainModel(LargeLanguageModelBase[MessageDict]):
    """
    **Only Available in OpenAI**
    A LargeLanguageModelBase implementation that uses the LangChain library to create a tool-calling agent.
    This model is not intended to be used with the SmartModel, as the base prompt is overridden by the LangChain library.
    The LangChainToolChainModel initializes with a LangChain chat model,
    a list of LangChain tools, a model configuration, and a system prompt.
    It then creates a tool-calling agent using the LangChain library and provides methods to asynchronously ask questions and chat with the agent, as well as a synchronous chat method.
    Example Usage With Tool:
        class CustomCalculatorTool(BaseTool):
            name: str = "Calculator"
            description: str = "useful for when you need to answer questions about math"
            args_schema: Type[BaseModel] = CalculatorInput
            return_direct: bool = False

            def _run(
                self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None
            ) -> int:
                ""Use the tool.""
                logger.info(f"[CALLING MULTIPLY] {a} x {b} = {a * b}")
                return a * b

        calculator = CustomCalculatorTool()
        tools: list[BaseTool] = [calculator]
        llm = ChatOpenAI()
        model = LangChainToolChainModel(llm, tools, config, "You are a math genius")
        print(model.ask("3 * 2 * 20 = "))

    """

    def __init__(
        self,
        langchain_chat_model: BaseChatModel,
        tools: Iterable[BaseTool],
        model_config: OpenAIModelConfig | OllamaModelConfig,
        system_prompt: str,
    ) -> None:
        self.system_prompt = system_prompt
        self.llm = langchain_chat_model
        self.tools = tools
        self.model_config = model_config

        self.__handle_vendor_model_init_requirement()

    def _yield_agent_output(self, chunk: AddableDict) -> Iterable[str]:
        # Agent Action
        if "actions" in chunk:
            for action in chunk["actions"]:
                yield f"[TOOL CALLING] Calling Tool: `{action.tool}` with input `{action.tool_input}`"
        # Observation
        elif "steps" in chunk:
            for step in chunk["steps"]:
                yield f"[TOOL CALLING] Tool Result: `{step.observation}`"

        # Final result
        elif "output" in chunk:
            yield f'[CHAINING FINISHED] Final Output: {chunk["output"]}'

    def async_ask(
        self, prompt: str, stream_mode: StreamModeT = "chunk"
    ) -> Iterable[StreamChunkMessageDict]:
        for chunk in self.async_chat(
            [{"role": "user", "content": prompt}], stream_mode
        ):
            yield chunk

    def async_chat(
        self, prompts: list[MessageDict], stream_mode: StreamModeT = "chunk"
    ) -> Iterable[StreamChunkMessageDict]:
        is_full: bool = stream_mode == "full"
        executor, chat_history = self.__create_base_agent_executor_with_chat_history(
            prompts
        )
        current_message = ""
        for chunk in executor.stream(
            {"input": prompts[-1]["content"], "chat_history": chat_history}
        ):
            for _log in self._yield_agent_output(chunk):
                if is_full:
                    current_message += f"\n{_log}"

                message_chunk: StreamChunkMessageDict = {
                    "content": current_message if is_full else _log,
                    "is_final_word": False,
                }
                yield message_chunk
            message_chunk["is_final_word"] = True
            yield message_chunk

    def __create_base_agent_executor_with_chat_history(
        self, prompts: list[MessageDict]
    ) -> tuple[AgentExecutor, list[BaseMessage]]:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.system_prompt,
                ),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        chat_history: list[BaseMessage] = []
        for _message_dict in prompts[:-1]:
            message_func: Type[BaseMessage] = (
                AIMessage if _message_dict["role"] == "assistant" else HumanMessage
            )
            message_body = message_func(content=_message_dict["content"])
            chat_history.append(message_body)

        agent = create_tool_calling_agent(self.llm, self.tools, prompt)  # type: ignore
        agent_executor = AgentExecutor(agent=agent, tools=self.tools, verbose=True)  # type: ignore
        return agent_executor, chat_history

    def ask(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    def __handle_openai(self) -> None:
        if isinstance(self.llm, ChatOpenAI) and "api_key" in self.model_config:
            logger.info(
                f"[SET OPENAI API KEY] SET KEY: {'*' * len(self.model_config['api_key'])}"
            )
            os.environ["OPENAI_API_KEY"] = self.model_config["api_key"]

    def __handle_vendor_model_init_requirement(self) -> None:
        handlers = [self.__handle_openai]
        for func in handlers:
            func()

    def chat(self, prompts: list[MessageDict]) -> str:
        executor, chat_history = self.__create_base_agent_executor_with_chat_history(
            prompts
        )
        return executor.invoke(
            {"input": prompts[-1]["content"], "chat_history": chat_history}
        )  # type: ignore
