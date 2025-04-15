import json
import re
from typing import Any, Callable, Dict, List, Optional, Union

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.agents.agent_builder.base_agent_executor_mixin import CrewAgentExecutorMixin
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    OutputParserException,
)
from crewai.agents.tools_handler import ToolsHandler
from crewai.llm import BaseLLM
from crewai.tools.base_tool import BaseTool
from crewai.tools.structured_tool import CrewStructuredTool
from crewai.tools.tool_types import ToolResult
from crewai.utilities import I18N, Printer
from crewai.utilities.agent_utils import (
    enforce_rpm_limit,
    format_message_for_llm,
    get_llm_response,
    handle_agent_action_core,
    handle_context_length,
    handle_max_iterations_exceeded,
    handle_output_parser_exception,
    handle_unknown_error,
    has_reached_max_iterations,
    is_context_length_exceeded,
    process_llm_response,
    show_agent_logs,
)
from crewai.utilities.constants import MAX_LLM_RETRY, TRAINING_DATA_FILE
from crewai.utilities.logger import Logger
from crewai.utilities.tool_utils import execute_tool_and_check_finality
from crewai.utilities.training_handler import CrewTrainingHandler

from crewai.agents.crew_agent_executor import CrewAgentExecutor

from .agent_utils import show_agent_logs_json

class Custom_CrewAgentExecutor(CrewAgentExecutor):
    _logger: Logger = Logger()

    def __init__(
        self,
        llm: Any,
        task: Any,
        crew: Any,
        agent: BaseAgent,
        prompt: dict[str, str],
        max_iter: int,
        tools: List[CrewStructuredTool],
        tools_names: str,
        stop_words: List[str],
        tools_description: str,
        tools_handler: ToolsHandler,
        step_callback: Any = None,
        original_tools: List[Any] = [],
        function_calling_llm: Any = None,
        respect_context_window: bool = False,
        request_within_rpm_limit: Optional[Callable[[], bool]] = None,
        callbacks: List[Any] = [],
        **kwargs,
    ):
        self._i18n: I18N = I18N()
        self.llm: BaseLLM = llm
        self.task = task
        self.agent = agent
        self.crew = crew
        self.prompt = prompt
        self.tools = tools
        self.tools_names = tools_names
        self.stop = stop_words
        self.max_iter = max_iter
        self.callbacks = callbacks
        self._printer: Printer = Printer()
        self.tools_handler = tools_handler
        self.original_tools = original_tools
        self.step_callback = step_callback
        self.use_stop_words = self.llm.supports_stop_words()
        self.tools_description = tools_description
        self.function_calling_llm = function_calling_llm
        self.respect_context_window = respect_context_window
        self.request_within_rpm_limit = request_within_rpm_limit
        self.ask_for_human_input = False
        self.messages: List[Dict[str, str]] = []
        self.iterations = 0
        self.log_error_after = 3
        self.tool_name_to_tool_map: Dict[str, Union[CrewStructuredTool, BaseTool]] = {
            tool.name: tool for tool in self.tools
        }
        existing_stop = self.llm.stop or []
        self.llm.stop = list(
            set(
                existing_stop + self.stop
                if isinstance(existing_stop, list)
                else self.stop
            )
        )

    def _invoke_loop(self) -> AgentFinish:
        """
        Main loop to invoke the agent's thought process until it reaches a conclusion
        or the maximum number of iterations is reached.
        """
        formatted_answer = None
        while not isinstance(formatted_answer, AgentFinish):
            try:
                if has_reached_max_iterations(self.iterations, self.max_iter):
                    formatted_answer = handle_max_iterations_exceeded(
                        formatted_answer,
                        printer=self._printer,
                        i18n=self._i18n,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                    )

                enforce_rpm_limit(self.request_within_rpm_limit)

                answer = get_llm_response(
                    llm=self.llm,
                    messages=self.messages,
                    callbacks=self.callbacks,
                    printer=self._printer,
                )
                formatted_answer = process_llm_response(answer, self.use_stop_words)

                if isinstance(formatted_answer, AgentAction):
                    # Extract agent fingerprint if available
                    fingerprint_context = {}
                    if (
                        self.agent
                        and hasattr(self.agent, "security_config")
                        and hasattr(self.agent.security_config, "fingerprint")
                    ):
                        fingerprint_context = {
                            "agent_fingerprint": str(
                                self.agent.security_config.fingerprint
                            )
                        }

                    tool_result = execute_tool_and_check_finality(
                        agent_action=formatted_answer,
                        fingerprint_context=fingerprint_context,
                        tools=self.tools,
                        i18n=self._i18n,
                        agent_key=self.agent.key if self.agent else None,
                        agent_role=self.agent.role if self.agent else None,
                        tools_handler=self.tools_handler,
                        task=self.task,
                        agent=self.agent,
                        function_calling_llm=self.function_calling_llm,
                    )
                    formatted_answer = self._handle_agent_action(
                        formatted_answer, tool_result
                    )

                self._invoke_step_callback(formatted_answer)
                self._append_message(formatted_answer.text, role="assistant")

            except OutputParserException as e:
                formatted_answer = handle_output_parser_exception(
                    e=e,
                    messages=self.messages,
                    iterations=self.iterations,
                    log_error_after=self.log_error_after,
                    printer=self._printer,
                )

            except Exception as e:
                if e.__class__.__module__.startswith("litellm"):
                    # Do not retry on litellm errors
                    raise e
                if is_context_length_exceeded(e):
                    handle_context_length(
                        respect_context_window=self.respect_context_window,
                        printer=self._printer,
                        messages=self.messages,
                        llm=self.llm,
                        callbacks=self.callbacks,
                        i18n=self._i18n,
                    )
                    continue
                else:
                    handle_unknown_error(self._printer, e)
                    raise e
            finally:
                self.iterations += 1

        # During the invoke loop, formatted_answer alternates between AgentAction
        # (when the agent is using tools) and eventually becomes AgentFinish
        # (when the agent reaches a final answer). This assertion confirms we've
        # reached a final answer and helps type checking understand this transition.
        assert isinstance(formatted_answer, AgentFinish)
        self._show_logs(formatted_answer)
        return formatted_answer

    def _show_logs(self, formatted_answer: Union[AgentAction, AgentFinish]):
        """Show logs for the agent's execution."""
        if self.agent is None:
            raise ValueError("Agent cannot be None")
        
        show_agent_logs(
            printer=self._printer,
            agent_role=self.agent.role,
            formatted_answer=formatted_answer,
            verbose=self.agent.verbose
            or (hasattr(self, "crew") and getattr(self.crew, "verbose", False)),
        )

        show_agent_logs_json(
            path=self.agent.task_execution_output_json_path,
            printer=self._printer,
            agent_role=self.agent.role,
            formatted_answer=formatted_answer,
            verbose=self.agent.verbose
            or (hasattr(self, "crew") and getattr(self.crew, "verbose", False)),
        )