import shutil
import subprocess
from typing import Any, Dict, List, Literal, Optional, Sequence, Type, Union

from pydantic import Field, InstanceOf, PrivateAttr, model_validator

from crewai.agents import CacheHandler
from crewai.agents.agent_builder.base_agent import BaseAgent
from .custom_crew_agent_executor import Custom_CrewAgentExecutor
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource
from crewai.knowledge.utils.knowledge_utils import extract_knowledge_context
from crewai.lite_agent import LiteAgent, LiteAgentOutput
from crewai.llm import BaseLLM
from crewai.memory.contextual.contextual_memory import ContextualMemory
from crewai.security import Fingerprint
from crewai.task import Task
from crewai.tools import BaseTool
from crewai.tools.agent_tools.agent_tools import AgentTools
from crewai.utilities import Converter, Prompts
from crewai.utilities.agent_utils import (
    get_tool_names,
    parse_tools,
    render_text_description_and_args,
)
from crewai.utilities.constants import TRAINED_AGENTS_DATA_FILE, TRAINING_DATA_FILE
from crewai.utilities.converter import generate_model_description
from crewai.utilities.events.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.llm_utils import create_llm
from crewai.utilities.token_counter_callback import TokenCalcHandler
from crewai.utilities.training_handler import CrewTrainingHandler

from crewai import Agent

class Custom_Agent(Agent):
    def execute_task(
        self,
        task: Task,
        context: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> str:
        """Execute a task with the agent.

        Args:
            task: Task to execute.
            context: Context to execute the task in.
            tools: Tools to use for the task.

        Returns:
            Output of the agent
        """
        if self.tools_handler:
            self.tools_handler.last_used_tool = {}  # type: ignore # Incompatible types in assignment (expression has type "dict[Never, Never]", variable has type "ToolCalling")

        task_prompt = task.prompt()

        # If the task requires output in JSON or Pydantic format,
        # append specific instructions to the task prompt to ensure
        # that the final answer does not include any code block markers
        if task.output_json or task.output_pydantic:
            # Generate the schema based on the output format
            if task.output_json:
                # schema = json.dumps(task.output_json, indent=2)
                schema = generate_model_description(task.output_json)
                task_prompt += "\n" + self.i18n.slice(
                    "formatted_task_instructions"
                ).format(output_format=schema)

            elif task.output_pydantic:
                schema = generate_model_description(task.output_pydantic)
                task_prompt += "\n" + self.i18n.slice(
                    "formatted_task_instructions"
                ).format(output_format=schema)

        if context:
            task_prompt = self.i18n.slice("task_with_context").format(
                task=task_prompt, context=context
            )

        if self.crew and self.crew.memory:
            contextual_memory = ContextualMemory(
                self.crew.memory_config,
                self.crew._short_term_memory,
                self.crew._long_term_memory,
                self.crew._entity_memory,
                self.crew._user_memory,
                self.crew._external_memory,
            )
            memory = contextual_memory.build_context_for_task(task, context)
            if memory.strip() != "":
                task_prompt += self.i18n.slice("memory").format(memory=memory)

        if self.knowledge:
            agent_knowledge_snippets = self.knowledge.query([task.prompt()])
            if agent_knowledge_snippets:
                agent_knowledge_context = extract_knowledge_context(
                    agent_knowledge_snippets
                )
                if agent_knowledge_context:
                    task_prompt += agent_knowledge_context

        if self.crew:
            knowledge_snippets = self.crew.query_knowledge([task.prompt()])
            if knowledge_snippets:
                crew_knowledge_context = extract_knowledge_context(knowledge_snippets)
                if crew_knowledge_context:
                    task_prompt += crew_knowledge_context

        tools = tools or self.tools or []
        self.create_agent_executor(tools=tools, task=task)

        if self.crew and self.crew._train:
            task_prompt = self._training_handler(task_prompt=task_prompt)
        else:
            task_prompt = self._use_trained_data(task_prompt=task_prompt)

        try:
            crewai_event_bus.emit(
                self,
                event=AgentExecutionStartedEvent(
                    agent=self,
                    tools=self.tools,
                    task_prompt=task_prompt,
                    task=task,
                ),
            )
            result = self.agent_executor.invoke(
                {
                    "input": task_prompt,
                    "tool_names": self.agent_executor.tools_names,
                    "tools": self.agent_executor.tools_description,
                    "ask_for_human_input": task.human_input,
                }
            )["output"]
        except Exception as e:
            if e.__class__.__module__.startswith("litellm"):
                # Do not retry on litellm errors
                crewai_event_bus.emit(
                    self,
                    event=AgentExecutionErrorEvent(
                        agent=self,
                        task=task,
                        error=str(e),
                    ),
                )
                raise e
            self._times_executed += 1
            if self._times_executed > self.max_retry_limit:
                crewai_event_bus.emit(
                    self,
                    event=AgentExecutionErrorEvent(
                        agent=self,
                        task=task,
                        error=str(e),
                    ),
                )
                raise e
            result = self.execute_task(task, context, tools)

        if self.max_rpm and self._rpm_controller:
            self._rpm_controller.stop_rpm_counter()

        # If there was any tool in self.tools_results that had result_as_answer
        # set to True, return the results of the last tool that had
        # result_as_answer set to True
        for tool_result in self.tools_results:  # type: ignore # Item "None" of "list[Any] | None" has no attribute "__iter__" (not iterable)
            if tool_result.get("result_as_answer", False):
                result = tool_result["result"]
        crewai_event_bus.emit(
            self,
            event=AgentExecutionCompletedEvent(agent=self, task=task, output=result),
        )
        return result

    def create_agent_executor(
        self, tools: Optional[List[BaseTool]] = None, task=None
    ) -> None:
        """Create an agent executor for the agent.

        Returns:
            An instance of the CrewAgentExecutor class.
        """
        raw_tools: List[BaseTool] = tools or self.tools or []
        parsed_tools = parse_tools(raw_tools)

        prompt = Prompts(
            agent=self,
            has_tools=len(raw_tools) > 0,
            i18n=self.i18n,
            use_system_prompt=self.use_system_prompt,
            system_template=self.system_template,
            prompt_template=self.prompt_template,
            response_template=self.response_template,
        ).task_execution()

        stop_words = [self.i18n.slice("observation")]

        if self.response_template:
            stop_words.append(
                self.response_template.split("{{ .Response }}")[1].strip()
            )

        self.agent_executor = Custom_CrewAgentExecutor(
            llm=self.llm,
            task=task,
            agent=self,
            crew=self.crew,
            tools=parsed_tools,
            prompt=prompt,
            original_tools=raw_tools,
            stop_words=stop_words,
            max_iter=self.max_iter,
            tools_handler=self.tools_handler,
            tools_names=get_tool_names(parsed_tools),
            tools_description=render_text_description_and_args(parsed_tools),
            step_callback=self.step_callback,
            function_calling_llm=self.function_calling_llm,
            respect_context_window=self.respect_context_window,
            request_within_rpm_limit=(
                self._rpm_controller.check_or_wait if self._rpm_controller else None
            ),
            callbacks=[TokenCalcHandler(self._token_process)],
        )