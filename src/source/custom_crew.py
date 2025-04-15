from crewai import Crew  # 假设你使用的是 CrewAI 或类似结构
from pydantic import (
    model_validator,
)
from crewai.agents.cache import CacheHandler
from crewai.utilities.events.event_listener import EventListener
from crewai.utilities import I18N, Logger, RPMController
from crewai.utilities.llm_utils import create_llm
from crewai.llm import LLM, BaseLLM
from .filehandler import FileHandler
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast
from crewai.tasks.task_output import TaskOutput
# from crewai.task import Task
from concurrent.futures import Future
from crewai.tools.base_tool import BaseTool, Tool
from crewai.tasks.conditional_task import ConditionalTask
from crewai.crews.crew_output import CrewOutput
from pydantic_core import PydanticCustomError
from crewai.tools.agent_tools.agent_tools import AgentTools

from .custom_task import Custom_Task
from .custom_agent import Custom_Agent
from pydantic import PrivateAttr

class Custom_Crew(Crew):

    # @model_validator(mode="after")
    # def check_config(self):
    #     """Validates that the crew is properly configured with agents and tasks."""
    #     if not self.config and not self.tasks and not self.agents:
    #         raise PydanticCustomError(
    #             "missing_keys",
    #             "Either 'agents' and 'tasks' need to be set or 'config'.",
    #             {},
    #         )

    #     if self.config:
    #         self._setup_from_config()

    #     if self.agents:
    #         for agent in self.agents:
    #             if self.cache:
    #                 agent.set_cache_handler(self._cache_handler)
    #             if self.max_rpm:
    #                 agent.set_rpm_controller(self._rpm_controller)
    #     return self

    @model_validator(mode="after")
    def set_private_attrs(self) -> "Crew":
        """Set private attributes."""

        self._cache_handler = CacheHandler()
        event_listener = EventListener()
        event_listener.verbose = self.verbose
        event_listener.formatter.verbose = self.verbose
        self._logger = Logger(verbose=self.verbose)
        if self.output_log_file:
            self._file_handler = FileHandler(self.output_log_file)
        self._rpm_controller = RPMController(max_rpm=self.max_rpm, logger=self._logger)
        if self.function_calling_llm and not isinstance(self.function_calling_llm, LLM):
            self.function_calling_llm = create_llm(self.function_calling_llm)

        return self

    def _setup_from_config(self):
        assert self.config is not None, "Config should not be None."

        """Initializes agents and tasks from the provided config."""
        if not self.config.get("agents") or not self.config.get("tasks"):
            raise PydanticCustomError(
                "missing_keys_in_config", "Config should have 'agents' and 'tasks'.", {}
            )

        self.process = self.config.get("process", self.process)
        self.agents = [Custom_Agent(**agent) for agent in self.config["agents"]]
        
        # for i in self.agents:
        #     i.task_execution_output_json_path = self.task_execution_output_json_path

        self.tasks = [self._create_task(task) for task in self.config["tasks"]]

    def _create_manager_agent(self):
        i18n = I18N(prompt_file=self.prompt_file)
        if self.manager_agent is not None:
            self.manager_agent.allow_delegation = True
            manager = self.manager_agent
            if manager.tools is not None and len(manager.tools) > 0:
                self._logger.log(
                    "warning", "Manager agent should not have tools", color="orange"
                )
                manager.tools = []
                raise Exception("Manager agent should not have tools")
        else:
            self.manager_llm = create_llm(self.manager_llm)
            manager = Custom_Agent(
                role=i18n.retrieve("hierarchical_manager_agent", "role"),
                goal=i18n.retrieve("hierarchical_manager_agent", "goal"),
                backstory=i18n.retrieve("hierarchical_manager_agent", "backstory"),
                tools=AgentTools(agents=self.agents).tools(),
                allow_delegation=True,
                llm=self.manager_llm,
                verbose=self.verbose,
            )
            self.manager_agent = manager
        manager.crew = self

    def _execute_tasks(
        self,
        tasks: List[Custom_Task],
        start_index: Optional[int] = 0,
        was_replayed: bool = False,
    ) -> CrewOutput:
        """Executes tasks sequentially and returns the final output.

        Args:
            tasks (List[Task]): List of tasks to execute
            manager (Optional[BaseAgent], optional): Manager agent to use for delegation. Defaults to None.

        Returns:
            CrewOutput: Final output of the crew
        """

        task_outputs: List[TaskOutput] = []
        futures: List[Tuple[Custom_Task, Future[TaskOutput], int]] = []
        last_sync_output: Optional[TaskOutput] = None

        for task_index, task in enumerate(tasks):
            if start_index is not None and task_index < start_index:
                if task.output:
                    if task.async_execution:
                        task_outputs.append(task.output)
                    else:
                        task_outputs = [task.output]
                        last_sync_output = task.output
                continue

            agent_to_use = self._get_agent_to_use(task)
            if agent_to_use is None:
                raise ValueError(
                    f"No agent available for task: {task.description}. Ensure that either the task has an assigned agent or a manager agent is provided."
                )

            # Determine which tools to use - task tools take precedence over agent tools
            tools_for_task = task.tools or agent_to_use.tools or []
            # Prepare tools and ensure they're compatible with task execution
            tools_for_task = self._prepare_tools(
                agent_to_use,
                task,
                cast(Union[List[Tool], List[BaseTool]], tools_for_task),
            )

            self._log_task_start(task, agent_to_use.role)

            if isinstance(task, ConditionalTask):
                skipped_task_output = self._handle_conditional_task(
                    task, task_outputs, futures, task_index, was_replayed
                )
                if skipped_task_output:
                    task_outputs.append(skipped_task_output)
                    continue

            if task.async_execution:
                context = self._get_context(
                    task, [last_sync_output] if last_sync_output else []
                )
                future = task.execute_async(
                    agent=agent_to_use,
                    context=context,
                    tools=cast(List[BaseTool], tools_for_task),
                )
                futures.append((task, future, task_index))
            else:
                if futures:
                    task_outputs = self._process_async_tasks(futures, was_replayed)
                    futures.clear()

                context = self._get_context(task, task_outputs)
                task_output = task.execute_sync(
                    agent=agent_to_use,
                    context=context,
                    tools=cast(List[BaseTool], tools_for_task),
                )
                task_outputs.append(task_output)
                self._process_task_result(task, task_output)
                self._store_execution_log(task, task_output, task_index, was_replayed)

        if futures:
            task_outputs = self._process_async_tasks(futures, was_replayed)

        return self._create_crew_output(task_outputs)
