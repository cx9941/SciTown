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
from crewai.task import Task
from concurrent.futures import Future
from crewai.tools.base_tool import BaseTool, Tool
from crewai.tasks.conditional_task import ConditionalTask
from crewai.crews.crew_output import CrewOutput

class Custom_Crew(Crew):

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



    def _execute_tasks(
        self,
        tasks: List[Task],
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
        futures: List[Tuple[Task, Future[TaskOutput], int]] = []
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
