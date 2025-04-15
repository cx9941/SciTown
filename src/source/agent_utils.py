import json
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
)
from crewai.utilities import I18N, Printer

def show_agent_logs_json(
    path: str,
    printer: Printer,
    agent_role: str,
    formatted_answer: Optional[Union[AgentAction, AgentFinish]] = None,
    task_description: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Show agent logs for both start and execution states.

    Args:
        printer: Printer instance for output
        agent_role: Role of the agent
        formatted_answer: Optional AgentAction or AgentFinish for execution logs
        task_description: Optional task description for start logs
        verbose: Whether to show verbose output
    """
    if not verbose:
        return

    agent_role = agent_role.split("\n")[0]

    if formatted_answer is None:
        # Start logs
        printer.print(
            content=f"\033[1m\033[95m# Agent:\033[00m \033[1m\033[92m{agent_role}\033[00m"
        )
        if task_description:
            printer.print(
                content=f"\033[95m## Task:\033[00m \033[92m{task_description}\033[00m"
            )
    else:
        # Execution logs
        printer.print(
            content=f"\n\n\033[1m\033[95m# Agent:\033[00m \033[1m\033[92m{agent_role}\033[00m"
        )

        if isinstance(formatted_answer, AgentAction):
            thought = re.sub(r"\n+", "\n", formatted_answer.thought)
            formatted_json = json.dumps(
                formatted_answer.tool_input,
                indent=2,
                ensure_ascii=False,
            )
            if thought and thought != "":
                printer.print(
                    content=f"\033[95m## Thought:\033[00m \033[92m{thought}\033[00m"
                )
            printer.print(
                content=f"\033[95m## Using tool:\033[00m \033[92m{formatted_answer.tool}\033[00m"
            )
            printer.print(
                content=f"\033[95m## Tool Input:\033[00m \033[92m\n{formatted_json}\033[00m"
            )
            printer.print(
                content=f"\033[95m## Tool Output:\033[00m \033[92m\n{formatted_answer.result}\033[00m"
            )
        elif isinstance(formatted_answer, AgentFinish):
            printer.print(
                content=f"\033[95m## Final Answer:\033[00m \033[92m\n{formatted_answer.output}\033[00m\n\n"
            )
