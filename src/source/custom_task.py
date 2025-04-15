import datetime
import inspect
import json
import logging
import re
import threading
import uuid
from concurrent.futures import Future
from copy import copy
from hashlib import md5
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
)

from pydantic import (
    UUID4,
    BaseModel,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError

from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.security import Fingerprint, SecurityConfig
from crewai.tasks.guardrail_result import GuardrailResult
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from crewai.tools.base_tool import BaseTool
from crewai.utilities.config import process_config
from crewai.utilities.converter import Converter, convert_to_model
from crewai.utilities.events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.utilities.events.crewai_event_bus import crewai_event_bus
from crewai.utilities.i18n import I18N
from crewai.utilities.printer import Printer
from crewai.utilities.string_utils import interpolate_only

from crewai import Task

class Custom_Task(Task):
    def custom_func():
        pass
