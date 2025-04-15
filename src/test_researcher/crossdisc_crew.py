# from crewai import Agent, Crew, Process, Task
from crewai import Process
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from myllm.routers import ChatOpenRouter
from ipdb import set_trace
from source.custom_crew import Custom_Crew
from source.custom_agent import Custom_Agent
from source.custom_task import Custom_Task
from pydantic import (
    model_validator,
)
from crewai.agents.cache import CacheHandler
from crewai.utilities.events.event_listener import EventListener
from crewai.utilities import I18N, Logger, RPMController
from crewai.utilities.llm_utils import create_llm
from crewai.llm import LLM, BaseLLM
import os
from langchain_openai import ChatOpenAI
import os
from .init_args import args
import json
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

output_log_file = f"{args.log_dir}/{args.query}/{timestamp}.json"
task_execution_output_json_path = f"{args.execution_logs_dir}/{args.query}/{timestamp}.json"

os.makedirs(f"{args.log_dir}/{args.query}", exist_ok=True)
os.makedirs(f"{args.execution_logs_dir}/{args.query}", exist_ok=True)


if args.model_name == 'qwen':
    os.environ["OPENAI_API_BASE"] = "http://localhost:8001/v1"
    os.environ["OPENAI_API_KEY"] = "NA"
    manager_llm = ChatOpenAI(model="openai/qwen72b")
    executor_llm = ChatOpenAI(model="openai/qwen72b")
elif args.model_name == 'deepseek-v3':
    os.environ["OPENAI_API_BASE"] = "https://uni-api.cstcloud.cn/v1"
    manager_llm = ChatOpenAI(model="openai/deepseek-v3:671b")
    executor_llm = ChatOpenAI(model="openai/deepseek-v3:671b")
else:
    assert False
    # executor_llm = ChatOpenRouter(model_name="openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free", temperature=0.4)
    # manager_llm = ChatOpenRouter(model_name="openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free", temperature=0.4)


config_file = f'/Users/chenxi/Desktop/Projects/SciTown/src/test_researcher/config/{args.task_name}.json'

@CrewBase
class TestResearcher():
    agents_config = f'config/{args.task_name}/agents.yaml'
    tasks_config = f'config/{args.task_name}/tasks.yaml'

    @agent
    def Biology_export(self) -> Custom_Agent:
        return Custom_Agent(
            config=self.agents_config['Biology_export'],
            llm=executor_llm,
            verbose=True
        )
    
    @agent
    def Physics_export(self) -> Custom_Agent:
        return Custom_Agent(
            config=self.agents_config['Physics_export'],
            llm=executor_llm,
            verbose=True
        )
    
    @agent
    def Mathematics_export(self) -> Custom_Agent:
        return Custom_Agent(
            config=self.agents_config['Mathematics_export'],
            llm=executor_llm,
            verbose=True
        )
    
    @agent
    def Chemistry_export(self) -> Custom_Agent:
        return Custom_Agent(
            config=self.agents_config['Chemistry_export'],
            llm=executor_llm,
            verbose=True
        )
    
    @agent
    def Geography_export(self) -> Custom_Agent:
        return Custom_Agent(
            config=self.agents_config['Geography_export'],
            llm=executor_llm,
            verbose=True
        )
    
    @agent
    def AI_export(self) -> Custom_Agent:
        return Custom_Agent(
            config=self.agents_config['AI_export'],
            llm=executor_llm,
            verbose=True
        )
    
    @task
    def Decompose_Problem_Into_Subtasks(self) -> Custom_Task:
        return Custom_Task(
            config=self.tasks_config['Decompose_Problem_Into_Subtasks'],
            llm=manager_llm,
            verbose=True
        )
    
    @task
    def Subtask_MultiDomain_Expert_Analysis(self) -> Custom_Task:
        return Custom_Task(
            config=self.tasks_config['Subtask_MultiDomain_Expert_Analysis'],
            llm=executor_llm,
            verbose=True
        )
    
    @task
    def CrossDomain_Support_Expansion(self) -> Custom_Task:
        return Custom_Task(
            config=self.tasks_config['CrossDomain_Support_Expansion'],
            llm=executor_llm,
            verbose=True
        )
    
    @task
    def Final_Solution_Proposal(self) -> Custom_Task:
        return Custom_Task(
            config=self.tasks_config['Final_Solution_Proposal'],
            llm=executor_llm,
            verbose=True
        )

    @crew
    def crew(self) -> Custom_Crew:

        return Custom_Crew(
            # config=json.load(open(self.config_file, 'r')),
            task_execution_output_json_path=task_execution_output_json_path,
            agents=self.agents, 
            tasks=self.tasks, 
            manager_llm=manager_llm,
            # planning_llm=planning_llm,
            # planning=True,
            verbose=True,
            output_log_file=output_log_file,
            # _file_handler=FileHandler(self.output_log_file)
            process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )