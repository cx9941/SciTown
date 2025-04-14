from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from myllm.routers import ChatOpenRouter
from ipdb import set_trace
from source.custom_crew import Custom_Crew
from crewai import Crew  # 假设你使用的是 CrewAI 或类似结构
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

os.environ["OPENAI_API_BASE"] = "http://10.0.82.212:8865/v1"
os.environ["OPENAI_API_KEY"] = "NA"

manager_llm = ChatOpenAI(model="openai/llama8b")
executor_llm = ChatOpenAI(model="openai/llama8b")

# executor_llm = ChatOpenRouter(model_name="openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free", temperature=0.4)
# manager_llm = ChatOpenRouter(model_name="openrouter/nvidia/llama-3.1-nemotron-70b-instruct:free", temperature=0.4)

@CrewBase
class TestResearcher():
    agents_config = 'config/crossdisc/agents.yaml'
    tasks_config = 'config/crossdisc/tasks.yaml'

    @agent
    def Biology_export(self) -> Agent:
        return Agent(
            config=self.agents_config['Biology_export'],
            llm=executor_llm,
            verbose=True
        )
    
    @agent
    def Physics_export(self) -> Agent:
        return Agent(
            config=self.agents_config['Physics_export'],
            llm=executor_llm,
            verbose=True
        )
    
    @agent
    def Mathematics_export(self) -> Agent:
        return Agent(
            config=self.agents_config['Mathematics_export'],
            llm=executor_llm,
            verbose=True
        )
    
    @agent
    def Chemistry_export(self) -> Agent:
        return Agent(
            config=self.agents_config['Chemistry_export'],
            llm=executor_llm,
            verbose=True
        )
    
    @agent
    def Geography_export(self) -> Agent:
        return Agent(
            config=self.agents_config['Geography_export'],
            llm=executor_llm,
            verbose=True
        )
    
    @agent
    def AI_export(self) -> Agent:
        return Agent(
            config=self.agents_config['AI_export'],
            llm=executor_llm,
            verbose=True
        )
    
    @agent
    def Integrative_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['Integrative_researcher'],
            llm=executor_llm,
            verbose=True
        )
    
    @task
    def Step1_Main_Expert_Response(self) -> Task:
        return Task(
            config=self.tasks_config['Step1_Main_Expert_Response'],
            llm=manager_llm,
            verbose=True
        )
    
    @task
    def Step2_Cross_Domain_Support(self) -> Task:
        return Task(
            config=self.tasks_config['Step2_Cross_Domain_Support'],
            llm=executor_llm,
            verbose=True
        )
    
    @task
    def Step3_Integrated_Summary_Response(self) -> Task:
        return Task(
            config=self.tasks_config['Step3_Integrated_Summary_Response'],
            llm=executor_llm,
            verbose=True
        )

    @crew
    def crew(self) -> Custom_Crew:
        """Creates the TestResearcher crew"""
        return Custom_Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            manager_llm=manager_llm,
            # planning_llm=planning_llm,
            # planning=True,
            verbose=True,
            output_log_file="../outputs/corssdisc/log.json",
            # _file_handler=FileHandler(self.output_log_file)
            process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )