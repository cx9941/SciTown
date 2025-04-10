from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from myllm.routers import ChatOpenRouter
from ipdb import set_trace


@CrewBase
class TestResearcher():
    agents_config = 'config/MAST_agents.yaml'
    tasks_config = 'config/MAST_tasks.yaml'

    @agent
    def AI_export(self) -> Agent:
        return Agent(
            config=self.agents_config['AI_export'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @agent
    def Microbiology_export(self) -> Agent:
        return Agent(
            config=self.agents_config['Microbiology_export'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @agent
    def Integrative_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['Integrative_researcher'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @task
    def AI_export_answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['AI_export_answer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @task
    def Microbiology_export_answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['Microbiology_export_answer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @task
    def Integrative_researcher_summary_answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['Integrative_researcher_summary_answer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )

    @task
    def AI_export_reanswer_task(self) -> Task:
        return Task(
            config=self.tasks_config['AI_export_reanswer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @task
    def Microbiology_export_reanswer_task(self) -> Task:
        return Task(
            config=self.tasks_config['Microbiology_export_reanswer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )


    @crew
    def crew(self) -> Crew:
        """Creates the TestResearcher crew"""
        return Crew(
            agents=self.agents, 
            tasks=self.tasks, 
            # process=Process.sequential,
            verbose=True,
            output_log_file="log.json"
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
