from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, before_kickoff, after_kickoff
from myllm.routers import ChatOpenRouter
from ipdb import set_trace


@CrewBase
class TestResearcher():
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def radiological_professor(self) -> Agent:
        return Agent(
            config=self.agents_config['radiological_professor'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )

    @agent
    def immunology_professor(self) -> Agent:
        return Agent(
            config=self.agents_config['immunology_professor'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @agent
    def genomics_professor(self) -> Agent:
        return Agent(
            config=self.agents_config['genomics_professor'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @agent
    def biomedical_informatics_professor(self) -> Agent:
        return Agent(
            config=self.agents_config['biomedical_informatics_professor'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @agent
    def regenerative_medicine_professor(self) -> Agent:
        return Agent(
            config=self.agents_config['regenerative_medicine_professor'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @agent
    def clinical_medicine_professor(self) -> Agent:
        return Agent(
            config=self.agents_config['clinical_medicine_professor'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @agent
    def drug_discovery_professor(self) -> Agent:
        return Agent(
            config=self.agents_config['drug_discovery_professor'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @agent
    def medical_ethics_professor(self) -> Agent:
        return Agent(
            config=self.agents_config['medical_ethics_professor'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @agent
    def integrative_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['integrative_researcher'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @task
    def radiological_professor_answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['radiological_professor_answer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @task
    def immunology_professor_answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['immunology_professor_answer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @task
    def genomics_professor_answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['genomics_professor_answer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @task
    def biomedical_informatics_professor_answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['biomedical_informatics_professor_answer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @task
    def regenerative_medicine_professor_answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['regenerative_medicine_professor_answer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @task
    def clinical_medicine_professor_answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['clinical_medicine_professor_answer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @task
    def drug_discovery_professor_answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['drug_discovery_professor_answer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @task
    def integrative_researcher_summary_answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['integrative_researcher_summary_answer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @task
    def medical_ethics_professor_check_answer_task(self) -> Task:
        return Task(
            config=self.tasks_config['medical_ethics_professor_check_answer_task'],
            llm=ChatOpenRouter(model_name="openrouter/deepseek/deepseek-chat-v3-0324", temperature=0.4),
            verbose=True
        )
    
    @crew
    def crew(self, selected_agents: tuple = None, selected_tasks: tuple = None) -> Crew:
        """Creates the TestResearcher crew"""
        all_agents = self.agents
        # set_trace()
        if selected_agents:
            # Filter the agents based on the selected list
            all_agents = [agent for agent in self.agents if agent.agent_ops_agent_name.strip() in selected_agents]
        all_tasks = self.tasks
        if selected_tasks:
            # Filter the tasks based on the selected list
            all_tasks = [task for task in self.tasks if task.name in selected_tasks]
        return Crew(
            agents=all_agents, # Automatically created by the @agent decorator
            tasks=all_tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            output_log_file="log.json"
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
