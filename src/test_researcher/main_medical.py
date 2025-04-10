#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

# from test_researcher.crew import TestResearcher
from test_researcher.crew_medical import TestResearcher

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        "domain": "medical",
        "query": "A 23-year-old college student experiences monthly chest discomfort, shortness of breath,"
                    "shakiness, and excessive sweating, likely due to stress. He fears having an episode in public, causing"
                    "him to avoid leaving home. His medical history is unclear, and the physical exam is normal. Which"
                    "of the following is the best medication for the long-term management of this patient’s condition? (A)"
                    "Citalopram (B) Lithium (C) Lorazepam (D) Propranolol (E) Quetiapine",
        # "query": "How can the combination of radiological imaging, genomic data, and clinical history of breast cancer patients "
        #          "be used to optimize treatment plans and predict the risk of recurrence?",
        "current_year": str(datetime.now().year)
    }
    
    try:
        TestResearcher().crew(
            selected_agents=('Professor of Radiological Sciences', 'Professor of Clinical Medicine and Translational Therapeutics',
                             'Senior Research Fellow in Medical Knowledge Integration and Expert Coordination'),
            selected_tasks=('radiological_professor_answer_task', 'clinical_medicine_professor_answer_task', 'integrative_researcher_summary_answer_task')
        ).kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        TestResearcher().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        TestResearcher().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs",
        "current_year": str(datetime.now().year)
    }
    try:
        TestResearcher().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
