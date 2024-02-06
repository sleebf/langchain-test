import os
from src.unit_test_generator.agent import UnitTestAgent

def run_agent():

    agent = UnitTestAgent(
        openai_api_key=os.environ['OPENAI_API_KEY'],
        model="gpt-4",
        temperpature=0,
        max_retries=1,
        streaming=True,
        verbose=True,
    )

    code = """
        class Test():
            print("Howdy!")

    """

    input_values = {"code": code}

    agent.run_agent_executor(input_values=input_values)

if __name__ == "__main__":
    
    run_agent()