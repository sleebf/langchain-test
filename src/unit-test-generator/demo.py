from src.unit-test-generator.agent import Agent
import os

def run_agent():

    agent = Agent(
        open_api_key=os.environ['OPENAI_API_KEY'],
        model="gpt-4",
        temp=0,
        max_retries=1,
        streaming=True,
        verbose=True,
    )

    input_values = {"code": code}

    agent.run_agent_executor(input_values=input_values)

if __name__ == "__main__":
    
    run_agent()