from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain.tools import tool
import os

@tool
def read_file(file_path:str):
    """Read file from local directory"""

    with open(file_path, 'r') as f:
            return "File contents: " + f.read()
    
@tool
def write_file(file_path:str, contents:str):
    """Write file to local directory"""

    with open(file_path, 'w') as f:
        f.write(contents)
        return "File '" + str(file_path) + "' saved."
    
class UnitTestAgent():
    
    def __init__(
        self,
        open_api_key:str,
        model:str,
        temp:float,
        max_retries:int,
        streaming:bool,
        verbose:bool,
    ) -> None:
        
        llm = ChatOpenAI( 
            openai_api_key =open_api_key,
            model_name     =model,
            temperature    =temp,
            max_retries    =max_retries,
            streaming      =streaming,
            verbose        =verbose
        )

        human_message = """
            Based on the following code create a python test class.
            Save the generated test to the local directory .\\generated_files\\ .
            Execute generated test.
            If there are any errors, fix them until resolved.

            {code}
        """

        input_variables = ["code"]

        human_message_test = """
            Save the following code to the local directory .\\generated_files\\ .
            Then read contents of saved file.

            {code}
        """

        human_prompt_template = PromptTemplate(template=human_message_test, input_variables=input_variables)

        chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content="You are a professional python developer."),
                    HumanMessagePromptTemplate(prompt=human_prompt_template),
                    MessagesPlaceholder(variable_name='agent_scratchpad')
                ]
            )
        
        tools = [read_file, write_file]
        
        file_agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=chat_prompt_template)

        self.file_agent_executor = AgentExecutor(agent=file_agent, tools=tools, verbose=True)

        print("INIT COMPLETE")

    def run_agent_executor(self, input_values:dict):
        
        self.file_agent_executor.invoke(input_values)

        print("RUN COMPLETE")

def run_agent():

    agent = UnitTestAgent(
        open_api_key=os.environ['OPENAI_API_KEY'],
        model="gpt-4",
        temp=0,
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