from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.agents import create_openai_tools_agent, AgentExecutor
from src.unit_test_generator.agent_tools import read_file, write_file
    
class UnitTestAgent():
    
    def __init__(self,
                openai_api_key:str,
                model:str,
                temperpature:float,
                max_retries:int,
                streaming:bool,
                verbose:bool) -> None:
        
        llm = ChatOpenAI(openai_api_key=openai_api_key,
                        model_name=model,
                        temperature=temperpature,
                        max_retries=max_retries,
                        streaming=streaming,
                        verbose=verbose)

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
