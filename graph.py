from langchain.agents import create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools.render import format_tool_to_openai_function
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import FunctionMessage, SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor, ToolInvocation, chat_agent_executor, create_agent_executor
from src.unit_test_generator.agent_tools import read_file, write_file
from typing import TypedDict, Annotated, Union
import operator
import os

llm = ChatOpenAI(openai_api_key=os.environ['OPENAI_API_KEY'],
                 model="gpt-4",
                 temperature=0,
                 max_retries=1,
                 streaming=True,
                 verbose=True)

tools = [read_file, write_file]

#################################################
# USING PREBUILT TOOL EXECUTOR
#################################################
# AGENT OUTCOME: content="As an AI text-based model, I'm unable to perform file operations directly."
#################################################

# template = """Open file.
#             Remove one letter.
#             Save file.
#             Continue until there are no letters.
#             File: {file}"""

# input_variables = ["file"]

# human_prompt_template = PromptTemplate(template=template, input_variables=input_variables)

# chat_prompt_template = ChatPromptTemplate.from_messages(
#                 [
#                     SystemMessage(content="You are a helpful assistant."),
#                     HumanMessagePromptTemplate(prompt=human_prompt_template)
#                 ]
#             )

# file = ".\\generated_files\\test.txt"

# message = chat_prompt_template.format(file=file)

# tool_executor = ToolExecutor(tools)

# # CLASS RESPONSIBLE FOR CONSTRUCTING THE GRAPH
# class AgentState(TypedDict):
#     input: str
#     agent_outcome: Union[AgentAction,AgentFinish,None]
#     intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

# # DEFINE THE FUNCTION THAT CALLS THE MODEL
# def run_agent(state:AgentState):
#     agent_outcome = llm.invoke(input=state['input'])
#     print(f"AGENT OUTCOME: {agent_outcome}")
#     return {"agent_outcome": agent_outcome}

# # DEFINE THE FUNCTION TO EXECUTE TOOLS
# def execute_tools(state:AgentState):
#     agent_action = state['agent_outcome']
#     print(f"AGENT ACTION: {agent_action}")
#     output = tool_executor.invoke(agent_action)
#     return {"intermediate_steps": [(agent_action, str(output))]}

# # DEFINE THE FUNCTION THAT DETERMINES WHETHER TO CONTINUE OR NOT
# def should_continue(state:AgentState):
#     print(f"AGENT OUTCOME: {state['agent_outcome']}")
#     if isinstance(state['agent_outcome'], AgentFinish):
#         return "end"
#     else:
#         return "continue"
    
# # DEFINE A NEW GRAPH
# workflow = StateGraph(AgentState)

# # DEFINE THE NODES TO CYCLE BETWEEN
# workflow.add_node(key="agent", action=run_agent)
# workflow.add_node(key="action", action=execute_tools)

# # SET THE ENTRYPOINT AS `AGENT` (FIRST ONE CALLED)
# workflow.set_entry_point("agent")

# # ADD A CONDITIONAL EDGE
# workflow.add_conditional_edges(
#     # DEFINE THE START NODE
#     "agent",
#     # PASS IN THE FUNCTION THAT WILL DETERMINE WHICH NODE IS CALLED NEXT
#     should_continue,
#     # Pass in a mapping
#     # KEYS ARE STRINGS; VALUES ARE OTHER NODES
#     # END IS A SPECIAL NODE MARKING THAT THE GRAPH SHOULD FINISH
#     {
#         # IF `TOOLS`, THEN CALL TOOL NODE
#         "continue": "action",
#         # OTHERWISE FINISH
#         "end": END
#     }
# )

# # ADD A NORMAL EDGE FROM `TOOLS` TO `AGENT`
# workflow.add_edge('action', 'agent')
# print(f"WORKFLOW: {workflow}")

# # COMPILES INTO A LANGCHAIN RUNNABLE
# app = workflow.compile()
# print(f"APP: {app}")

# app.invoke(input={"input": message})

#################################################
# USING PREBUILT CHAT_AGENT_EXECUTOR CREATE_FUNCTION_CALLING_EXECUTOR
#################################################
# Recursion limit of 25 reachedwithout hitting a stop condition. You can increase the limitby setting the `recursion_limit` config key.
# STANDARD CONFIG IS SUCCESSFUL IF FILE CONTAINS ONE LETTER
#################################################

# template = """Open file.
#             Remove one letter.
#             Save file.
#             Continue until there are no letters.
#             File: {file}"""

# input_variables = ["file"]

# human_prompt_template = PromptTemplate(template=template, input_variables=input_variables)

# chat_prompt_template = ChatPromptTemplate.from_messages(
#                 [
#                     SystemMessage(content="You are a helpful assistant."),
#                     HumanMessagePromptTemplate(prompt=human_prompt_template)
#                 ]
#             )

# file = ".\\generated_files\\test.txt"

# message = chat_prompt_template.format(file=file)

# app = chat_agent_executor.create_function_calling_executor(llm, tools)

# app.invoke(input={"messages": [message]})

#################################################
# USING PREBUILT CREATE_AGENT_EXECUTOR
#################################################
# Recursion limit of 50 reachedwithout hitting a stop condition.
#################################################

template = """Open file.
            Remove one letter.
            Save file.
            Continue until there are no letters.
            File: {input}
            Agent_scratchpad: {agent_scratchpad}"""

input_variables = ["input"]

human_prompt_template = PromptTemplate(template=template, input_variables=input_variables)

chat_prompt_template = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content="You are a helpful assistant."),
                    HumanMessagePromptTemplate(prompt=human_prompt_template)
                ]
            )

file = ".\\generated_files\\test.txt"

agent = create_openai_functions_agent(llm, tools, chat_prompt_template)

app = create_agent_executor(agent, tools)

app.invoke(input={"input": file}, config={"recursion_limit": 50})