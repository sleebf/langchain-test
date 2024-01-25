import os
from langchain.globals import set_verbose
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

set_verbose(True)

### INITIALIZE MODEL
llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    temperature=0,
    model_name='gpt-4',
    max_retries=1
)

### LOAD DOCUMENT
loader = TextLoader(file_path="./documents/lrr.txt")
document = loader.load()

### CREATE VECTOR STORE
embeddings = OpenAIEmbeddings()
textSplitter = RecursiveCharacterTextSplitter()
documentSplit = textSplitter.split_documents(documents=document)
documentVector = FAISS.from_documents(documents=documentSplit, embedding=embeddings)

### SET VECTOR STORE AS RETRIEVER
documentRetriever = documentVector.as_retriever()
print(f"RETRIEVER TYPE: {type(documentRetriever)}")
# <class 'langchain_core.vectorstores.VectorStoreRetriever'>

### CREATE TOOL
documentRetrieverTool = create_retriever_tool(
    retriever=documentRetriever,
    name="graham_stark",
    description="Information about Loading Ready Run. For any question about Graham Stark, use this tool."
)
print(f"RETRIEVER TOOL TYPE: {type(documentRetrieverTool)}")

### SET LIST OF TOOLS
tools = [documentRetrieverTool]

### CREATE PROMPT TEMPLATE FROM A SINGLE MESSAGE ASSUMED TO BE FROM THE HUMAN
templateStr = """
    Question: {input}
        
    Agent_scratchpad: {agent_scratchpad}
    """
promptTemplate = ChatPromptTemplate.from_template(template=templateStr)

### INITIALIZE AGENT WITH LLM, TOOLS, AND PROMPT
agent = create_openai_functions_agent(llm=llm, tools=tools, prompt=promptTemplate)
print(f"AGENT TYPE: {type(agent)}")

### ENABLE AGENT TO EXECUTE USING TOOLS
agentExecutor = AgentExecutor(agent=agent, tools=tools)
print(f"AGENT EXECUTOR TYPE: {type(agentExecutor)}")

### RUN AGENT
response = agentExecutor.invoke(input={"input":"In once sentence, who is Graham Stark?"})
print(f"RESPONSE TYPE: {type(response)}")

print(f"RESPONSE: \n{response['output']}")