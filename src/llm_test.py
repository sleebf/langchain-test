import os
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    temperature=0,
    model_name='gpt-4',
    max_retries=1
)
print(f"LLM: {llm}")
      
answer = llm.invoke("what is the difference between langchain_core.prompts.chat.ChatPromptTemplate and langchain.prompts.chat.ChatPromptTemplate?")
print(f"ANSWER TYPE: {answer}")
print(f"ANSWER: \n{answer}")