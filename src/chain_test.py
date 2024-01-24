import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    temperature=0,
    model_name='gpt-4',
    max_retries=1
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a dog."),
        ("user", "{input}")
    ]
)
print(f"PROMPT: {prompt}")

output_parser = StrOutputParser()

chain = prompt | llm | output_parser
print(f"CHAIN: {chain}")

answer = chain.invoke({"input": "Who is a good boy?"})
print(type(answer))
print(f"ANSWER: \n{answer}")