import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

### CREATE PROMPT TEMPLATE FROM A LIST OF MESSAGE TEMPLATES
promptTemplate = ChatPromptTemplate.from_messages(messages=
    [
        ("system", "You are a dog."),
        ("user", "{input}")
    ]
)
print(f"PROMPT TEMPLATE TYPE: {type(promptTemplate)}")
# <class 'langchain_core.prompts.chat.ChatPromptTemplate'>

### INITIALIZE MODEL
llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    temperature=0,
    model_name='gpt-4',
    max_retries=1
)

### INITIALIZE OUTPUT TYPE AS A STRING
outputParser = StrOutputParser()

### CREATE CHAIN
chain = promptTemplate | llm | outputParser

### RUN CHAIN WITH INPUT
answer = chain.invoke(input={"input": "Who is a good boy?"})

print(f"ANSWER TYPE: {type(answer)}")
# <class 'str'>

print(f"ANSWER: \n{answer}")
# "As a dog, I would wag my tail excitedly and bark happily, indicating that I am the good boy.""