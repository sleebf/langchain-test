import os
from langchain_openai.chat_models import ChatOpenAI

def llm_test():
    ### INITIALIZE MODEL
    llm = ChatOpenAI(
        openai_api_key=os.environ['OPENAI_API_KEY'],
        temperature=0,
        model_name='gpt-3.5-turbo',
        max_retries=1
    )

    ### RUN MODEL WITH INPUT
    answer = llm.invoke(input="In one sentence, what is a dog?")

    print(f"ANSWER TYPE: \n{type(answer)}")
    # <class 'langchain_core.messages.ai.AIMessage'>

    print(f"ANSWER.CONTENT TYPE: {type(answer.content)}")
    # <class 'str'>

    print(f"ANSWER: \n{answer.content}")
    # "A dog is a domesticated mammal that is commonly kept as a pet and known for its loyalty and companionship to humans."

if __name__ == "__main__":
    llm_test()