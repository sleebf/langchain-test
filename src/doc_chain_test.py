import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.chains.combine_documents import create_stuff_documents_chain

loader = TextLoader("./documents/job_description.txt")
jobDescription = loader.load()
print(f"JOB DESCRIPTION TYPE: {type(jobDescription)}")
print(f"JOB DESCRIPTION: \n{jobDescription}")

llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    temperature=0,
    model_name='gpt-4',
    max_retries=1
)

jsonTemplate = (
    """
    {
        'influence_areas': [
            { 'area': 'area 1 name', 'description': 'area 1 description' },
            { 'area': 'area 2 name', 'description': 'area 2 description' }
        ],
    }
    """
)
print(f"JSON TEMPLATE: {jsonTemplate}")

prompt = ChatPromptTemplate.from_template(
    """
    Based only on the provided context, determine the three top areas of influence that an entity has. Return your answer in JSON format using the provided template.

    <context>{context}</context>

    <template>{template}</template>
    """
)
print(f"PROMPT: {prompt}")

documentChain = create_stuff_documents_chain(llm, prompt)
print(f"DOCUMENT CHAIN: {documentChain}")

answer = documentChain.invoke({"context": jobDescription, "template": jsonTemplate})
print(type(answer))
print(f"ANSWER: \n{answer}")