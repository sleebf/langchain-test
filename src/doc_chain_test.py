import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.output_parsers import JsonOutputParser

### INITIALIZE MODEL
llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    temperature=0,
    model_name='gpt-3.5-turbo',
    max_retries=1
)

### CREATE PROMPT TEMPLATE FROM A SINGLE MESSAGE ASSUMED TO BE FROM THE HUMAN
promptTemplate = ChatPromptTemplate.from_template(template=
    """
    Based on the provided context, determine the three top areas of influence that an entity has. Return your answer in JSON format using the provided template.

    <context>{context}</context>

    <template>{template}</template>
    """
)

### LOAD DOCUMENT
loader = TextLoader("./documents/job_description.txt")
jobDescription = loader.load()

### INITIALIZE OUTPUT TYPE AS A JSON
jsonOutputParser = JsonOutputParser()

### CREATE CHAIN FOR PASSING A LIST OF DOCUMENTS TO A MODEL
documentChain = create_stuff_documents_chain(llm=llm, prompt=promptTemplate, output_parser=jsonOutputParser)

### SET JSON TEMPLATE FOR RESPONSE
jsonTemplate = """
    {
        "influence_areas": [
            { "area": "area 1 name", "description": "area 1 description" },
            { "area": "area 2 name", "description": "area 2 description" }
        ]
    }
    
"""

### RUN DOCUMENT CHAIN WITH INPUT OF CONTEXT AND TEMPLATE
answer = documentChain.invoke(input={"context": jobDescription, "template": jsonTemplate})

print(f"ANSWER TYPE: {type(answer)}")
# <class 'dict'>

print(f"ANSWER: \n{answer}")
# {'influence_areas': [{'area': 'Financial Management', 'description': 'Responsible for financial planning, budgeting, forecasting, revenue cycle management, accounts payable management, financial and regulatory reporting, and coordination with the central billing office.'}, {'area': 'Operational Management', 'description': 'Oversees business office operations, patient access departmental operations, finance and accounts payable operations, medical records (HIM) operations, and shipping & receiving operations.'}, {'area': 'Compliance and Ethics', 'description': 'Ensures hospital and corporate compliance per policies and regulatory procedures. Updates, monitors, establishes, and educates internal controls program for compliance. Also serves as the Ethics and Compliance Officer.'}]}