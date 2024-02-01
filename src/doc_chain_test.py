import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain

### SET JSON TEMPLATE FOR RESPONSE
class jsonTemplate(BaseModel):
    areas_of_influence: list = Field(description="list of area and description")
    area: str = Field(description="name of area of influnce")
    description: str = Field(description="description of area of influence")

def doc_chain_test():
    ### INITIALIZE MODEL
    llm = ChatOpenAI(
        openai_api_key=os.environ['OPENAI_API_KEY'],
        temperature=0,
        model_name='gpt-4',
        max_retries=1
    )

    ### CREATE PROMPT TEMPLATE FROM A SINGLE MESSAGE ASSUMED TO BE FROM THE HUMAN
    templateStr = """
        Based on the provided context, determine the three top areas of influence that an entity has.

        {context}

        {format_instructions}
        """

    promptTemplate = ChatPromptTemplate.from_template(template=templateStr)

    ### INITIALIZE OUTPUT TYPE AS A JSON
    jsonOutputParser = JsonOutputParser(pydantic_object=jsonTemplate)
    jsonOutputParserInstructions = jsonOutputParser.get_format_instructions()
    print(f"JSON OUTPUT PARSER INSTRUCTIONS: {jsonOutputParserInstructions}")
    # The output should be formatted as a JSON instance that conforms to the JSON schema below.

    # As an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}
    # the object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.

    # Here is the output schema:
    # ```
    # {"properties": {"areas_of_influence": {"title": "Areas Of Influence", "description": "list of area and description", "type": "array", "items": {}}, "area": {"title": "Area", "description": "name of area of influnce", "type": "string"}, "description": {"title": "Description", "description": "description of area of influence", "type": "string"}}, "required": ["areas_of_influence", "area", "description"]}
    # ```

    ### CREATE CHAIN FOR PASSING A LIST OF DOCUMENTS TO A MODEL
    documentChain = create_stuff_documents_chain(llm=llm, prompt=promptTemplate, output_parser=jsonOutputParser)

    ### LOAD DOCUMENT
    loader = TextLoader("./documents/job_description.txt")
    jobDescription = loader.load()

    ### RUN DOCUMENT CHAIN WITH INPUT OF CONTEXT AND TEMPLATE
    answer = documentChain.invoke(input={"context": jobDescription, "format_instructions": jsonOutputParserInstructions})

    print(f"ANSWER TYPE: {type(answer)}")
    # <class 'dict'>

    print(f"ANSWER: \n{answer}")
    # {'areas_of_influence': [{'area': 'Financial Management', 'description': 'Responsible for financial planning, budgeting and forecasting, revenue cycle management, accounts payable management, financial and regulatory reporting, and coordination with the central billing office. Operational responsibilities related to finance and accounts payable operations.'}, {'area': 'Compliance', 'description': 'Ensures the hospital and corporate compliance per policies and regulatory procedures. Updates, monitors, establishes, and educates the internal controls program for compliance. Serves as the Ethics and Compliance Officer.'}, {'area': 'Leadership and Communication', 'description': 'Senior management team member, providing overall leadership and management of the hospital. Participates in hospital-wide strategic and operational planning. Communicates directly with various stakeholders, including the hospital Management Team, staff, physicians, patients, Executive Committees, Governing Board, and other leadership.'}]}

if __name__ == "__main__":
    doc_chain_test()