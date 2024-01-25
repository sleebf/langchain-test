import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

### LOAD DOCUMENT
loader = TextLoader(file_path="./documents/lrr.txt")
docs = loader.load()

### CREATE VECTOR STORE
embeddings = OpenAIEmbeddings()
textSplitter = RecursiveCharacterTextSplitter()
documents = textSplitter.split_documents(documents=docs)
vector = FAISS.from_documents(documents=documents, embedding=embeddings)

### SET VECTOR STORE AS RETRIEVER
retriever = vector.as_retriever()
print(f"RETRIEVER TYPE: {type(retriever)}")
# <class 'langchain_core.vectorstores.VectorStoreRetriever'>

### INITIALIZE MODEL
llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    temperature=0,
    model_name='gpt-4',
    max_retries=1
)

### CREATE PROMPT TEMPLATE FROM A SINGLE MESSAGE ASSUMED TO BE FROM THE HUMAN
promptTemplate = ChatPromptTemplate.from_template(template=
    """
    Using the provided context, answer the following question:

    <context>{context}</context>

    Question: {input}
    """
)

### CREATE CHAIN FOR PASSING A LIST OF DOCUMENTS TO A MODEL
documentChain = create_stuff_documents_chain(llm=llm, prompt=promptTemplate)
print(f"DOCUMENT CHAIN TYPE: {type(documentChain)}")
# <class 'langchain_core.runnables.base.RunnableBinding'>

### CREATE RETIEVAL CHAIN THAT RETRIEVES DOCUMENTS THEN PASSES CONTEXT TO DOC CHAIN
retrievalChain = create_retrieval_chain(retriever=retriever, combine_docs_chain=documentChain)
print(f"RETRIEVAL CHAIN TYPE: {type(retrievalChain)}")
# <class 'langchain_core.runnables.base.RunnableBinding'>

### RUN RETRIEVAL CHAIN WITH INPUT
response = retrievalChain.invoke(input={"input":"Who is Graham Stark?"})
print(f"RESPONSE TYPE: {type(response)}")
# <class 'dict'>

print(f"RESPONSE: \n{response['answer']}")
# "The text suggests that employee morale can be impacted by promoting inefficient workers and giving them undeserved promotions, discriminating against efficient workers and unjustly complaining about their work, holding conferences when there is more critical work to be done, and multiplying paperwork."
