import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

loader = TextLoader("./documents/sabotage_manual.txt")
docs = loader.load()
print(f"DOC TYPE: {type(docs)}")

embeddings = OpenAIEmbeddings()

textSplitter = RecursiveCharacterTextSplitter()

documents = textSplitter.split_documents(docs)
print(f"DOCUMENTS TYPE: {type(documents)}")

vector = FAISS.from_documents(documents, embeddings)

llm = ChatOpenAI(
    openai_api_key=os.environ['OPENAI_API_KEY'],
    temperature=0,
    model_name='gpt-4',
    max_retries=1
)

prompt = ChatPromptTemplate.from_template(
    """
    Using the provided context, answer the following question:

    <context>{context}</context>

    Question: {input}
    """
)

documentChain = create_stuff_documents_chain(llm, prompt)

retriever = vector.as_retriever()

retrievalChain = create_retrieval_chain(retriever, documentChain)

response = retrievalChain.invoke({"input":"How to impact employee morale?"})
print(f"RESPONSE TYPE: {type(response)}")
print(f"RESPONSE: \n{response['answer']}")
