import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


def load_env_variables():
    """
    Loads environment variables from a .env file and sets the OPENAI_API_KEY environment variable.
    """
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

def initialize_llm(model_name="gpt-4o-mini"):
    """
    Initializes and returns an instance of the ChatOpenAI language model.

    This function creates an instance of the ChatOpenAI model using the specified model name.
    It provides an easy way to switch or update the model name as required by the application.

    Parameters:
    - model_name (str): The name of the language model to initialize. Defaults to "gpt-4o-mini".

    Returns:
    - ChatOpenAI: An instance of the initialized language model.
    """
    return ChatOpenAI(model=model_name)

def load_documents(doc_path):
    """
    Loads and splits documents from a provided PDF file.

    Parameters:
    - doc_path (str): The file path to the PDF file containing the documents.

    Returns:
    - list: A list of document objects obtained from the PDF file.
    """
    loader = PyPDFLoader(file_path=doc_path)
    return loader.load()

def chunking(docs):
    """
    Splits a list of documents into smaller chunks using a recursive character text splitter.

    Parameters:
    - docs (list): A list of document objects to be split into chunks.

    Returns:
    - list: A list containing the split document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    splitted_docs = text_splitter.split_documents(docs)
    print(f'TOTAL CHUNKS: {len(splitted_docs)}')
    return splitted_docs

def store_document_vectorstore(embeddings, docs):
    """
    Initializes a FAISS vector store and adds documents to it.

    This function creates a FAISS index based on the dimensionality of the provided embeddings.
    It then initializes a FAISS vector store with an in-memory docstore and populates it with the
    provided list of documents. A message is printed to indicate the progress of data addition.

    Parameters:
    - embeddings: The embeddings model used to generate query embeddings.
    - docs (list): A list of document objects to be added to the vector store.

    Returns:
    - FAISS: An instance of the FAISS vector store containing the added documents.
    """
    index = faiss.IndexFlatL2(len(embeddings.embed_query(" ")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    print('INSERTING DATA TO VECTOR DB.....')
    doc_ids = vector_store.add_documents(documents=docs)
    print(f'INSERTED {len(doc_ids)} TO VECTOR DB....')
    return vector_store

def create_rag_chain(llm, retriever):
    """
    Creates a retrieval-augmented generation (RAG) chain for processing user queries.

    Parameters:
    - llm: The language model (ChatOpenAI instance) used to generate responses.
    - retriever: The retriever component responsible for fetching relevant context from the vector store.

    Returns:
    - RetrievalChain: A retrieval-augmented generation chain configured to generate answers based on the retrieved context.
    """
    system_prompt = (
        "You are an AI assistant designed to answer user queries based on the provided retrieved context. "
        "Use the following pieces of retrieved context to answer the question"
        " If you don't know the answer, just say that you don't know.\n"

        "### Instructions:\n"
        "- Carefully analyze the retrieved context before answering.\n"
        "- Ensure responses are accurate, clear, and directly based on the retrieved data.\n"
        "- Keep answers concise (preferably 3-4 sentences) unless more explanation is required.\n"
        "- If the retrieved context does not provide enough information, state that explicitly.\n"
        "- If the query is unclear, ask the user for clarification before providing an answer.\n\n"

        "### Important Guidelines:\n"
        "- Do NOT make up information that is not in the retrieved context.\n"
        "- If the data seems incomplete, inform the user rather than guessing.\n"
        "- Maintain a professional and informative tone suitable for any dataset.\n\n"

        "### Retrieved Context:\n"
        "{context}\n\n"

        "Now, based on this retrieved context, answer the following user query."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

def initialize_embeddings():
    """
    Initializes and returns an instance of OpenAIEmbeddings.

    Parameters:
    - None

    Returns:
    - OpenAIEmbeddings: An instance of the OpenAIEmbeddings used to compute text embeddings.
    """
    return OpenAIEmbeddings()

# Main function
def main():
    doc_path = r"D:\\shubham\\LLM\\generative_AI\\data\\policy_doc.pdf"
    load_env_variables()
    llm = initialize_llm()
    embeddings = initialize_embeddings()
    docs = load_documents(doc_path)
    splitted_docs = chunking(docs)
    vector_store = store_document_vectorstore(embeddings, splitted_docs)
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    rag_chain = create_rag_chain(llm, retriever)
    
    while True:
        query = input('Query: ')
        answer = rag_chain.invoke({"input": query})
        print(answer['answer'])

if __name__ == "__main__":
    main()
