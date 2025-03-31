import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables from .env file
def load_env_variables():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Initialize LLM model
def initialize_llm(model_name="gpt-4o-mini"):
    return ChatOpenAI(model=model_name)

# Load and split documents from CSV
def load_documents(csv_path):
    loader = CSVLoader(file_path=csv_path)
    return loader.load_and_split()

# Initialize FAISS vector store
def initialize_vector_store(embeddings, docs):
    index = faiss.IndexFlatL2(len(embeddings.embed_query(" ")))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    print('Adding data to vector store...')
    vector_store.add_documents(documents=docs)
    return vector_store

# Create retrieval-augmented generation (RAG) chain
def create_rag_chain(llm, retriever):
    system_prompt = (
        "You are an AI assistant designed to answer user queries based on the provided retrieved context. "
        "The information comes from a CSV file, which may contain various types of structured data. "
        "Use only the retrieved context to generate responses. Do not use prior knowledge.\n\n"

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

# Main function
def main():
    load_env_variables()
    llm = initialize_llm()
    csv_path = r"D:\\shubham\\LLM\\generative_AI\\data\\customerSupport.csv"
    docs = load_documents(csv_path)
    embeddings = OpenAIEmbeddings()
    vector_store = initialize_vector_store(embeddings, docs)
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    rag_chain = create_rag_chain(llm, retriever)
    
    while True:
        query = input('Query: ')
        answer = rag_chain.invoke({"input": query})
        print(answer['answer'])

if __name__ == "__main__":
    main()
