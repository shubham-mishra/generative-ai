import os
from pathlib import Path
import warnings

from dotenv import find_dotenv, load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import chain
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import umap

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_SEED = 42  

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

def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: int | None = None,
    metric: str = "cosine",
) -> np.ndarray:
    """
    Perform global dimensionality reduction on the embeddings using UMAP.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - n_neighbors: Optional; the number of neighbors to consider for each point.
                   If not provided, it defaults to the square root of the number of embeddings.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    """
    Perform local dimensionality reduction on the embeddings using UMAP, typically after global clustering.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for the reduced space.
    - num_neighbors: The number of neighbors to consider for each point.
    - metric: The distance metric to use for UMAP.

    Returns:
    - A numpy array of the embeddings reduced to the specified dimensionality.
    """
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    """
    Determine the optimal number of clusters using the Bayesian Information Criterion (BIC) with a Gaussian Mixture Model.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - max_clusters: The maximum number of clusters to consider.
    - random_state: Seed for reproducibility.

    Returns:
    - An integer representing the optimal number of clusters found.
    """
    max_clusters = min(max_clusters, len(embeddings))
    n_clusters = np.arange(1, max_clusters)
    bics = []
    
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings)
        bics.append(gm.bic(embeddings))
        
    return n_clusters[np.argmin(bics)]


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = RANDOM_SEED):
    """
    Cluster embeddings using a Gaussian Mixture Model (GMM) based on a probability threshold.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - threshold: The probability threshold for assigning an embedding to a cluster.
    - random_state: Seed for reproducibility.

    Returns:
    - A tuple containing the cluster labels and the number of clusters determined.
    """
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> list[np.ndarray]:
    """
    Perform clustering on the embeddings by first reducing their dimensionality globally, then clustering
    using a Gaussian Mixture Model, and finally performing local clustering within each global cluster.

    Parameters:
    - embeddings: The input embeddings as a numpy array.
    - dim: The target dimensionality for UMAP reduction.
    - threshold: The probability threshold for assigning an embedding to a cluster in GMM.

    Returns:
    - A list of numpy arrays, where each array contains the cluster IDs for each embedding.
    """
    if len(embeddings) <= dim + 1:
        # Avoid clustering when there's insufficient data
        return [np.array([0]) for _ in range(len(embeddings))]

    # Global dimensionality reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
    # Global clustering
    global_clusters, n_global_clusters = GMM_cluster(
        reduced_embeddings_global, threshold
    )

    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    # Iterate through each global cluster to perform local clustering
    for i in range(n_global_clusters):
        # Extract embeddings belonging to the current global cluster
        global_cluster_embeddings_ = embeddings[
            np.array([i in gc for gc in global_clusters])
        ]

        if len(global_cluster_embeddings_) == 0:
            continue
        if len(global_cluster_embeddings_) <= dim + 1:
            # Handle small clusters with direct assignment
            local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
            n_local_clusters = 1
        else:
            # Local dimensionality reduction and clustering
            reduced_embeddings_local = local_cluster_embeddings(
                global_cluster_embeddings_, dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        # Assign local cluster IDs, adjusting for total clusters already processed
        for j in range(n_local_clusters):
            local_cluster_embeddings_ = global_cluster_embeddings_[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == local_cluster_embeddings_[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )

        total_clusters += n_local_clusters

    return all_local_clusters


def embed(texts):
    """
    Generate embeddings for a list of text documents.

    This function assumes the existence of an `embd` object with a method `embed_documents`
    that takes a list of texts and returns their embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be embedded.

    Returns:
    - numpy.ndarray: An array of embeddings for the given text documents.
    """
    text_embeddings = embeddings.embed_documents(texts)
    text_embeddings_np = np.array(text_embeddings)
    return text_embeddings_np


def embed_cluster_texts(texts):
    """
    Embeds a list of texts and clusters them, returning a DataFrame with texts, their embeddings, and cluster labels.

    This function combines embedding generation and clustering into a single step. It assumes the existence
    of a previously defined `perform_clustering` function that performs clustering on the embeddings.

    Parameters:
    - texts: List[str], a list of text documents to be processed.

    Returns:
    - pandas.DataFrame: A DataFrame containing the original texts, their embeddings, and the assigned cluster labels.
    """
    text_embeddings_np = embed(texts)  # Generate embeddings
    cluster_labels = perform_clustering(
        text_embeddings_np, 10, 0.1
    )  # Perform clustering on the embeddings
    df = pd.DataFrame()  # Initialize a DataFrame to store the results
    df["text"] = texts  # Store original texts
    df["embd"] = list(text_embeddings_np)  # Store embeddings as a list in the DataFrame
    df["cluster"] = cluster_labels  # Store cluster labels
    return df


def format_texts(texts: list[str]) -> str:
    """
    Formats the text documents in a DataFrame into a single string.

    Parameters:
    - texts: List of texts to format.

    Returns:
    - A single string where all text documents are joined by a specific delimiter.
    """
    return "--- --- \n --- --- ".join(texts)


def embed_cluster_summarize_texts(
    texts: list[str], level: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Embeds, clusters, and summarizes a list of texts. This function first generates embeddings for the texts,
    clusters them based on similarity, expands the cluster assignments for easier processing, and then summarizes
    the content within each cluster.

    Parameters:
    - texts: A list of text documents to be processed.
    - level: An integer parameter that could define the depth or detail of processing.

    Returns:
    - Tuple containing two DataFrames:
      1. The first DataFrame (`df_clusters`) includes the original texts, their embeddings, and cluster assignments.
      2. The second DataFrame (`df_summary`) contains summaries for each cluster, the specified level of detail,
         and the cluster identifiers.
    """
    # Summarization
    prompt_template = """Here is a subset of Company policy document. 

    Give a detailed summary of the documentation provided.
    
    Documentation:
    {context}
    """
    
    @chain
    def summarize_cluster(texts):
        formatted_txt = format_texts(texts)
        prompt = prompt_template.format(
            context=formatted_txt
        )
        response = llm.invoke([
            HumanMessage(content=prompt)
        ])
        return response.content

    # Embed and cluster the texts, resulting in a DataFrame with 'text', 'embd', and 'cluster' columns
    df_clusters = embed_cluster_texts(texts)

    # Prepare to expand the DataFrame for easier manipulation of clusters
    expanded_list = []

    # Expand DataFrame entries to document-cluster pairings for straightforward processing
    for index, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append(
                {"text": row["text"], "embd": row["embd"], "cluster": cluster}
            )

    # Create a new DataFrame from the expanded list
    expanded_df = pd.DataFrame(expanded_list)

    # Retrieve unique cluster identifiers for processing
    all_clusters = expanded_df["cluster"].unique().tolist()

    print(f"--Generated {len(all_clusters)} clusters--")

    summaries = summarize_cluster.batch(
        [
            expanded_df.loc[expanded_df["cluster"] == cluster_idx, "text"].tolist() 
            for cluster_idx in all_clusters
        ], 
        config={"max_concurrency": 10}
    )

    # Create a DataFrame to store summaries with their corresponding cluster and level
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary


def recursive_embed_cluster_summarize(
    texts: list[str], level: int = 1, n_levels: int = 3
) -> dict[int, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Recursively embeds, clusters, and summarizes texts up to a specified level or until
    the number of unique clusters becomes 1, storing the results at each level.

    Parameters:
    - texts: List[str], texts to be processed.
    - level: int, current recursion level (starts at 1).
    - n_levels: int, maximum depth of recursion.

    Returns:
    - Dict[int, Tuple[pd.DataFrame, pd.DataFrame]], a dictionary where keys are the recursion
      levels and values are tuples containing the clusters DataFrame and summaries DataFrame at that level.
    """
    results = {}  # Dictionary to store results at each level

    # Perform embedding, clustering, and summarization for the current level
    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)

    # Store the results of the current level
    results[level] = (df_clusters, df_summary)

    # Determine if further recursion is possible and meaningful
    unique_clusters = df_summary["cluster"].nunique()
    
    if level < n_levels and unique_clusters > 1:
        # Use summaries as the input texts for the next level of recursion
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )

        # Merge the results from the next level into the current results dictionary
        results.update(next_level_results)

    return results

def create_merged_doc_list(leaf_texts, cluster_summarizer_results):
    """
    Merges a list of base texts and cluster summarizer results into a single list of Document objects.

    This function creates Document objects for each text in the base list (`leaf_texts`) and then iterates
    through the provided cluster summarizer results, extracting summaries from each level. For each summary,
    it creates a Document with a metadata attribute indicating the level of summarization. The resulting list
    combines both the original texts and their corresponding summaries, making it ready for further processing.

    Parameters:
    - leaf_texts (list): A list of strings representing the base or leaf texts.
    - cluster_summarizer_results (dict): A dictionary where each key is a level (int) and the value is a tuple,
      with the second element being a DataFrame that contains a "summaries" column holding text summaries.

    Returns:
    - list: A list of Document objects that includes both the base texts and the summaries, each annotated
      with a metadata field "level" indicating its hierarchical level.
    """
    all_docs = [
        Document(
            page_content=text, 
            metadata={"level": 0}
        )
        for text in leaf_texts
    ]


    for level in sorted(cluster_summarizer_results.keys()):
        # Extract summaries from the current level's DataFrame
        docs = [
            Document(
                page_content=summary, 
                metadata={"level": level}
            )
            for summary in cluster_summarizer_results[level][1]["summaries"]
        ]
        # Extend all_texts with the summaries from the current level
        all_docs.extend(docs)
    return all_docs

if __name__ == "__main__":
    doc_path = r"D:\\shubham\\LLM\\generative_AI\\data\\policy_doc.pdf"

    load_env_variables()
    llm = initialize_llm()
    embeddings = initialize_embeddings()
    docs = load_documents(doc_path)
    splitted_docs = chunking(docs)

    #extracting texts from chunks and considering initial chunks as leaf nodes
    leaf_texts = [doc.page_content for doc in splitted_docs]

    #creating clusters and summarizing them - at each level of tree
    cluster_summarizer_results = recursive_embed_cluster_summarize(leaf_texts, level=1, n_levels=3)
    
    #appending leaf nodes and all nodes text to create a single list of documents.
    merged_docs = create_merged_doc_list(leaf_texts, cluster_summarizer_results)

    #storing data in  vector store
    vector_store = store_document_vectorstore(embeddings, merged_docs)
    
    #Query on stored document
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})
    rag_chain = create_rag_chain(llm, retriever)
    
    while True:
        query = input('Query: ')
        answer = rag_chain.invoke({"input": query})
        print(answer['answer'])