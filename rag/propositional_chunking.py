"""
Proposition Chunking System

This script breaks down a text document into concise, factual propositions for granular 
information retrieval. It evaluates the effectiveness of proposition-based retrieval 
compared to larger document chunks.

Key Steps:
1. **Load Environment Variables** - Ensures access to necessary resources (e.g., API keys).
2. **Document Chunking** - Splits input text into smaller chunks using RecursiveCharacterTextSplitter.
3. **Proposition Generation** - Uses LLM ("llama-3.1-70b-versatile") to generate self-contained facts.
4. **Quality Check** - Evaluates propositions on accuracy, clarity, completeness, and conciseness.
5. **Embedding & Storage** - Stores validated propositions as embeddings for efficient retrieval.
6. **Retrieval & Comparison** - Compares proposition-based retrieval with larger document chunks.

The goal is to enhance precision in information retrieval by leveraging fine-grained 
proposition-based indexing.
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from pydantic import BaseModel, Field

# Load environment variables from .env file
def load_env_variables():
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Initialize LLM model
def initialize_llm(model_name="gpt-4o-mini"):
    return ChatOpenAI(model=model_name)

# Split text into smaller chunks for better processing
def split_text(docs_list, chunk_size=200, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs_list)

# Assign chunk IDs to split documents
def assign_chunk_ids(doc_splits):
    for i, doc in enumerate(doc_splits):
        doc.metadata['chunk_id'] = i + 1
    return doc_splits

# Define data model for proposition generation
class GeneratePropositions(BaseModel):
    """List of all the propositions in a given document"""
    propositions: List[str] = Field(
        description="List of propositions (factual, self-contained, and concise information)"
    )

# Generate structured propositions using LLM
def generate_propositions(doc_splits, llm, prompt):
    structured_llm = llm.with_structured_output(GeneratePropositions)
    proposition_generator = prompt | structured_llm
    propositions = []
    
    for i, doc in enumerate(doc_splits):
        response = proposition_generator.invoke({"document": doc.page_content})
        for proposition in response.propositions:
            propositions.append(Document(
                page_content=proposition, 
                metadata={"chunk_id": i+1}
            ))
    return propositions

# Define data model for proposition evaluation
class GradePropositions(BaseModel):
    """Grade a given proposition on accuracy, clarity, completeness, and conciseness"""
    accuracy: int = Field(description="Rate from 1-10 based on how well the proposition reflects the original text.")
    clarity: int = Field(description="Rate from 1-10 based on how easy it is to understand the proposition without additional context.")
    completeness: int = Field(description="Rate from 1-10 based on whether the proposition includes necessary details (e.g., dates, qualifiers).")
    conciseness: int = Field(description="Rate from 1-10 based on whether the proposition is concise without losing important information.")

# Evaluate generated propositions
def evaluate_propositions(propositions, doc_splits, llm, evaluation_prompt):
    structured_llm = llm.with_structured_output(GradePropositions)
    proposition_evaluator = evaluation_prompt | structured_llm
    
    thresholds = {"accuracy": 7, "clarity": 7, "completeness": 7, "conciseness": 7}
    evaluated_propositions = []
    
    for idx, proposition in enumerate(propositions):
        response = proposition_evaluator.invoke({
            "proposition": proposition.page_content,
            "original_text": doc_splits[proposition.metadata['chunk_id'] - 1].page_content
        })
        scores = {"accuracy": response.accuracy, "clarity": response.clarity, "completeness": response.completeness, "conciseness": response.conciseness}
        
        if all(scores[cat] >= thresholds[cat] for cat in thresholds):
            evaluated_propositions.append(proposition)
        else:
            print(f"{idx+1}) Proposition: {proposition.page_content} \n Scores: {scores}")
            print("Fail")
    return evaluated_propositions

# Create FAISS vector store and retriever
def create_vector_store(documents, embedding_model):
    vectorstore = FAISS.from_documents(documents, embedding_model)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={'k': 4})

# Retrieve relevant documents based on query
def retrieve_documents(query, retriever):
    return retriever.invoke(query)

    
sample_content = """
In 1905, Albert Einstein, a young physicist working at the Swiss Patent Office, published four groundbreaking papers that would change the course of modern physics. One of these papers introduced the special theory of relativity, which proposed that the laws of physics are the same for all non-accelerating observers and that the speed of light remains constant regardless of the observer’s motion.
A key outcome of special relativity was the famous equation E=mc², which revealed that energy (E) and mass (m) are interchangeable, connected by the speed of light squared (c²). This discovery laid the foundation for nuclear energy and particle physics.
Einstein did not stop there. In 1915, he expanded on his ideas and introduced the general theory of relativity, which fundamentally changed our understanding of gravity. Instead of viewing gravity as a force, as described by Isaac Newton, Einstein proposed that massive objects warp the fabric of spacetime, causing smaller objects to move along curved paths. This prediction was confirmed in 1919, when British astronomer Sir Arthur Eddington observed the bending of starlight during a solar eclipse, providing the first experimental proof of general relativity.
Einstein's theories not only reshaped physics but also had profound implications for cosmology. His equations predicted that the universe was not static but could expand or contract. In the 1920s, astronomer Edwin Hubble provided observational evidence that the universe was expanding, further validating Einstein’s work.
Throughout his life, Einstein continued to develop his theories and advocate for scientific progress. In 1939, he co-signed a letter to U.S. President Franklin D. Roosevelt, warning that Nazi Germany might be developing atomic weapons, which led to the creation of the Manhattan Project. Although Einstein did not work directly on the project, his equation E=mc² was crucial in understanding nuclear fission.
Einstein spent his later years at the Institute for Advanced Study in Princeton, where he worked on a unified field theory that aimed to merge gravity and electromagnetism. Though he was unable to complete this work, his contributions to science remain among the most influential in history.
"""
# Define prompts for proposition generation and evaluation
system_prompt = """
Break down the provided text into simple, factual, and self-contained propositions. Ensure each proposition meets the following criteria:
**Express a Single Fact**: Each proposition must convey only one specific fact, claim, or piece of information. Avoid combining multiple ideas.
**Be Self-Sufficient**: The proposition should be understandable without additional context from the original text.
**Use Explicit References**: Use full names, locations, and entities instead of pronouns or ambiguous terms.
**Retain Essential Details**: Include relevant dates, locations, numbers, and qualifiers where applicable to ensure accuracy.
**Maintain a Clear Structure**: Each proposition should follow a single subject-predicate-object relationship without conjunctions or multiple clauses.
**Avoid Subjective or Inferred Information**: Extract only explicitly stated facts without assumptions or personal interpretations.
"""
evaluation_prompt_template = """
Please evaluate the following proposition based on the criteria below:

- **Accuracy (1-10)**: How well does the proposition reflect the original text? Is it factually correct?
- **Clarity (1-10)**: Is the proposition easy to understand on its own?
- **Completeness (1-10)**: Does the proposition contain necessary details (e.g., names, dates, qualifiers) to be fully meaningful?
- **Conciseness (1-10)**: Is the proposition free from unnecessary details while preserving key information?

If a score is below 7, briefly justify why.

### Example:
Original Text: The Eiffel Tower, completed in 1889, is one of the most visited landmarks in the world and stands at 330 meters tall.

**Proposition 1:** The Eiffel Tower was completed in 1889.  
Evaluation: "accuracy": 10, "clarity": 10, "completeness": 10, "conciseness": 10

**Proposition 2:** The Eiffel Tower is one of the most visited landmarks in the world.  
Evaluation: "accuracy": 10, "clarity": 10, "completeness": 9, "conciseness": 10
Justification: The proposition is missing the year of completion, which could add more context.

**Proposition 3:** The Eiffel Tower is a very tall building.  
Evaluation: "accuracy": 5, "clarity": 8, "completeness": 5, "conciseness": 10  
Justification: The phrase "very tall" is vague, and the specific height is missing.

### Format:
**Proposition:** "{proposition}"  
**Original Text:** "{original_text}"
"""
# Main execution flow
def main():
    load_env_variables()
    llm = initialize_llm()
    embedding_model = OpenAIEmbeddings()

    docs_list = [Document(page_content=sample_content)]
    doc_splits = assign_chunk_ids(split_text(docs_list))

    proposition_examples = [
        {
            "document": 
                "In 2004, Mark Zuckerberg and his college roommates launched Facebook, which became one of the world's largest social media platforms.",
            "propositions": 
                "['Mark Zuckerberg co-founded Facebook in 2004.', "
                "'Facebook was launched by Mark Zuckerberg and his college roommates.', "
                "'Facebook became one of the largest social media platforms.', "
                "'Facebook was founded while Zuckerberg was in college.']"
        },
    ]

    example_proposition_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{document}"),
            ("ai", "{propositions}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt = example_proposition_prompt,
        examples = proposition_examples,
    )
    proposition_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        ("human", "{document}"),
    ])
    evaluation_prompt = ChatPromptTemplate.from_messages([
        ("system", evaluation_prompt_template),
        ("human", "{proposition}, {original_text}"),
    ])
    
    propositions = generate_propositions(doc_splits, llm, proposition_prompt)
    evaluated_propositions = evaluate_propositions(propositions, doc_splits, llm, evaluation_prompt)
    
    # Create vector stores
    retriever_propositions = create_vector_store(evaluated_propositions, embedding_model)
    retriever_larger = create_vector_store(doc_splits, embedding_model)
    while True:
        query = input("Query: ")
        # query = "What key scientific discovery did Albert Einstein make in 1905, and why was it significant?"
        response_proposition = retrieve_documents(query, retriever_propositions)
        response_larger = retrieve_documents(query, retriever_larger)
        
        # Print results
        print("Proposition-Based Retrieval: ")
        for i, r in enumerate(response_proposition):
            print(f"{i+1}) Content: {r.page_content} --- Chunk_id: {r.metadata['chunk_id']}")
        
        print("Larger Chunk-Based Retrieval:")
        for i, r in enumerate(response_larger):
            print(f"{i+1}) Content: {r.page_content} --- Chunk_id: {r.metadata['chunk_id']}")

if __name__ == "__main__":
    main()
