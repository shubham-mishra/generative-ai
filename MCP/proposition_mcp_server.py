from mcp.server.fastmcp import FastMCP
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List

# Initialize MCP
mcp = FastMCP("PropositionMCP")

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# Initialize LLM and embedding model
llm = ChatOpenAI(model="gpt-4o-mini")
embedding_model = OpenAIEmbeddings()

# System Prompt
SYSTEM_PROMPT = """
Break down the provided text into simple, factual, and self-contained propositions. Ensure each proposition meets the following criteria:
**Express a Single Fact**: Each proposition must convey only one specific fact, claim, or piece of information. Avoid combining multiple ideas.
**Be Self-Sufficient**: The proposition should be understandable without additional context from the original text.
**Use Explicit References**: Use full names, locations, and entities instead of pronouns or ambiguous terms.
**Retain Essential Details**: Include relevant dates, locations, numbers, and qualifiers where applicable to ensure accuracy.
**Maintain a Clear Structure**: Each proposition should follow a single subject-predicate-object relationship without conjunctions or multiple clauses.
**Avoid Subjective or Inferred Information**: Extract only explicitly stated facts without assumptions or personal interpretations.
"""

# Few-shot examples for better generation quality
EXAMPLES = [
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

# Few-shot prompt setup
EXAMPLE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("human", "{document}"),
        ("ai", "{propositions}"),
    ]
)
FEW_SHOT_PROMPT = FewShotChatMessagePromptTemplate(
        example_prompt = EXAMPLE_PROMPT,
        examples = EXAMPLES,
    )
PROPOSITION_PROMPT = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        FEW_SHOT_PROMPT,
        ("human", "{document}"),
])
# Define data model for proposition generation
class GeneratePropositions(BaseModel):
    """List of all the propositions in a given document"""
    propositions: List[str] = Field(
        description="List of propositions (factual, self-contained, and concise information)"
    )
    
# Split text into smaller chunks for better processing
def split_text(docs_list, chunk_size=200, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs_list)

@mcp.tool()
def extract_propositions(content: str) -> List[str]:
    """Extracting Propositions From Provided Content"""
    docs_list = [Document(page_content=content)]
    docs_list = split_text(docs_list)
    structured_llm = llm.with_structured_output(GeneratePropositions)
    proposition_generator = PROPOSITION_PROMPT | structured_llm
    propositions = []
    for i, doc in enumerate(docs_list):
        response = proposition_generator.invoke({"document": doc.page_content})
        for proposition in response.propositions:
            propositions.append(proposition)
    return propositions if propositions else ["No valid propositions generated."]

if __name__ == "__main__":
    mcp.run(transport="stdio")