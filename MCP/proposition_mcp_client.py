
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import asyncio
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
model = ChatOpenAI(model="gpt-4o-mini")

server_params = StdioServerParameters(
    command="python",
    args=["D:\shubham\LLM\generative_AI\MCP\proposition_mcp_server.py"],
)
content = """
             In 1905, Albert Einstein, a young physicist working at the Swiss Patent Office, published four groundbreaking papers that would change the course of modern physics. One of these papers introduced the special theory of relativity, which proposed that the laws of physics are the same for all non-accelerating observers and that the speed of light remains constant regardless of the observer’s motion.
             A key outcome of special relativity was the famous equation E=mc², which revealed that energy (E) and mass (m) are interchangeable, connected by the speed of light squared (c²). This discovery laid the foundation for nuclear energy and particle physics.
             Einstein did not stop there. In 1915, he expanded on his ideas and introduced the general theory of relativity, which fundamentally changed our understanding of gravity. Instead of viewing gravity as a force, as described by Isaac Newton, Einstein proposed that massive objects warp the fabric of spacetime, causing smaller objects to move along curved paths. This prediction was confirmed in 1919, when British astronomer Sir Arthur Eddington observed the bending of starlight during a solar eclipse, providing the first experimental proof of general relativity.
             Einstein's theories not only reshaped physics but also had profound implications for cosmology. His equations predicted that the universe was not static but could expand or contract. In the 1920s, astronomer Edwin Hubble provided observational evidence that the universe was expanding, further validating Einstein’s work.
             Throughout his life, Einstein continued to develop his theories and advocate for scientific progress. In 1939, he co-signed a letter to U.S. President Franklin D. Roosevelt, warning that Nazi Germany might be developing atomic weapons, which led to the creation of the Manhattan Project. Although Einstein did not work directly on the project, his equation E=mc² was crucial in understanding nuclear fission.
             Einstein spent his later years at the Institute for Advanced Study in Princeton, where he worked on a unified field theory that aimed to merge gravity and electromagnetism. Though he was unable to complete this work, his contributions to science remain among the most influential in history.
            """
async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)
            # Create and run the agent
            print('Calling Agent For Proposition Chunking')
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": f"""You are an expert at extracting propositions. First verify if the tool extract_propositions is needed. If it is needed, call the tool.
extract propositions from following text: {content}"""})
            for m in agent_response['messages']:
                m.pretty_print()

if __name__ == '__main__':
    asyncio.run(run_agent())