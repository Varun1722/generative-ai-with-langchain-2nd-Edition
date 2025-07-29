from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
import numexpr as ne
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-pro")

def calculator(expression: str) -> str:
    """Calculates a single mathematical expression, incl. complex numbers."""
    return str(ne.evaluate(expression.strip(), local_dict={}))

calculator_tool = StructuredTool.from_function(
    func=calculator,
    handle_tool_error=True
)

agent = create_react_agent(
    llm, [calculator_tool]
)

for event in agent.stream({"messages": [("user", "How much is (2+3i)^2")]}, stream_mode="values"):
    event['messages'][-1].pretty_print()