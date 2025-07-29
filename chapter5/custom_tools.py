import numexpr as ne
import math
from langchain_core.tools import tool
from langchain_core.tools import BaseTool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-pro")


math_constants = {"pi": math.pi, "i": 1j, "e": math.exp},
c = ne.evaluate(("2+2"), local_dict=math_constants)



# We define our calculator as a Python function and wrap it with a built-in @tool decorator to create a tool from it

@tool
def calculator(expression: str) -> str:
    """Calculates a single mathematical expression, incl. complex numbers.

    Always add * to operations, examples:
      73i -> 73*i
      7pi**2 -> 7*pi**2 
    """
    math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
    result = ne.evaluate(expression.strip(), local_dict=math_constants)
    return str(result)

assert isinstance(calculator, BaseTool)
print(f"Tool name: {calculator.name}")
print(f"Tool name: {calculator.description}")
print(f"Tool schema: {calculator.args_schema.model_json_schema()}")

query = "How much is 2+3i squared?"

agent = create_react_agent(llm, [calculator])

for event in agent.stream({"messages": [("user", query)]}, stream_mode="values"):
    event["messages"][-1].pretty_print()