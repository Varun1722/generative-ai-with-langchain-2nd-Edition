from langchain_core.runnables import RunnableLambda
from langchain_core.tools import convert_runnable_to_tool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import math
import numexpr as ne
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-pro")

def calculator(expression: str) -> str:
    math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
    result = ne.evaluate(expression.strip(), local_dict=math_constants)
    return str(result)

calculator_with_retry = RunnableLambda(calculator).with_retry(
    wait_exponential_jitter=True,
    stop_after_attempt=3,
)

calculator_tool = convert_runnable_to_tool(
    calculator_with_retry,
    name="calculator",
    description=(
        "Calculates a single mathematical expression, incl. complex numbers."
        "'\nAlways add * to operations, examples:\n73i -> 73*i\n"
        "7pi**2 -> 7*pi**2"
    ),
    arg_types={"expression": "str"},
)

response = calculator_tool.invoke({"expression": "(2+3*i)**2"})
print(response)

llm_response = llm.invoke("How much is (2+3i)**2", tools=[calculator_tool]).tool_calls[0]
print(llm_response)