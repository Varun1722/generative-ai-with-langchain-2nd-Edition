from langchain_core.runnables import RunnableLambda
from langchain_core.tools import convert_runnable_to_tool, BaseTool
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import math
import numexpr as ne
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-pro")

class CalculatorArgs(BaseModel):
    expression: str = Field(description="Mathematical expression to be evaluated")

math_constants = {"pi": math.pi, "i": 1j, "e": math.exp}
config = {"configurable": {"math_constants": math_constants}}

def calculator(state: CalculatorArgs, config: RunnableConfig) -> str:
    expression = state["expression"]
    math_constants = config["configurable"].get("math_constants", {})
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
    args_schema=CalculatorArgs,
    arg_types={"expression": "str"},
)

assert isinstance(calculator_tool, BaseTool)
print(f"Tool name: {calculator_tool.name}")
print(f"Tool name: {calculator_tool.description}")
print(f"Args schema: {calculator_tool.args_schema.model_json_schema()}")


tool_call = llm.invoke("How much is (2+3i)**2", tools=[calculator_tool]).tool_calls[0]
print(tool_call)

# class object 
res = calculator_tool.invoke(tool_call["args"], config=config)
print(res)