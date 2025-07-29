import math
import numexpr as ne
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent, ToolNode,tools_condition
from langgraph.graph import MessagesState, StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()


search = DuckDuckGoSearchRun()

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


llm_with_tools = ChatGoogleGenerativeAI(model="gemini-2.5-pro").bind_tools([search, calculator])

# We will build our own ReACT agent, but this time we will use a ToolNode to execute tool calls automatically
def invoke_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("invoke_llm", invoke_llm)
builder.add_node("tools",ToolNode([search,calculator]))

builder.add_edge(START,"invoke_llm")
builder.add_conditional_edges("invoke_llm",tools_condition)
builder.add_edge("tools","invoke_llm")
builder.add_edge("tools",END)
builder.add_edge("invoke_llm",END)
graph = builder.compile()

for e in graph.stream({"messages":("human","How much is 5*2")}):
    print(e)