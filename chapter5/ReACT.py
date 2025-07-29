from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START, END
import math
from dotenv import load_dotenv

load_dotenv()

# llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite")
llm = ChatOpenAI(model = "gpt-4o-mini-2024-07-18")

# Define Mock tools for now 
def mocked_google_search(query: str) -> str:
  print(f"CALLED GOOGLE_SEARCH with query={query}")
  return "Narendra Modi is the PM of INDIA and he's 74 years old"

def mocked_calculator(expression: str) -> float:
  print(f"CALLED CALCULATOR with expression={expression}")
  if "sqrt" in expression:
    return math.sqrt(74*56)
  return 74*56

# Defining tools schema
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Computes mathematical expressions",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "title": "expression",
                    "description": "A mathematical expression to be evaluated by a calculator"
                }
            },
            "required": ["expression"]
        }
    }
}

search_tool = {
    "type": "function",
    "function": {
        "name": "google_search",
        "description": "Returns about common facts, fresh events and news from Google Search engine based on a query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "title": "search_query",
                    "description": "Search query to be sent to the search engine"
                }
            },
            "required": ["query"]
        }
    }
}

system_prompt = (
    "Always use a calculator for mathematical computations, and use Google Search "
    "for information about common facts, fresh events and news. Do not assume anything, keep in "
    "mind that things are changing and always "
    "check yourself with external sources if possible."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
])

llm_with_tools = prompt | llm.bind(tools=[search_tool,calculator_tool])

""" 
Now we will define methods for our application
1. Function to invoke LLMs
2. function that invokes tools and returns tool-calling results 
(by appending ToolMessages to the list of messages in the state)
3. function that will determine whether the orchestrator should continue calling tools
 or whether it can return the result to the user
"""
def invoke_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


def call_tools(state: MessagesState):
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls

    new_messages = []

    for tool_call in tool_calls:
      if tool_call["name"] == "google_search":
        tool_result = mocked_google_search(**tool_call["args"])
        new_messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
      elif tool_call["name"] == "calculator":
        tool_result = mocked_calculator(**tool_call["args"])
        new_messages.append(ToolMessage(content=tool_result, tool_call_id=tool_call["id"]))
      else:
        raise ValueError(f"Tool {tool_call['name']} is not defined!")
    return {"messages": new_messages}


def should_run_tools(state: MessagesState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
      return "call_tools"
    return END

# Building the graph and flow of application
builder = StateGraph(MessagesState)
builder.add_node("invoke_llm", invoke_llm)
builder.add_node("call_tools", call_tools)

builder.add_edge(START, "invoke_llm")
builder.add_conditional_edges("invoke_llm", should_run_tools)
builder.add_edge("call_tools", "invoke_llm")
graph = builder.compile()

question = "What is a square root of the current India's PM age multiplied by 56?"

result = graph.invoke({"messages": [HumanMessage(content=question)]})
print(result["messages"][-1].content)