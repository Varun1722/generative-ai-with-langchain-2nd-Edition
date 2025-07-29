from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-pro")
python_repl = PythonREPL()
python_repl.run("print(2**4)")

code_interpreter_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

query_strawberry = "How many r are in the word strawberry?"
# print(llm.invoke(query_strawberry).content)

# But we can see that ReACT agent easily solved this problem by invoking an external tool
agent = create_react_agent(
  model=llm,
  tools=[code_interpreter_tool])

for event in agent.stream({"messages": [("user", query_strawberry)]}):
  messages = event.get("agent", event.get("tools", {})).get("messages", [])
  for m in messages:
     m.pretty_print()