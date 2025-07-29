from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools.ddg_search.tool import DDGInput
from langgraph.prebuilt import create_react_agent
from IPython.display import Image, display
from ddgs import DDGS
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

# Lets explore Duckduckgo search tools information
search = DuckDuckGoSearchRun(api_wrapper_kwargs={"backend": "api"})
print(f"Tool's name = {search.name}")
print(f"Tool's description = {search.description}")
print(f"Tool's arg schema = {search.args_schema}")

print(DDGInput.model_fields)

query = "What is the weather of Delhi tomorrow?"
# search_input = DDGInput(query=query)
# result = search.invoke(search_input.model_dump())
# print(result)

with DDGS() as ddgs:
    results = [r for r in ddgs.text(query, max_results=1)]
    print(results)
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']} - {r['href']}")

result = llm.invoke([("system", "Always use a duckduckgo_search tool for queries that require a fresh information"),
                      ("user", query)], tools=[search])
print(result)
print(result.tool_calls[0])

# Create a ReACT agent with this tool
agent = create_react_agent(
    model=llm,
    tools=[search],
    prompt="Always use a duckduckgo_search tool for queries that require a fresh information"
)

# display(Image(agent.get_graph().draw_mermaid_png()))
# with open("react_workflow.png",'wb') as f:
#     f.write(agent.get_graph().draw_mermaid_png())

for event in agent.stream({"messages": [("user", query)]}):
  messages = event.get("agent", event.get("tools", {})).get("messages", [])
  for m in messages:
     m.pretty_print()