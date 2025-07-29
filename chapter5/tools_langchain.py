from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,ToolMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17")
question = "What is the age of PM of India?"

search_result = (
"Narendra Modi › Age 74 years September 17, 1950\n"
"Narendra Modi is the 14th Prime Minister of India, currently serving his third consecutive term since June 9, 2024."
" Born in Vadnagar, Gujarat, he rose from humble beginnings as a tea-seller to become a key leader of the Bharatiya Janata Party (BJP)."
" Before becoming PM, he served as the Chief Minister of Gujarat from 2001 to 2014. Known for policies like Digital India, Swachh Bharat, and GST, Modi is both praised for infrastructure growth and criticized for increasing polarization. Wikipedia"
)

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
# Modern LLMs hide the need to construct a prompt from the user, you can define your tools as schemas instead and pass them as a separate argument
step1 = llm.invoke(question, tools=[search_tool])

# output will have tool_calls
# print(step1.tool_calls)
# [{'name': 'google_search', 'args': {'query': 'age of PM of India'}, 'id': '2219b405-cb6b-4d1b-9ea4-378157fe5854', 'type': 'tool_call'}]

# We can pass the tool calling result back to an LLM as a special ToolMessage message
tool_result = ToolMessage(content=search_result,tool_call_id=step1.tool_calls[0]['id'])
step2 = llm.invoke(
    [HumanMessage(content=question), step1, tool_result],
    tools=[search_tool]
)
assert len(step2.tool_calls) == 0 #to make sure that step2 doesn't have any tool calls 

# print(step2.content)
""" For the convinience, we can also bind tools to an LLM so that they would be 
auto-added to arguments on every invocation"""

llm_with_tools = llm.bind(tools=[search_tool])
result = llm_with_tools.invoke(question)
print(result)