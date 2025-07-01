from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")
messages = [
    SystemMessage(content="You are a helpful assistant"),
    HumanMessage(content="What is the capital of USA")
]

result = model.invoke(messages)
print(result.content)