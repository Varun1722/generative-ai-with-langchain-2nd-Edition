from langchain_openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

openai_llm = OpenAI()
gemini_pro = GoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")
# gemini_pro = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

response = gemini_pro.invoke("Tell me a funny joke about llms")
print(response)
