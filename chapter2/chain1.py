from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

template = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant"),
        ("user","{prompt}")
    ]
)

chain = template | model

response = chain.invoke({"prompt":"Who is the CEO of Google"})
print(response.content)
