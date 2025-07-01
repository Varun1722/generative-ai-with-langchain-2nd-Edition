from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

template = PromptTemplate.from_template("Tell me a joke about {topic}")
output = StrOutputParser()

chain = template|model|output
text = template.format(topic="physics")
result = chain.invoke(text)
# result = chain.invoke({"topic":"formula 1"})
print(result)