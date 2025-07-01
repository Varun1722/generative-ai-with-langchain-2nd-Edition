from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite-preview-06-17")

question_template = PromptTemplate.from_template("Answer this question in one word: {question}")

ques_with_context = PromptTemplate.from_template(" Context: {context}" \
"answer the question in one word: {question}")

prompt_text = question_template.format(question = "What is the capital of France")

result = model.invoke(prompt_text)
print(result.content)