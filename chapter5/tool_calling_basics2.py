from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17")

question1 = "How old is the India's Prime Minister?"
question2 = "How many states are there in India?"

search_result = (
"Narendra Modi › Age 74 years September 17, 1950\n"
"Narendra Modi is the 14th Prime Minister of India, currently serving his third consecutive term since June 9, 2024."
" Born in Vadnagar, Gujarat, he rose from humble beginnings as a tea-seller to become a key leader of the Bharatiya Janata Party (BJP)."
" Before becoming PM, he served as the Chief Minister of Gujarat from 2001 to 2014. Known for policies like Digital India, Swachh Bharat, and GST, Modi is both praised for infrastructure growth and criticized for increasing polarization. Wikipedia"
)

query = "age of India's PM"
raw_template = (
  "You have access to search engine that provides you an "
  "information about fresh events and news given the query. "
  "Given the question, decide whether you need an additional "
  "information from the search engine (reply with 'SEARCH: "
   "<generated query>' or you know enough to answer the user "
   "then reply with 'RESPONSE <final response>').\n"
   "Today is {date}."
   "Now, act to answer a user question and "
   "take into account your previous actions:\n"
   "HUMAN: {question}\n"
   "AI: SEARCH: {query}\n"
   "RESPONSE FROM SEARCH: {search_result}\n"
)

prompt = PromptTemplate.from_template(raw_template)

chain = prompt | llm
# response = chain.invoke(question1)  If enough info not present It results in SEARCH: India's Prime Minister age
response = chain.invoke({"question":question1,"query":query,"search_result":search_result,"date":"July 2025"})
print(response.content)
