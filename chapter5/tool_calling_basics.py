from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash-lite-preview-06-17")

question1 = "How old is the India's Prime Minister?"
question2 = "How many states are there in India?"

raw_template = (
  "You have access to search engine that provides you an "
  "information about fresh events and news given the query. "
  "Given the question, decide whether you need an additional "
  "information from the search engine (reply with 'SEARCH: "
   "<generated query>' or you know enough to answer the user "
   "then reply with 'RESPONSE <final response>').\n"
   "Do not make any assumptions on recent events or things that can change."
   "Now, act to answer a user question:\n{QUESTION}"
)

prompt = PromptTemplate.from_template(raw_template)

chain = prompt | llm
# response = chain.invoke(question1)  If enough info not present It results in SEARCH: India's Prime Minister age
response = chain.invoke(question2) # RESPONSE There are 28 states in India.

print(response.content)
