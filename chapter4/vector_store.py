from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embedding_model = OpenAIEmbeddings()
vector_store = Chroma(embedding_function=embedding_model)

