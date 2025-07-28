"""Loading LLMs and Embeddings."""
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite-preview-06-17",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

store = LocalFileStore("./cache/")

underlying_embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
)
# Avoiding unnecessary costs by caching the embeddings.
EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)