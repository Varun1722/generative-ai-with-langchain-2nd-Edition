from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()

''' 
For a hybrid approach we will need two methods:
1. Semantic search
2. Lexical search
'''

docs = []
# setup semantic retriever
semantic_retriever = vector_store.as_retriever(search_kwargs={'k':5})

#setup lexical retriever
lexical_retriever = BM25Retriever.from_documents(docs)
lexical_retriever.k = 5

# combining 
hybrid_retriever = EnsembleRetriever(
    retrievers=[semantic_retriever,lexical_retriever],
    weights=[0.75,0.25]  
)

results = hybrid_retriever.get_relevant_documents("how AI is changing the world")