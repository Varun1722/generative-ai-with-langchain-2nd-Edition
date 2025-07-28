from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings_model = OpenAIEmbeddings()

text1 = "The cat sat on the mat"
text2 = "A feline rested on the carpet" 
text3 = "Python is a programming language"

embeddings = embeddings_model.embed_documents(texts=[text1,text2,text3])

embeddings_for_query = embeddings_model.embed_query("Testing embedding with just one query")
# print(embeddings_for_query)

embedding1 = embeddings[0] # Embedding for "The cat sat on the mat" 
embedding2 = embeddings[1] # Embedding for "A feline rested on the carpet"

print(f'Number of docs: {len(embeddings)}') 

# 1536 for OPENAI Embeddings
print(f'Dimension of embedding1: {len(embedding1)}')
print(f'Dimension of embedding2: {len(embedding2)}')


