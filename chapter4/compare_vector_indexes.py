import numpy as np
import time
import faiss

# NOTE: Faiss is only supported in macOS/Linux
# create sample data - 5k vectors with 128 dimension
dimension = 128
num_of_vectors = 5000
vectors = np.random.random((num_of_vectors,dimension)).astype('float32')
query = np.random.random((1,dimension)).astype('float32')

# We will compare between Exact search(Brute Force Method) bs HNSW(Hierarchical Navigable Small World)
exact_index = faiss.IndexFlatL2(dimension)
exact_index.add(vectors)

hnsw_index = faiss.IndexHNSWFlat(dimension,32) #32 connections per node
hnsw_index.add(vectors)

# compare search results 
start_time = time.time()
exact_D, exact_I = exact_index.search(query, k=10)
exact_time = time.time()-start_time

start_time = time.time()
hnsw_D,hnsw_I = hnsw_index.search(query,k=10)
hnsw_time = time.time() - start_time

# check how many results overlap with each other
overlap = len(set(exact_I[0]).intersection(set(hnsw_I[0])))
overlap_percentage = overlap*100/10

print(f"Exact search time: {exact_time:.6f} seconds") 
print(f"HNSW search time: {hnsw_time:.6f} seconds")
print(f"Speed improvement: {exact_time/hnsw_time:.2f}x faster") 
print(f"Result overlap: {overlap_percentage:.1f}%")