import sys
import json
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

class Model:
    def __init__(self, name):
        self.endpoint = 'http://ollama:80/api/generate'
        self.headers = {'Content-Type': 'application/json'}
        self.name = name

    def get_response(self, prompt):
        # create data dict
        data = {
            'model': self.name,
            'prompt': prompt,
            'stream': False
        }

        # do API call
        response = requests.post(
            self.endpoint,
            headers=self.headers,
            data=json.dumps(data)
        )

        return response.json()
    
# class Memory:
#     def __init__(self, dimension=128):
#         # this class contains sentence embedding model 
#         # FAISS lookup is performed via API
#         self.model = SentenceTransformer('all-MiniLM-L6-v2')
#         self.embeddings = []

#         # FAISS API endpoints
#         self.init_endpoint = 'http://faiss:80/init'
#         self.add_endpoint = 'http://faiss:80/add'
#         self.search_endpoint = 'http://faiss:80/search'
#         self.headers = {'Content-Type': 'application/json'}
        
#         # initialize the index
#         self._init_index(dimension)

#     def _init_index(self, dimension=128):
#         data = {'dimension': dimension}
#         response = requests.post(
#             self.init_endpoint,
#             headers=self.headers,
#             data=json.dumps(data)
#         )

#         return response.json()
    
#     def _add_vectors(self, vectors):
#         data = {'vectors': vectors}
#         response = requests.post(
#             self.add_endpoint,
#             headers=self.headers,
#             data=json.dumps(data)
#         )

#         return response.json()
    
#     def _search_vectors(self, query, k=5):
#         data = {'query': query, 'k': k}
#         response = requests.post(
#             self.search_endpoint,
#             headers=self.headers,
#             data=json.dumps(data)
#         )

#         return response.json()
    
#     def add_sentences(self, sentences):
#         embedding = self.get_embeddings(sentences)
#         self.embeddings.extend(embedding)
#         self._add_vectors(embedding)
    
#     def get_embeddings(self, sentences):
#         return self.model.encode(sentences)

# Function to test FAISS server
def test_faiss_server():
    # Load Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # this model default embed dimension is 384

    # Example sentences to embed
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown animal leaps over a lazy canine.",
        "The rain in Spain stays mainly in the plain.",
        "Artificial intelligence is the future.",
        "Data science is transforming industries."
    ]

    # Generate embeddings for each sentence
    embeddings = model.encode(sentences)

    # Initialize FAISS server (POST request to /init)
    dimension = embeddings.shape[1]
    init_payload = {"dimension": dimension}
    init_response = requests.post("http://faiss:80/init", json=init_payload)

    if init_response.status_code == 200:
        print("FAISS index initialized successfully.")
    else:
        print(f"Failed to initialize FAISS index: {init_response.status_code}")
        return

    # Add sentence embeddings to FAISS index (POST request to /add)
    add_payload = {"vectors": embeddings.tolist()}
    add_response = requests.post("http://faiss:80/add", json=add_payload)

    if add_response.status_code == 200:
        print("Vectors added to FAISS index successfully.")
    else:
        print(f"Failed to add vectors: {add_response.status_code}")
        return

    # Search for nearest neighbors of a query sentence (POST request to /search)
    query_sentence = "The fast brown fox."
    query_embedding = model.encode([query_sentence])[0]
    search_payload = {"query": query_embedding.tolist()}
    search_response = requests.post("http://faiss:80/search", json=search_payload)

    if search_response.status_code == 200:
        neighbors = search_response.json()
        print(f"Query sentence: {query_sentence}")
        print(f"> Nearest neighbors: {neighbors}")
    else:
        print(f"Failed to search FAISS index: {search_response.status_code}")
    

if __name__ == "__main__":
    # test message
    print('Agent is running ...')

    # test memory
    print('Testing FAISS server ...')
    test_faiss_server()
    print('FAISS server test completed.\n\n')

    # test model
    print('Testing model ...')
    # initialize the model wrapper
    model = Model("llama3.1")
    prompt = "Why did the chicken cross the road?"
    response = model.get_response(prompt)
    print('\n' + response["response"] + '\n\n')

    print('Model test completed.\nExiting ...')
    sys.exit(0)

    # print("Type 'q' to quit.")
    # while True:
    #     prompt = input(">> ")
        
    #     if prompt.lower() == "q":
    #         print("Exiting")
    #         break
        
    #     response = model.get_response(prompt)
    #     print("\n" + response["response"] + "\n\n")
