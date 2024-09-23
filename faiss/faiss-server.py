from flask import Flask, request, jsonify
import faiss
import numpy as np

app = Flask(__name__)

# Initialize FAISS index
dimension = 128  # Example dimension, change as needed
index = faiss.IndexFlatL2(dimension)

# simple test endpoint
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "success", "message": "Server is running"}), 200

# add vectors to the index
@app.route('/add', methods=['POST'])
def add_vectors():
    data = request.json
    vectors = np.array(data['vectors']).astype('float32')
    index.add(vectors)
    return jsonify({"status": "success", "message": "Vectors added to index"}), 200

# search vectors in the index
@app.route('/search', methods=['POST'])
def search_vectors():
    data = request.json
    query_vector = np.array(data['query']).astype('float32').reshape(1, -1)
    k = data.get('k', 5)  # Number of nearest neighbors to return
    distances, indices = index.search(query_vector, k)
    return jsonify({"distances": distances.tolist(), "indices": indices.tolist()}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)