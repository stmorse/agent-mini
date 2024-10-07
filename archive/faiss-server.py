from flask import Flask, request, jsonify
import faiss
import numpy as np

app = Flask(__name__)

# Initialize FAISS index
# dimension = 128  # Example dimension, change as needed
# index = faiss.IndexFlatL2(dimension)
index = None

# simple test endpoint
@app.route('/test', methods=['GET'])
def test():
    return jsonify({"status": "success", "message": "Server is running"}), 200

# initialize the index
@app.route('/init', methods=['POST'])
def init_index():
    # initialize dimension from payload, default as 128
    dim = request.json.get('dimension', 128)
    index = faiss.IndexFlatL2(dim)
    return jsonify({"status": "success", "message": "Index initialized"}), 200

# add vectors to the index
@app.route('/add', methods=['POST'])
def add_vectors():
    if index is None:
        return jsonify({"status": "error", "message": "Index not initialized"}), 400
    
    # store payload as `data` and extract vectors
    data = request.json
    vectors = np.array(data['vectors']).astype('float32')

    # add to index
    index.add(vectors)
    return jsonify({"status": "success", "message": "Vectors added to index"}), 200

# search vectors in the index
@app.route('/search', methods=['POST'])
def search_vectors():
    if index is None:
        return jsonify({"status": "error", "message": "Index not initialized"}), 400

    # store payload as `data` and extract query vector
    data = request.json
    query_vector = np.array(data['query']).astype('float32').reshape(1, -1)

    # extract `k` (Number of nearest neighbors to return) (default as 5)
    k = data.get('k', 5)

    # search in the index
    distances, indices = index.search(query_vector, k)
    return jsonify({"distances": distances.tolist(), "indices": indices.tolist()}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)