import requests
import json

# from utilities.faiss_utils import *

if __name__ == '__main__':
    # FAISS server tests

    # initialize the manager
    # manager = get_faiss_manager(a=('faiss', 80), key=b'faiss')
    # create_faiss_index(manager, 1, 2)

    # # add vectors to the index
    # vectors = [[1.0, 2.0], [3.0, 4.0]]
    # add_vectors(manager, 1, vectors, [1, 2])

    # # search the index
    # queries = [[1.0, 2.0], [3.0, 4.0], [1.5, 2.5]]
    # search_index(manager, 1, queries, k=1)


    # LoRAX server tests

    prompt = {
        'inputs': 'translate English to French: Hello, my name is Trenton.',
        'parameters': {
            'max_new_tokens': 100
        }
    }

    print(f'Sending prompt: {prompt}')

    response = requests.post(
        'http://lorax:80/generate',
        data=json.dumps(prompt),
        headers={
            'Content-Type': 'application/json'
        })
    
    print(json.loads(response.text))