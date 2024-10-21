import requests
import json

import psycopg2

from utilities.faiss_utils import *

TEST_POSTGRES = True
TEST_FAISS = False
TEST_LORAX = False

def test_postgres():
    print('Connecting to Postgres ...')
    connection = psycopg2.connect(
        host="postgres", 
        port="5432",
        database="mydb", 
        user="user", 
        password="password"
    )

    print('Postgres version:')
    cursor = connection.cursor()
    cursor.execute("SELECT version();")
    print(cursor.fetchone())

    print('Closing connection...')
    cursor.close()
    connection.close()


def test_faiss():
    # FAISS server tests

    print('Testing FAISS server ...')

    # initialize the manager
    manager = get_faiss_manager(a=('faiss', 90), key=b'faiss')
    create_faiss_index(manager, 1, 2)

    # add vectors to the index
    vectors = [[1.0, 2.0], [3.0, 4.0]]
    add_vectors(manager, 1, vectors, [1, 2])

    # search the index
    queries = [[1.0, 2.0], [3.0, 4.0], [1.5, 2.5]]
    search_index(manager, 1, queries, k=1)


def test_lorax():
    # LoRAX server tests

    print('Testing LoRAX server ...')

    prompt = {
        'inputs': 'Translate English to French: Hello, my name is Trenton.',
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



if __name__ == '__main__':    
    print('Agent is running ...')
    print(f'Testing: \
          Postgres={TEST_POSTGRES}, FAISS={TEST_FAISS}, LoRAX={TEST_LORAX}')

    if TEST_POSTGRES:
        test_postgres()
    
    if TEST_FAISS:
        test_faiss()
    
    if TEST_LORAX:
        test_lorax()

    print('Agent is done.')