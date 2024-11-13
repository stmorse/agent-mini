import requests
import json

import psycopg2

from utilities.utils import load_config
from utilities.faiss_utils import get_faiss_manager, create_faiss_index, \
    add_vectors, search_index
from utilities.logging_utils import send_log_message
from utilities.db_utils import get_connection_pool

TEST_LOGGER = False
TEST_POSTGRES = True
TEST_FAISS = False
TEST_LORAX = False

def test_logger():
    config = load_config('config.ini')
    h = config.get('DEFAULT', 'LOGGING_HOST')
    p = config.get('DEFAULT', 'LOGGING_PORT')
    result = send_log_message(f'Testing logger on host {h} and port {p}', 
                            _host=h, _port=p, verbose=True)
    print('Logger success? ', result)

def test_postgres():
    print('Connecting to Postgres ...')
    
    config = load_config('config.ini')
    conn_pool = get_connection_pool(config)
    conn = conn_pool.getconn()
    cursor = conn.cursor()

    print('Postgres version:')
    cursor.execute("SELECT version();")
    print(cursor.fetchone())

    print('Closing connection...')
    cursor.close()
    conn_pool.putconn(conn)


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
        Logger={TEST_LOGGER}, \
        Postgres={TEST_POSTGRES}, \
        FAISS={TEST_FAISS}, \
        LoRAX={TEST_LORAX}')

    if TEST_LOGGER:
        test_logger()

    if TEST_POSTGRES:
        test_postgres()
    
    if TEST_FAISS:
        test_faiss()
    
    if TEST_LORAX:
        test_lorax()

    print('Agent is done.')