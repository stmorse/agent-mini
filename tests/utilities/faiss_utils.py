from multiprocessing.managers import BaseManager
from utilities.logging_utils import send_log_message


class FaissManager(BaseManager):
    pass


def get_faiss_manager(a=('localhost', 5000), key=b'faiss'):
    manager = FaissManager(address=a, authkey=key)
    manager.connect()
    return manager


def create_faiss_index(manager, index_id, dimension):
    manager.create_faiss_index(index_id, dimension)
    send_log_message(f"Index {index_id} created")
    return index_id


def delete_faiss_index(manager, index_id):
    manager.delete_faiss_index(index_id)
    send_log_message(f"Index {index_id} deleted")


def add_vectors(manager, index_id, vectors, vector_ids):
    success = manager.add_vectors(index_id, vectors, vector_ids)
    if success:
        send_log_message(f"Vectors added to index {index_id}")
    else:
        send_log_message(f"Failed to add vectors to index {index_id}")


def search_index(manager, index_id, queries, k):
    send_log_message(f"Searching Index {index_id}. Msg 1")
    distances, indexes = manager.search_index(index_id, queries, k)
    if distances and indexes is not None:
        #Do we want to add the full distances / indexes into the log?
        send_log_message(f"Search results from index {index_id}:", distances, indexes)
        return distances, indexes
    else:
        send_log_message(f"Failed to search index {index_id}")
        return [], []


FaissManager.register('get_index_dict')
FaissManager.register('search_index')
FaissManager.register('add_vectors')
FaissManager.register('create_faiss_index')
FaissManager.register('load_faiss_index')
FaissManager.register('delete_faiss_index')
