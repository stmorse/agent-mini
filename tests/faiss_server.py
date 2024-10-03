import argparse
import os
import pickle
import threading
import time
import shutil

import faiss
import numpy as np

from utilities.faiss_utils import FaissManager
from utilities.logging_utils import send_log_message
from utilities.utils import load_config

gpu_avail = 1 if faiss.get_num_gpus() > 0 else 0
if gpu_avail:
    res = faiss.StandardGpuResources()
else:
    res = None
index_dict = None


def create_faiss_index(index_dict, index_id, dimensions):
    send_log_message(f"Create index_dict: {index_dict}")
    if index_id in index_dict:
        send_log_message(f"Index {index_id} already exists\n")
        send_log_message(f"Either delete this index before recreating it, or use a different index_id.\n")
        return
    # Initialize FAISS index
    try:
        faiss_index = faiss.IndexFlatL2(dimensions)
        faiss_index = faiss.IndexIDMap(faiss_index)
        if gpu_avail:
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
        index_dict[index_id] = faiss_index
        send_log_message(f"Index {index_id} created and stored in manager dictionary")
        return index_id
    except Exception as e:
        send_log_message(f"Error creating index: {e}")
        return None


# TODO: @Rohith, please implement the following function:
def delete_faiss_index(index_dict, index_id):
    if index_id in index_dict:
        del index_dict[index_id]
        send_log_message(f"Index {index_id} deleted from manager dictionary")
    else:
        send_log_message(f"Index {index_id} not found, could not delete")


def load_faiss_index(manager_dict, index_id):
    if index_id not in manager_dict:
        send_log_message("Index not found.")
    else:
        send_log_message(f"loading faiss index {index_id}")
        faiss_index = faiss.read_index(uuid_to_filename(index_id, "faiss_index"))
        if gpu_avail:
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
        manager_dict[index_id] = faiss_index


def search_index(manager_dict, index_id, queries, k):
    send_log_message(f"Searching Index {index_id}. Msg 2A")
    index = manager_dict[index_id]
    if index is not None:
        send_log_message(f"Searching Index {index_id}. Msg 2B")
        queries = np.array(queries).astype('float32')
        D, I = index.search(queries, k)
        return D, I
    else:
        send_log_message(f"Index {index_id} not found")
        return None


def add_vectors(manager_dict, index_id, vectors, vector_ids):
    index = manager_dict[index_id]
    if index is not None:
        index.add_with_ids(np.array(vectors).astype('float32'), vector_ids)
        send_log_message(f"Vectors added to index {index_id}")
    else:
        send_log_message(f"Index {index_id} not found")


def start_faiss_server(host, port, logging_host=None, logging_port=None):
    print("Starting FAISS Management Server")
    index_dict = load_index_dict('faiss_index_dir')

    FaissManager.register('get_index_dict', callable=lambda: index_dict)
    FaissManager.register('search_index', callable=lambda index_id, queries, k: search_index(index_dict, index_id, queries,
                                                                                     k))
    FaissManager.register('add_vectors',
                          callable=lambda index_id, vectors, vector_ids: add_vectors(index_dict, index_id, vectors,
                                                                                     vector_ids))
    FaissManager.register('create_faiss_index',
                          callable=lambda index_id, dimensions: create_faiss_index(index_dict, index_id, dimensions))
    FaissManager.register('load_faiss_index', callable=load_faiss_index)
    FaissManager.register('delete_faiss_index', callable=lambda index_id: delete_faiss_index(index_dict, index_id))

    manager = FaissManager(address=(host, int(port)), authkey=b'faiss')
    server = manager.get_server()
    send_log_message("FAISS Management Server started, listening on port " + str(port), logging_host, logging_port)
    # Start periodic saving of the index dictionary
    periodic_save(index_dict, "faiss_index_dir")
    server.serve_forever()


def load_index_dict(directory):
    create_directory(directory)
    index_dict_temp = {}
    for filename in os.listdir(directory):
        if filename.endswith(".index"):
            key = os.path.splitext(filename)[0]
            filepath = os.path.join(directory, filename)
            index_dict_temp[key] = faiss.read_index(filepath)
    return index_dict_temp


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")


def uuid_to_filename(uuid, ext='pkl'):
    return f"{uuid}.{ext}"


def save_index_dict(index_dict, directory):
    os.makedirs(directory, exist_ok=True)
    for key, index in index_dict.items():
        filename = os.path.join(directory, f"{key}.index")
        index = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index, filename)


def rotate_files(dirname):
    # Remove .2 if it exists
    if os.path.exists(dirname + ".2"):
        shutil.rmtree(dirname + ".2")

    # Move .1 to .2 if it exists
    if os.path.exists(dirname + ".1"):
        shutil.move(dirname + ".1", dirname + ".2")

    # Check if the directory exists and is not empty
    if os.path.exists(dirname) and os.listdir(dirname):
        # Create a new .1 directory
        os.makedirs(dirname + ".1", exist_ok=True)

        # Copy contents of dirname to .1
        for item in os.listdir(dirname):
            s = os.path.join(dirname, item)
            d = os.path.join(dirname + ".1", item)
            if os.path.isdir(s):
                shutil.copytree(s, d, symlinks=True, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    else:
        print(f"Directory {dirname} does not exist or is empty. Skipping rotation.")

    # Ensure the main directory exists
    os.makedirs(dirname, exist_ok=True)


def periodic_save(index_dict, filename, interval=60):
    def save():
        while True:
            time.sleep(interval)
            rotate_files(filename)
            save_index_dict(index_dict, filename)
            send_log_message(f"Dictionary saved to {filename} and rotated files")

    thread = threading.Thread(target=save, daemon=True)
    thread.start()


if __name__ == "__main__":
    config = load_config('config.ini')
    config_faiss_host = config.get('DEFAULT', 'FAISS_HOST')
    config_faiss_port = int(config.get('DEFAULT', 'FAISS_PORT'))
    config_logging_host = config.get('DEFAULT', 'LOGGING_HOST')
    config_logging_port = int(config.get('DEFAULT', 'LOGGING_PORT'))

    parser = argparse.ArgumentParser("faiss_server.py")
    parser.add_argument("--FAISS_HOST", nargs='?', const=config_faiss_host, default=config_faiss_host,
                        help="Host name for the faiss server to live at. Default localhost", type=str)
    parser.add_argument("--FAISS_PORT", nargs='?', const=config_faiss_port, default=config_faiss_port, help="",
                        type=int)
    parser.add_argument("--LOGGING_HOST", nargs='?', const=config_logging_host, default=config_logging_host, help="",
                        type=str)
    parser.add_argument("--LOGGING_PORT", nargs='?', const=config_logging_port, default=config_logging_port, help="",
                        type=int)

    args = parser.parse_args()
    _faiss_host, _faiss_port, _logging_host, _logging_port = args.FAISS_HOST, args.FAISS_PORT, args.LOGGING_HOST, args.LOGGING_PORT
    start_faiss_server(_faiss_host, _faiss_port, _logging_host, _logging_port)
