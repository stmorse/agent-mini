import socket
import threading
import logging
import argparse

from utilities.utils import load_config

# Configure logging
logging.basicConfig(filename='server_logs.log', level=logging.INFO, format='%(asctime)s - %(message)s')


def handle_client_connection(client_socket):
    """Handles a single client connection."""
    with client_socket:
        while True:
            message = client_socket.recv(1024).decode('utf-8')
            if not message:
                break
            logging.info(message)
            print('Received message from client: {}'.format(message))


def start_logging_server(logging_host, logging_port: int):
    """Starts the logging server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((logging_host, int(logging_port)))
        server_socket.listen()
        print(f'Server started, listening on {logging_host}:{logging_port}')

        while True:
            client_socket, addr = server_socket.accept()
            # print(f'Connected by {addr}')
            client_handler = threading.Thread(target=handle_client_connection, args=(client_socket,))
            client_handler.start()


if __name__ == '__main__':
    # get commandline variables

    config = load_config('config.ini')
    parser = argparse.ArgumentParser("logging_server.py")

    config_logging_host = config.get('DEFAULT', 'LOGGING_HOST')
    config_logging_port = int(config.get('DEFAULT', 'LOGGING_PORT'))

    parser.add_argument("--LOGGING_HOST", nargs='?', 
                        const=config_logging_host, default=config_logging_host, 
                        help="Host name for the faiss server to live at. Default localhost")
    parser.add_argument("--LOGGING_PORT", nargs='?', 
                        const=config_logging_port, default=config_logging_port, 
                        help="Port number for the faiss server to live at. Default from config.ini")

    args = parser.parse_args()
    host, port = args.LOGGING_HOST, args.LOGGING_PORT

    start_logging_server(host, port)