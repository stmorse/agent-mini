import socket
import threading
import logging
import argparse

from utilities.utils import load_config

# config logging to save to file
logging.basicConfig(filename='server_logs.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(message)s')


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
            client_handler = threading.Thread(
                target=handle_client_connection, args=(client_socket,))
            client_handler.start()


if __name__ == '__main__':
    # when run from command line, allow custom set of port
    # if nothing set, use config.ini

    # host needs to be 0.0.0.0 (on logger)
    # host needs to be logger (on clients)

    config = load_config('config.ini')
    config_logging_port = int(config.get('DEFAULT', 'LOGGING_PORT'))

    parser = argparse.ArgumentParser("logging_server.py")
    parser.add_argument("--LOGGING_PORT", nargs='?', 
                        const=config_logging_port, default=config_logging_port, 
                        help="Port number for the faiss server to live at. Default from config.ini")

    args = parser.parse_args()
    host, port = '0.0.0.0', args.LOGGING_PORT

    start_logging_server(host, port)