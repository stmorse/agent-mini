import socket

from utilities.utils import load_config


def send_log_message(message, _host=None, _port=None):
    #Adding default versions to help manage moments when the _host and _port are NOT sent to the log message.
    #in this instance, it defaults to whatever is in the config.
        #Todo: Trenton: For Instance: start_faiss_server() calls send_log_message WITHOUT logging_host and logging_port

    if _host is None or _port is None:
        config = load_config('config.ini')
        _host, _port = config.get('DEFAULT', 'LOGGING_HOST'), str(config.get('DEFAULT', 'LOGGING_PORT'))
        send_log_message(f"Sanity Check: Host:{_host}, Port:{_port}", "localhost", 50004)

    """Sends a log message to the server."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((_host, int(_port)))
            client_socket.sendall(message.encode('utf-8'))
        return True
    except Exception as e:
        return e


if __name__ == '__main__':
    # Server configuration
    config = load_config('config.ini')
    h, p = config.get('DEFAULT', 'LOGGING_HOST'), str(config.get('DEFAULT', 'LOGGING_PORT'))
    send_log_message('This is a test log message.', h, p)
