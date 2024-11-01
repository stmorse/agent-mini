import socket

def send_log_message(message, _host=None, _port=None):
    # if host/port not set, send error
    if _host is None or _port is None:
        print('Error: host and/or port not set.')
        return False

    # send a log message to the server
    try:
        print(f'trying message: {message} on host {_host} and port {_port}')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((_host, int(_port)))
            client_socket.sendall(message.encode('utf-8'))
        return True
    except Exception as e:
        return e