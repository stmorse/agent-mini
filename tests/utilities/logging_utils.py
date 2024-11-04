import socket

def send_log_message(message, _host=None, _port=None, verbose=False):
    # if host/port not set, send error
    if _host is None or _port is None:
        print('Error: host and/or port not set. Message: ', message)
        return False

    # send a log message to the server
    try:
        if verbose:
            print(f'{_host}:{_port}: {message}')
            
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((_host, int(_port)))
            client_socket.sendall(message.encode('utf-8'))
        return True
    except Exception as e:
        return e