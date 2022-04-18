import socket

class Client:
    def __init__(self, host, port) -> None:
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        print('Connected to {}:{}'.format(host, port))

    def run(self) -> None:
        while True:
            data = input('> ')
            if not data:
                break
            self.sock.sendall(data.encode())
            data = self.sock.recv(1024)
            print(data.decode())

    def __del__(self) -> None:
        self.sock.close()
        print('Connection closed')


if __name__ == '__main__':
    client = Client('127.0.0.1', 12345)
    client.run()