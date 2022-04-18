from datetime import datetime
import socket

class Server:
    def __init__(self, port) -> None:
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(('', port))
        self.sock.listen()
        print('Server started on port {}'.format(port))

    def run(self) -> None:
        while True:
            conn, addr = self.sock.accept()
            print('Got connection from', addr)
            with conn:
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    else:
                        print(f'Got: {data.decode()}')
                    cur_time = datetime.now()
                    x = f'{cur_time} : {data.decode()}'
                    conn.sendall(x.encode())
            

    def __del__(self) -> None:
        self.sock.close()
        print('Server closed')


if __name__ == '__main__':
    server = Server(12345)
    server.run()
