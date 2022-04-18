import socket
import os

IP = "127.0.0.1"
PORT = 12345

class FileServer:
    SEPARATOR = "<SEPARATOR>"
    BUFFER_SIZE = 4096

    def __init__(self, ip, port) -> None:
        self.sock = socket.socket()
        self.sock.bind((ip, port))
        self.sock.listen()
        print(f"Listening on {ip}:{port}")

    def receive(self):
        client_socket, address = self.sock.accept()
        received = client_socket.recv(self.BUFFER_SIZE).decode()
        filename, filesize = received.split(self.SEPARATOR)

        filename = "recv-" + os.path.basename(filename)
        filesize = int(filesize)

        with open(filename, "wb") as f:
            print(f"Incoming file, saving as {filename}")
            while True:
                bytes_read = client_socket.recv(self.BUFFER_SIZE)
                if not bytes_read:
                    # we've completed receiving files
                    break
                f.write(bytes_read)
        print(f"Done receiving {filename}")

        client_socket.close()
        self.sock.close()


if __name__ == "__main__":
    server = FileServer(IP, PORT)
    server.receive()
