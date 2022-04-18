import socket
import sys
import os

class FileClient:
    SEPARATOR = "<SEPARATOR>"
    BUFFER_SIZE = 4096

    def __init__(self, ip, port) -> None:
        self.sock = socket.socket()
        print(f"Connecting to {ip}:{port}")
        self.sock.connect((ip, port))
        print(f"Connected to {ip}:{port}")

    def send(self, filename):
        if not os.path.isfile(filename):
            print(f"{filename} does not exist!")
            return
        filesize = os.path.getsize(filename)
        self.sock.send(f"{filename}{self.SEPARATOR}{filesize}".encode())

        with open(filename, "rb") as f:
            print(f"Sending {filename}")
            while True:
                bytes_read = f.read(self.BUFFER_SIZE)
                if not bytes_read:
                    # we've completed sending files
                    break
                self.sock.sendall(bytes_read)
        print(f"Done sending {filename}")

        self.sock.close()


if __name__ == "__main__":
    client = FileClient("127.0.0.1", 12345)
    filename = sys.argv[1]
    client.send(filename)