import socket
import subprocess

HOST = "127.0.0.1"
PORT = 4204

class Client:
    def __init__(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((HOST, PORT))
        print("This program executes commands on the server and shows you the output.")

    def start(self):
        while True:
            command = input("Enter command: ")
            self.socket.sendall(command.encode('utf-8'))
            data = self.socket.recv(1024)
            print(data.decode('utf-8'))

if __name__ == "__main__":
    server = Client()
    server.start()