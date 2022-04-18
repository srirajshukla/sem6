import os
import socket
import sys
import threading
import datetime

class UDPChatServer:
    def __init__(self, name, client_addr=None, client_port=None) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', 1234))
        self.name = name
        self.client_addr = "127.0.0.1"
        self.client_port = 1235

    def send(self):
        while True:
            message = input('>> ')
            if (message=='exit'):
                break
            message = f'{self.name} [{datetime.datetime.now()}]: {message}'
            self.socket.sendto(message.encode(), (self.client_addr, self.client_port))
        self.socket.close()
        sys.exit(0)

    def recieve(self):
        while True:
            data, addr = self.socket.recvfrom(1024)
            print(data.decode())
            print("\n>> ", end="")



if __name__ == '__main__':
    name = input("Enter your name: ")
    server = UDPChatServer(name)
    x1 = threading.Thread(target=server.send)
    x2 = threading.Thread(target=server.recieve)
    x1.start()
    x2.start()