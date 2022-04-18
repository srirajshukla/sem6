import socket

class UDPClient:
    def __init__(self) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, message) -> None:
        try:
            sent = self.sock.sendto(message, ("localhost", 1234))
            data, server = self.sock.recvfrom(1024)
            print(data.decode())
        except Exception as e:
            print(e)
            
    def close(self) -> None:
        self.sock.close()

    def run(self) -> None:
        while True:
            message = input('> ')
            if (message=='exit'):
                self.close()
                break
            self.send(message.encode())

if __name__ == '__main__':
    client = UDPClient()
    client.run()