import socket
import datetime

class UDPServer:
    def __init__(self, addr: str, port: int) -> None:
        """
        Starting a UDP server with SOCK_DGRAM on the given address and port.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((addr, port))
        print(f"\nServer started on {addr}:{port}\n")


    def run(self) -> None:
        """
        Listen for incoming messages and echo them back to the client.
        """
        while True:
            data, addr = self.socket.recvfrom(1024)
            print(f"Got: {data.decode()}")

            if data:
                d = f"{datetime.datetime.now()} : {data.decode()}"
                self.socket.sendto(d.encode(), addr)
            else:
                self.socket.sendto("".encode(), addr)
                self.socket.close()
                break


if __name__ == '__main__':
    server = UDPServer("127.0.0.1", 1234)
    server.run()