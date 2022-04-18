import socket
import select
import errno
import sys


IP = "127.0.0.1"
PORT = 12345

class ChatClient:
    HEADER_LENGTH = 10
    def __init__(self, ip, port, username) -> None:
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((ip, port))
        self.client_socket.setblocking(False)

        self.username = username.encode("utf-8")
        self.username_header = f"{len(self.username):<{self.HEADER_LENGTH}}".encode("utf-8")
        self.client_socket.send(self.username_header + self.username)

    def start(self):
        while True:
            message = input(f"{self.username}> ")
            if message:
                message = message.encode("utf-8")
                message_header = f"{len(message):<{self.HEADER_LENGTH}}".encode("utf-8")
                self.client_socket.send(message_header + message)

            try:
                while True:
                    username_header = self.client_socket.recv(self.HEADER_LENGTH)
                    if not len(username_header):
                        print("Connection closed by the server")
                        sys.exit()
                        
                    username_length = int(username_header.decode("utf-8").strip())
                    username = self.client_socket.recv(username_length).decode("utf-8")

                    message_header = self.client_socket.recv(self.HEADER_LENGTH)
                    message_length = int(message_header.decode("utf-8").strip())
                    message = self.client_socket.recv(message_length).decode("utf-8")

                    print(f"{username}> {message}")
            except IOError as err:
                if err.errno != errno.EAGAIN and err.errno != errno.EWOULDBLOCK:
                    print(f"Reading error: {str(err)}")
                    sys.exit()

                continue

            except Exception as e:
                print(f"Reading error: {str(e)}")
                sys.exit()

if __name__ == "__main__":
    chat_client = ChatClient(IP, PORT, str(sys.argv[1]))
    chat_client.start()