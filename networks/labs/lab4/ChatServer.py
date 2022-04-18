import socket
import select

IP = "127.0.0.1"
PORT = 12345

class ChatServer:
    HEADER_LENGTH = 10
    def __init__(self, ip, port) -> None:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((ip, port))
        self.server_socket.listen()

        self.socket_list = [self.server_socket]
        self.clients = {}
        print("Server started on {}:{}".format(ip, port))

    def receive_message(self, client_socket):
        try:
            message_header = client_socket.recv(self.HEADER_LENGTH)
            
            if not len(message_header):
                return False

            message_length = int(message_header.decode("utf-8").strip())
            
            return {'header': message_header, 'data': client_socket.recv(message_length)}
        
        except:
            return False

    def start(self):
        while True:
            read_sockets, _, exception_sockets = select.select(self.socket_list, [], self.socket_list)
            for notified_socket in read_sockets:
                if notified_socket == self.server_socket:
                    client_socket, client_address = self.server_socket.accept()

                    user = self.receive_message(client_socket)

                    if user is False:
                        continue

                    self.socket_list.append(client_socket)
                    self.clients[client_socket] = user

                    print("Accepted new connection from {}:{}, username: {}"
                            .format(*client_address, user['data'].decode("utf-8")))
                
                else:
                    message = self.receive_message(notified_socket)

                    if message is False:
                        print("Closed connection from: {}".format(self.clients[notified_socket]['data'].decode("utf-8")))

                        self.socket_list.remove(notified_socket)
                        del self.clients[notified_socket]
                        continue

                    user = self.clients[notified_socket]
                    print("Received message from {}: {}".format(user['data'].decode("utf-8"), message['data'].decode("utf-8")))

                    for client_socket in self.clients:
                        if client_socket != notified_socket:
                            client_socket.send(user['header'] + user['data'] + message['header'] + message['data'])

            for notified_socket in exception_sockets:
                # If we've got exceptional socket, probably it's broken one
                # so we need to remove it from socket list
                self.socket_list.remove(notified_socket)
                del self.clients[notified_socket]


if __name__ == "__main__":
    server = ChatServer(IP, PORT)
    server.start()