import socket
import subprocess

HOST = "127.0.0.1"
PORT = 4204

class Server:
    def __init__(self) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((HOST, PORT))
        self.socket.listen()

    def accept(self):
        conn, addr = self.socket.accept()
        print(f"Connection from {addr} has been established!")
        with conn:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                print(f"Received data: {data.decode('utf-8')}")
                conn.sendall(self.execute(data.decode('utf-8')))
    
    def execute(self, command):
        print("Executing command: \n" + command)
        x = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
        if x.returncode == 0 and len(x.stdout) > 0:
            return x.stdout
        return "".encode('utf-8')

if __name__ == "__main__":
    server = Server()
    server.accept()