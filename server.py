import socket, json
from functools import wraps
from dispatcher import callback, call, commands

UDP_IP = "127.0.0.1"
UDP_PORT = 3030

sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

@callback("ping")
def train_frame(message):
    print("Pong")

print(commands)

while True:
    msg, addr = sock.recvfrom(15000)
    msg = json.loads(str(msg, 'utf-8'))
    call(msg)