import socket, json, time, base64
import gather_data
from dispatcher import callback, call, commands
import numpy as np
import cv2 as cv
from io import BytesIO
from PIL import Image
import time

TRAINING = True

UDP_IP = "localhost"

TOME_UDP_PORT = 11011
TOTHEM_UDP_PORT = 11111

tome_sock = socket.socket(socket.AF_INET,  # Internet
                          socket.SOCK_DGRAM)  # UDP

tothem_sock = socket.socket(socket.AF_INET,  # Internet
                            socket.SOCK_DGRAM)  # UDP


@callback("ping")
def train_frame(message):
    print("Pong")


@callback("send_frame")
def render_stream(payload):
    front = payload["frame"]
    speed = payload["speed"]  # TODO Usa questa variabile nel testing
    data = {"turn_rate": payload["turn_rate"]}
    gather_data.save_data(decode_image(front), data)


def decode_image(base64string):
    img = base64.b64decode(base64string)
    img = Image.open(BytesIO(img))
    return np.asarray(img)


if __name__ == "__main__":
    tome_sock.bind((UDP_IP, TOME_UDP_PORT))
    tome_sock.setblocking(0)
    tome_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)

    while True:
        try:
            msg, addr = tome_sock.recvfrom(15000)
            msg = bytes(msg).decode()
            msg = json.loads(msg)
            call(msg)
            if not TRAINING:
                tothem_sock.sendto("Hello world".encode(), (UDP_IP, TOTHEM_UDP_PORT))

        except BlockingIOError:
            pass
