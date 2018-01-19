import socket, json, time, base64
from functools import wraps
from dispatcher import callback, call, commands
import numpy as np
import cv2 as cv
from io import BytesIO
from PIL import Image

UDP_IP = "127.0.0.1"
UDP_PORT = 3030

sock = socket.socket(socket.AF_INET,  # Internet
                     socket.SOCK_DGRAM)  # UDP


@callback("ping")
def train_frame(message):
    print("Pong")


@callback("render_stream")
def render_stream(message):
    front = message["front"]
    front = base64.b64decode(front)
    img = Image.open(BytesIO(front))
    img = np.asarray(img)
    cv.imshow('win', img)
    cv.waitKey(1)


if __name__ == "__main__":
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(0)
    last_time = 0

    while True:
        try:
            msg, addr = sock.recvfrom(15000)
            msg = bytes(msg).decode()
            msg = json.loads(msg)
            call(msg)
            print((time.time() - last_time) ** -1)
            last_time = time.time()
            time.sleep(0.01)
        except BlockingIOError:
            pass