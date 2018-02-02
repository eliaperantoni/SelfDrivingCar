import socket, json, time, base64
import gather_data
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


@callback("send_frame")
def render_stream(payload):
    front = payload["frame"]
    data = {"speed": payload["speed"],"turn_rate": payload["turn_rate"]}
    gather_data.save_data(decode_image(front), data)


def decode_image(base64string):
    img = base64.b64decode(base64string)
    img = Image.open(BytesIO(img))
    return np.asarray(img)


if __name__ == "__main__":
    sock.bind((UDP_IP, UDP_PORT))
    sock.setblocking(0)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
    last_time = 0

    while True:
        try:
            msg, addr = sock.recvfrom(15000)
            msg = bytes(msg).decode()
            msg = json.loads(msg)
            call(msg)

            last_time = time.time()
            time.sleep(0.01)
        except BlockingIOError:
            pass
