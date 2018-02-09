import socket, json, time, base64
import gather_data
from dispatcher import callback, call, commands
import numpy as np
import cv2 as cv
from io import BytesIO
from PIL import Image
import time
import commander
from keras.models import load_model
from settings import settings
TRAINING = False

UDP_IP = "localhost"

TOME_UDP_PORT = 11011
TOTHEM_UDP_PORT = 11111

tome_sock = socket.socket(socket.AF_INET,  # Internet
                          socket.SOCK_DGRAM)  # UDP

tothem_sock = socket.socket(socket.AF_INET,  # Internet
                            socket.SOCK_DGRAM)  # UDP


if not TRAINING:
    model = load_model(settings["DEFAULT_MODEL_FILE"])

@callback("ping")
def train_frame(message):
    print("Pong")


@callback("send_frame")
def render_stream(payload):
    front = payload["frame"]
    speed = float(payload["speed"])
    img = decode_image(front)
    if TRAINING:
        data = {"turn_rate": float(payload["turn_rate"])}
        gather_data.save_data(img, data)
    if False: # Mostra in diretta
        cv.imshow('window', img)
        cv.waitKey(1)
    if TRAINING:
        return {"verticalInput": commander.calc_throttle(speed, data["turn_rate"])}
    else:
        tn = model.predict(img.reshape([1, settings["HEIGHT"], settings["WIDTH"], settings["CHANNELS"]]))[0,0]
        print(tn)
        return {"verticalInput": commander.calc_throttle(speed, tn),
                "turnRate": str(tn)}

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
            response = call(msg)
            tothem_sock.sendto(json.dumps(response).encode(), (UDP_IP, TOTHEM_UDP_PORT))
        except BlockingIOError:
            pass
