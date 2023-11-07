import _thread
import socket
import struct
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from yolov5 import detect_api


def threaded(client_socket, addr, i):
    print("Connected by: ", addr[0], addr[1])
    model, stride, names, pt, imgsz = detect_api.model_load(weights='/Users/Nabong/Desktop/capstone/Garodeung/yolov5/customdataset_epoch100.pt', device="")
    save_dir = detect_api.make_dir(name="raspi")
    print("Model Loading Success!!!")
    while True:
        try:
            path[-1] = str(i)
            image_data = receive_image(client_socket)
            if not image_data:
                break
            with open(f"{'/'.join(path)}.jpg", 'wb') as file:
                file.write(image_data)
            print("Received and Saved Success!!")
            i += 1
            detect_api.api(model=model,stride=stride,names=names,pt=pt,imgsz=imgsz,save_dir=save_dir,view_img=True, source=f"{'/'.join(path)}.jpg", save_txt=True)
        except Exception as e:
            print("Disconnected by ", addr[0], ":", addr[1])
            print(f"Error: {e}")
            break


def receive_image(client_socket):
    # Receive the image size (as a 4-byte integer)
    image_size_bytes = client_socket.recv(4)
    if not image_size_bytes:
        return None

    image_size = struct.unpack("!I", image_size_bytes)[0]

    # Receive the image data
    image_data = b""
    remaining_data = image_size
    while remaining_data > 0:
        data = client_socket.recv(min(remaining_data, 1024))
        if not data:
            break
        image_data += data
        remaining_data -= len(data)

    return image_data


i = 1
path = '/Users/Nabong/Desktop/capstone/Garodeung/Rasberry/videos/'
path = path.split('/')
path.append(str(i))

ip = "0.0.0.0"
port = 8080

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((ip, port))
server_socket.listen()

print("server start")

while True:
    try:
        print("wait")
        cs, addr = server_socket.accept()
        _thread.start_new_thread(threaded, (cs, addr, i))
    except Exception as e:
        print(f"Error: {e}")
        break
server_socket.close()
