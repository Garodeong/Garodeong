import _thread
import socket
import struct
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from yolov5 import detect_api


def threaded(client_socket, addr):
    device = ""  # CPU는 cpu, 기본 GPU는 빈 문자열
    print("Connected by: ", addr[0], addr[1])
    model, stride, names, pt, imgsz = detect_api.model_load(weights='/Users/Nabong/Desktop/capstone/Garodeung/yolov5/customdataset_epoch100.pt', device=device)
    save_dir = detect_api.make_dir(name="raspi")
    print("Model Loading Success!!!")

    
    
    data = client_socket.recv(1024).decode()
    tmp = data.split(":")
    HOST = tmp[0]
    PORT = 8081

    client_socket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket2.connect((HOST, PORT))

    source = f"http://{data}/stream.mjpg"
    print("Source : ",source)
    detect_api.api(model=model,stride=stride,names=names,pt=pt,imgsz=imgsz,save_dir=save_dir,HOST=HOST,PORT=PORT,view_img=True, source=source, save_txt=False, device=device)


ip = "0.0.0.0"
port = 8080

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((ip, port))
server_socket.listen()

print("Run IPget Server ...")
#model, stride, names, pt, imgsz = detect_api.model_load(weights=f"/Users/Nabong/Desktop/capstone/Garodeung/yolov5/yolov5s.pt", device='')
#save_dir = detect_api.make_dir()
#detect_api.api(model=model,stride=stride,names=names,pt=pt,imgsz=imgsz,save_dir=save_dir,
#view_img=True, source="http://192.168.200.150:8000/stream.mjpg", save_txt=False)
while True:
    try:
        print("wait")
        cs, addr = server_socket.accept()
        _thread.start_new_thread(threaded, (cs, addr))
    except Exception as e:
        print(f"Error: {e}")
        break
server_socket.close()