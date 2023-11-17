# 로컬 노트북에서 탐지한 객체를 소켓 통신으로 받아야하는데, 서버 역할을 하는 코드
# speak.py에서 speak함수로 소켓 통신으로 받은 객체가 전방에 있다고 알려주는 코드

import _thread
import socket
import speak
import os


def threaded(client_socket, addr):
    print("Connected by: ", addr[0], addr[1])
    while True:
        try:
            data = client_socket.recv(1024)
            data = data.decode()
            print("Recieved from " + addr[0], ":", addr[1], data)
            data = data.split(".")
            data.pop()
            data = set(data)
            
            for elem in data:
                speak.speak(msg=elem)
            #time.sleep() # 많은 객체가 탐지되면, 말이 중첩돼서, 일부러 1초 간격을 두게끔 한다.
            
        except ConnectionResetError as e:
            print("Disconnected by", addr[0], ":", addr[1])
            print(f"Error: {e}")

    

ip = "0.0.0.0"
port = 8081     

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((ip, port))
server_socket.listen()

print("Run Notification Server ... ")

while True:
    print("wait")
    cs, addr = server_socket.accept()
    _thread.start_new_thread(threaded, (cs, addr))
