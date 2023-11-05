import time
import picamera2 as p2
import datetime
import struct
import socket


def send_capture(picam2, client_socket):
    #picam2.start_preview(p2.Preview.QTGL)
    ts = datetime.datetime.now()
    ts = ts.strftime("%Y-%m-%d %H:%M:%S")
    picam2.start()
    filename = f'/home/nabong/Desktop/cctv/captures/{ts}.jpg'
    picam2.caputre_file(filename)

    with open(filename, 'rb') as file:
        image_data = file.read()
    image_size = len(image_data)
    image_size_bytes = struct.pack('!I', image_size)
    client_socket.send(image_size_bytes)

    sent_data = 0
    while sent_data < image_size:
        chunk = image_data[sent_data:sent_data+1024]
        client_socket.send(chunk)
        sent_data += len(chunk)
    
    #picam2.stop_preview()
    picam2.stop()


HOST = '192.168.200.169'
PORT = 8080

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST,PORT))

picam2 = p2.Picamera2()
camera_config = picam2.create_still_configuration(lores={'size':(640,480)}, display='lores')
picam2.configure(camera_config)

while True:
    send_capture(picam2, client_socket)
client_socket.close()
