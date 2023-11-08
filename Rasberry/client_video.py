import time
import picamera2 as p2
import datetime
import struct
import socket
import _thread


def send_video(file_name, client_socket):
	while True:
		with open(file_name, "rb") as file:
			data = file.read()
			while data:
				client_socket.send(data)
				data = file.read()
			time.sleep(1)


def record(picam2, encoder, client_socket):
	now = datetime.datetime.now()
	fs = now.strftime('%Y-%m-%d %H:%M:%S')
	filename = f"/home/nabong/Desktop/cctv/videos/{fs}.h264"
	cam.start_preview(p2.Preview.QTGL)
	cam.start_recording(encoder,filename)
	_thread.start_new_thread(send_video, (filename, client_socket))
	#while True:
	#	pass
	time.sleep(10)
	cam.stop_recording()
	cam.stop_preview()
	"""
	while True:
		with open(filename, 'rb') as file:
			data = file.read()
			while data:
				client_socket.send(data)
				data = file.read()
	"""
	cur = time.time()
	with open(filename, 'rb') as file:
			data = file.read()
			while data:
				client_socket.send(data)
				data = file.read()
	eta = time.time() - cur
	#transfer_rate = 

HOST = '192.168.200.169'
PORT = 8080

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST, PORT))

cam = p2.Picamera2()
encoder = p2.encoders.H264Encoder()
video_config = cam.create_video_configuration(main={'size':(1920,1080)},lores={'size':(640,480)},display='lores')
cam.configure(video_config)

record(cam, encoder, client_socket)
