import os
import sys
from camera import VideoCamera
from flask import Flask, render_template, Response, request, redirect, url_for, session

app = Flask(__name__)
VIDEO_PATH = sys.argv[1]
TOTAL_USERS = sys.argv[2]

@app.route('/')
# @login_required
def index():
    return render_template('/index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
# @login_required
def video_feed():
    return Response(gen(VideoCamera(VIDEO_PATH)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    app.run(host='0.0.0.0', port=9090+TOTAL_USERS, debug=True)

if __name__ == '__main__':
    main()