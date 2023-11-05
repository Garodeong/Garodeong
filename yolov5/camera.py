import io
import sys
import cv2
import torch
import numpy as np
import requests, json
from pathlib import Path


# print(model_path)
model_path = 'stairv2_epoch100.pt'
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=True)
# model = torch.hub.load('.', 'custom', path=model_path, source='local')  # 'custom'은 사용자 정의 모델을 나타냄
model = torch.hub.load('/Users/kakaogames/personal/Garodeung/yolov5', 
                       'yolov5s', 
                       source='local',  # 로컬 파일 시스템을 지정합니다.
                       pretrained=True, # 이 옵션이 작동하려면 로컬 가중치 파일이 있어야 합니다.
                       verbose=True)

def parse_detection_log(log):
    # 'image 1/1:'과 'Speed:' 사이의 부분을 추출합니다.
    start = log.find('image 1/1:') + len('image 1/1:')
    end = log.find('Speed:')
    detected_objects = log[start:end].strip()
    
    # 감지된 물체들을 쉼표로 구분하여 리스트로 만듭니다.
    objects_list = detected_objects.split(', ')
    
    return objects_list

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
        # self.video = cv2.resize(self.video,(840,640))
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        # self.video = cv2.VideoCapture('video.mp4')

        self.weights = 'stairv2_epoch100.pt'
        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = 'cpu'  # or 'cuda:0'
        self.view_img = True  # 결과를 보여줄지 여부
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
    
        # YOLOv5 객체 검출 실행
        results = model(image)

        detections = []
        
        new_log = results.pandas().xyxy[0]
        print(new_log)    
        for _, row in new_log.iterrows():
            detections.append({
                'name': row['name'],
                'confidence': row['confidence'],
                'xmin': row['xmin'],
                'ymin': row['ymin'],
                'xmax': row['xmax'],
                'ymax': row['ymax']
            })

        url = "http://localhost:5044"
        headers = {'Content-Type': 'application/json'}
        # print(detections)
        # response = requests.post(url, data=json.dumps(detections), headers=headers, timeout=10)
        # print(response.text)
        # Render results on the frame
        # rendered_img = results.render()[0]
        a = np.squeeze(results.render())
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
