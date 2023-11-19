import io
import sys
import cv2
import torch
import numpy as np
import requests, json
import logging
import datetime
from pathlib import Path
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, ConnectionError

logging.basicConfig(level=logging.INFO)

# print(model_path)
model_path = 'stairv2_epoch100.pt'
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, verbose=True)
# model = torch.hub.load('.', 'custom', path=model_path, source='local')  # 'custom'은 사용자 정의 모델을 나타냄
model = torch.hub.load('/Users/kakaogames/personal/Garodeung/yolov5', 
                       'yolov5s', 
                       source='local',  # 로컬 파일 시스템을 지정합니다.
                       pretrained=True, # 이 옵션이 작동하려면 로컬 가중치 파일이 있어야 합니다.
                       verbose=True)

def insertData(detection):
    try:
        # 환경 변수에서 사용자 이름과 비밀번호 가져오기
        es_username = "elastic"
        es_password = "changeme"
        es = Elasticsearch(
            ['http://localhost:9200'],
            basic_auth=(es_username, es_password)
        )
        index = "garodeong-user01"

        # Elasticsearch에 데이터 삽입
        es.index(index=index, body=detection)
        logging.info("Data insertion successful")
    except ConnectionError as e:
        logging.error(e)
    except NotFoundError as e:
        logging.error(e)
    except Exception as e:
        print("Error inserting data into Elasticsearch:", e)

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
        self.video = cv2.VideoCapture(0)

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
        for _, row in new_log.iterrows():
            detection = {
                'detection': row['name'],
                'confidence': row['confidence'],
                'xmin': row['xmin'],
                'ymin': row['ymin'],
                'xmax': row['xmax'],
                'ymax': row['ymax'],
                "@timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
            }
            insertData(detection)

        a = np.squeeze(results.render())
        ret, jpeg = cv2.imencode('.jpg', image)

        return jpeg.tobytes()
