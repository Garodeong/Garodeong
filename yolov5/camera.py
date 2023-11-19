import io
import os
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


sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from Rasberry import server_getIP

logging.basicConfig(level=logging.INFO)

# def insertData(detection):
#     try:
#         # 환경 변수에서 사용자 이름과 비밀번호 가져오기
#         es_username = "elastic"
#         es_password = "changeme"
#         es = Elasticsearch(
#             ['http://localhost:9200'],
#             basic_auth=(es_username, es_password)
#         )
#         index = "garodeong-user01"

#         # Elasticsearch에 데이터 삽입
#         es.index(index=index, body=detection)
#         logging.info("Data insertion successful")
#     except ConnectionError as e:
#         logging.error(e)
#     except NotFoundError as e:
#         logging.error(e)
#     except Exception as e:
#         print("Error inserting data into Elasticsearch:", e)

def parse_detection_log(log):
    # 'image 1/1:'과 'Speed:' 사이의 부분을 추출합니다.
    start = log.find('image 1/1:') + len('image 1/1:')
    end = log.find('Speed:')
    detected_objects = log[start:end].strip()
    
    # 감지된 물체들을 쉼표로 구분하여 리스트로 만듭니다.
    objects_list = detected_objects.split(', ')
    
    return objects_list

class VideoCamera(object):
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)
        print(self.video_path)

        self.img_size = 640
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.device = 'cpu'  # or 'cuda:0'
        self.view_img = True  # 결과를 보여줄지 여부
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()

        # results = model(image)
        # a = np.squeeze(results.render())
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
