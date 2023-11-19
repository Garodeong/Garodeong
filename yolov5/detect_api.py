# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
import csv
import os
import platform
import sys
from pathlib import Path
from collections import defaultdict
import winsound as sd
from typing import List
import socket

import torch

from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError, ConnectionError

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
#ROOT="/Users/Nabong/Desktop/capstone/Garodeung/"
from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, smart_inference_mode


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
        # LOGGER.info("Data insertion successful")
    # except ConnectionError as e:
    #     LOGGER.error(e)
    # except NotFoundError as e:
    #     LOGGER.error(e)
    except Exception as e:
        pass
        # LOGGER.error("Error inserting data into Elasticsearch:", e)

def model_load(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        dnn=False,  # use OpenCV DNN for ONNX inference
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        half=False,  # use FP16 half-precision inference
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        imgsz=(640, 640),  # inference size (height, width)
):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    
    return model, stride, names, pt, imgsz


def make_dir(
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        save_txt=False,  # save results to *.txt
):
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    return save_dir
    

@smart_inference_mode()
def api(
        model, stride, names, pt, imgsz, save_dir, 
        HOST, PORT,device,
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)    
        
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        #update=False,  # update all models
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        vid_stride=1,  # video frame-rate stride
):
    source = str(source)
    print(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    int_to_names = {0:'stairs',
                    1:'pothoel',
                    2:'scooter',
                    3:'traffic cone',
                    4:'cyclist',
                    5:'person',
                    6:'others'}
    cls_frame = [0,0,0,0,0,0,0] # for beep
    cls_still = [0,0,0,0,0,0,0]

    send_threshold = 15 if device=='cpu' else 60 # CPU로 처리시 15, GPU로 처리시 150
    still_threshold = 4

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    """
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    """

    # Load model
    """
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    """

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        # webcam일 경우, view_img 인자와 상관없이 반드시 opencv로 window 띄운다.
        # 근데, 코드를 보니, url일 경우, webcam으로 친다.
        # 그래서, url은 view_img인자에 영향을 받도록 한다.
        view_img = check_imshow(warn=True)
        print('view_img:', view_img)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            """
            for _, row in pred.xyxy[0]:
                prediction = {
                    'detection': row['name'],
                    'confidence': row['confidence'],
                    'xmin': row['xmin'],
                    'ymin': row['ymin'],
                    'xmax': row['xmax'],
                    'ymax': row['ymax'],
                    "@timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
                }
                insertData(prediction)"""

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        #print(pred)
        # pred는 n x 6 tensor다.
        # n은 발견한 객체 수, 즉 bounding box의 수
        # 각각의 행에 있는 6개의 element는 0~3은 bounding box의 좌표다
        # 4는 confidence, 5는 클래스 번호


        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Define the path for the CSV file
        csv_path = save_dir / 'predictions.csv'

        # Create or append to the CSV file
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            objs = det[:, 5].unique()
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in objs:
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    cls_frame[int(c)] += 1
                    """
                    여기에 알림 정책을 넣으면 될 듯.
                    # beep when we detect the same class for 5 frames
                    if objs[int(c)] >= 5:
                        sd.Beep(1000,2000)

                    # 탐지 신호 보내기
                    1. 객체가 탐지 되었다면, objs += 1
                    2. 3연속 탐지 되었다면, send하고, objs 0으로 초기화. + flag를 1로 초기화
                    
                    # 사라짐 신호 보내기 
                    1. 해당 객체가 탐지 되지 않을 때 마다, flag *= 2
                    2. flag가 2*3이라면, send하고, flag 0으로 reset

                    # 탐지가 계속 되면, 20~30 frame 지속된다면, 다시 remind 신호 보내기
                    1. 객체가 탐지되었고, flag가 1이상이라면, frame += 1
                    2. frame == 20~30 이라면, remind 신호 보내고, frame 0으로 reset
                    """
                    #print(f"cls_frame: {cls_frame}")
                    #print(f"cls_still: {cls_still}")
                    
                    if cls_frame[int(c)] >= send_threshold and not cls_still[int(c)]:
                        data =  f"There is a new {str(names[int(c)][int(c)])}."
                        """
                        from gtts import gTTS
                        gtts = gTTS(data, lang='en')
                        filename = "C:/Users/Nabong/Desktop/Garodeong/Rasberry/tts.mp3"
                        gtts.save(filename)
                        with open(filename, 'rb') as file:
                            while True:
                                data = file.read(1024)
                                if not data:
                                    break        
                                client_socket.send(data)
                        """
                        client_socket.send(data.encode())
                        cls_frame[int(c)] = 0
                        cls_still[int(c)] = 1
                    elif cls_frame[int(c)] >= send_threshold and cls_still[int(c)]:
                        cls_still[int(c)] += 1
                        cls_frame[int(c)] = 0
                    elif cls_still[int(c)] >= still_threshold:
                        data =  f"There is still a {str(names[int(c)][int(c)])}."
                        client_socket.sensetd(data.encode())
                        cls_still[int(c)] = 1
                        cls_frame[int(c)] = 0
                # Write results
                #import pdb
                #pdb.set_trace()
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'
                    prediction = {
                    'detection': label,
                    'confidence': confidence_str,
                    'xmin': xyxy[0].item(),
                    'ymin': xyxy[1].item(),
                    'width': xyxy[2].item(),
                    'height': xyxy[3].item(),
                    "@timestamp": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
                    "HOST": HOST,
                    "GPU" : device
                        }
                    insertData(prediction)

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            
            all_obj = set(map(int, objs))
            disappear = {0,1,2,3,4,5,6} - all_obj
            for elem in disappear:
                if cls_still[elem]:
                    cls_frame[elem] -= 1
            for idx, count in enumerate(cls_frame):
                if count <= -send_threshold:
                    data = f"The {names[int(idx)][int(idx)]} disappears."
                    client_socket.send(data.encode())
                    cls_frame[idx] = 0
                    cls_still[idx] = 0
            # Stream results
            im0 = annotator.result()
            
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    #windows.append(p)
                    windows.append(f"{HOST}:{PORT}")
                    #cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.namedWindow(f"{HOST}:{PORT}", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    #cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                    cv2.resizeWindow(f"{HOST}:{PORT}", im0.shape[1], im0.shape[0])
                #cv2.imshow(str(p), im0)
                cv2.imshow(f"{HOST}:{PORT}", im0)
                cv2.waitKey(1)  # 1 millisecond
            
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    #if update:
    #    strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


#api(weights=f"{ROOT}/customdataset_epoch100.pt",view_img=True, source=0)
#api(weights=f"{ROOT}/yolov5s.pt",view_img=True, source=0, save_txt=True, device='')
if __name__=="__main__":    
    #model, stride, names, pt, imgsz = model_load(weights=f"{ROOT}/customdataset_epoch100.pt", device='')
    #model, stride, names, pt, imgsz = model_load(weights=f"{ROOT}/yolov5s.pt", device='')
    #save_dir = make_dir()
    #api(model=model,stride=stride,names=names,pt=pt,imgsz=imgsz,save_dir=save_dir,HOST='1.1.1.1', PORT=2,
    #view_img=True, source=0, save_txt=False)
    pass
