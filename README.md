# Garodeung


## 1. Computer Vision (Yolov5)

1. 가상환경 생성

```bash
conda create -n yolov5 python=3.8
'''

2. 가상환경 실행

```bash
conda activate yolov5
```

3. 패키지 설치

```bash
cd yolov5
pip install -r requriements.txt
```

4. 모델 실행

```bash
python detect.py --weights stairv2_epoch100.pt --view-img --source "이미지 경로" 웹캠 사용원할때, 0
```

실행 결과는 runs 폴더에 저장됨.
