import argparse
import asyncio
import cv2
import websockets
import numpy as np
import base64

async def video_sending(websocket, path, video=None):
    cap = cv2.VideoCapture(video) if video else cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()  # 카메라에서 프레임을 읽습니다
            if not ret:
                print("Failed to grab frame")
                break

            _, buffer = cv2.imencode('.jpg', frame)  # 프레임을 JPEG 형식으로 인코딩합니다
            jpeg_as_text = base64.b64encode(buffer).decode()  # 인코딩된 데이터를 Base64로 변환합니다
            await websocket.send(jpeg_as_text)  # 프레임을 웹소켓을 통해 전송합니다

            await asyncio.sleep(0.03)  # 간단한 딜레이를 추가하여 프레임 속도를 제어합니다

    finally:
        cap.release()  # 비디오 캡처를 중지합니다

def main(video_path=None):
    async def handler(websocket, path):
        await video_sending(websocket, path, video=video_path)

    start_server = websockets.serve(video_sending, "localhost", 8765)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Send video stream to a web browser')
    parser.add_argument('--video', help='Path to video file (if not specified, use webcam)', default=None)
    args = parser.parse_args()
    main(args.video)