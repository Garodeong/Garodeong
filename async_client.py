import asyncio
import websockets
import base64
import cv2
import numpy as np
import logging
import time

async def receive_video():
    uri = "ws://localhost:8765/websockets/video"
    async with websockets.connect(uri) as websocket:
        logging.info("Connected to server")
        frame_number = 0
        try:
            while True:
                logging.info("Waiting for frame")
                await websocket.send("GET_FRAME")  # 프레임 요청
                logging.info("Frame received")
                base64_image = await websocket.recv()  # 인코딩된 이미지 수신
                image_bytes = base64.b64decode(base64_image)  # 이미지 디코딩
                np_array = np.frombuffer(image_bytes, np.uint8)  # Numpy 배열로 변환
                frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # 이미지 디코딩
                cv2.imshow("Video", frame)  # 이미지 표시
                
                # 이미지를 로그로 남깁니다.
                timestamp = int(time.time())  # 현재 시간의 타임스탬프를 가져옵니다.
                filename = f'frame_{frame_number}_{timestamp}.jpg'  # 파일 이름을 생성합니다.
                directory = './frames'  # 이미지를 저장할 폴더 이름을 지정합니다.
                cv2.imwrite(f'{directory}/{filename}', frame)  # 이미지를 저장합니다.
                logging.info(f'Frame {frame_number} saved as {filename}')
                frame_number += 1  # 프레임 번호를 증가시킵니다.
                
                if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
                    break
        finally:
            cv2.destroyAllWindows()

asyncio.get_event_loop().run_until_complete(receive_video())