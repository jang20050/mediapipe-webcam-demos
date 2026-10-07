# MediaPipe Webcam Demos

MediaPipe Tasks(Python)를 이용한 웹캠 실시간 비전 데모 모음.

| 스크립트 | 모델 | 기능 |
|---|---|---|
| `hand_landmarker_webcam.py` | `hand_landmarker.task` | 손 21개 랜드마크, 왼손/오른손 구분 (최대 2손) |
| `gesture_recognizer_webcam.py` | `gesture_recognizer.task` | 제스처 인식 (Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou) |
| `face_landmarker_webcam.py` | `face_landmarker.task` | 얼굴 478개 랜드마크 메쉬 + 블렌드쉐이프 상위 5개 |

## 설치

```bash
pip install mediapipe opencv-python
```

## 실행

```bash
python hand_landmarker_webcam.py
python gesture_recognizer_webcam.py
python face_landmarker_webcam.py
```

- `q` / `ESC`: 종료
- `m`: 얼굴 메쉬 표시 토글 (face_landmarker 전용)
- 웹캠이 안 열리면 각 스크립트의 `CAMERA_INDEX`를 변경

## 모델 출처

[Google AI Edge MediaPipe](https://developers.google.com/edge/mediapipe/solutions/vision) 공식 모델 (Apache 2.0).
