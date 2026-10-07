"""커스텀 제스처 수집/학습/추론 공통 모듈.

수집(gesture_collect.py), 학습(gesture_train.py), 추론(gesture_custom_webcam.py)이
같은 전처리를 쓰도록 여기서 한 번만 정의한다.
"""
from pathlib import Path

import cv2
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BASE_DIR = Path(__file__).parent
HAND_MODEL_PATH = BASE_DIR / "hand_landmarker.task"
DATA_PATH = BASE_DIR / "gesture_data" / "landmarks.csv"
CUSTOM_MODEL_PATH = BASE_DIR / "gesture_custom_model.pt"

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3  # (x, y, z) x 21

# 21개 랜드마크 연결 (손목-엄지-검지-중지-약지-새끼)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def create_hand_landmarker(num_hands=1):
    # 경로에 한글이 있으면 model_asset_path 로딩이 실패할 수 있어 버퍼로 전달
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=HAND_MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def landmarks_to_array(landmarks):
    """MediaPipe 랜드마크 21개 -> (21, 3) numpy 배열."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def normalize(points, handedness):
    """손 위치/크기/좌우에 상관없이 같은 모양이면 같은 값이 되도록 정규화.

    1) 손목(0번)을 원점으로 이동
    2) 왼손은 x를 뒤집어 오른손 기준으로 통일
    3) 손목에서 가장 먼 점까지의 거리로 나눠 크기 통일
    """
    pts = points - points[0]
    if handedness == "Left":
        pts[:, 0] *= -1
    scale = np.linalg.norm(pts[:, :2], axis=1).max()
    if scale > 1e-6:
        pts /= scale
    return pts.reshape(-1)


def draw_hand(frame, points, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in points]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)
    return pts
