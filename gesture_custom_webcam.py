"""학습한 커스텀 제스처 웹캠 실시간 추론.

실행: python gesture_custom_webcam.py
종료: q 또는 ESC
"""
import time

import cv2
import mediapipe as mp
import torch

from gesture_common import (
    CUSTOM_MODEL_PATH, create_hand_landmarker, draw_hand, landmarks_to_array, normalize,
)
from gesture_train import GestureMLP

CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.6  # 이보다 낮으면 "?"로 표시


def main():
    if not CUSTOM_MODEL_PATH.exists():
        raise SystemExit(f"모델이 없습니다: {CUSTOM_MODEL_PATH}\n먼저 gesture_train.py로 학습하세요.")
    ckpt = torch.load(CUSTOM_MODEL_PATH, weights_only=True)
    labels = ckpt["labels"]
    model = GestureMLP(len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("제스처:", labels)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")

    start = time.monotonic()
    prev = start
    with create_hand_landmarker(num_hands=2) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드 (수집 코드와 동일하게)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            for landmarks, handed in zip(result.hand_landmarks, result.handedness):
                points = landmarks_to_array(landmarks)
                handedness = handed[0].category_name
                x = torch.from_numpy(normalize(points, handedness)).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1)[0]
                score, idx = probs.max(0)
                name = labels[idx] if score >= CONFIDENCE_THRESHOLD else "?"

                pts = draw_hand(frame, points)
                x0 = min(p[0] for p in pts)
                y0 = min(p[1] for p in pts)
                cv2.putText(frame, f"{name} {score:.2f}", (x0, max(y0 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Custom Gesture", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
