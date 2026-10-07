"""MediaPipe Face Landmarker - 웹캠 실시간 얼굴 랜드마크(478점) + 표정 블렌드쉐이프.

실행: python face_landmarker_webcam.py
종료: q 또는 ESC
키:   m 메쉬(테셀레이션) 표시 켜기/끄기
"""
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = Path(__file__).with_name("face_landmarker.task")
CAMERA_INDEX = 0
TOP_BLENDSHAPES = 5

FLC = vision.FaceLandmarksConnections
# (연결 목록, BGR 색상)
CONTOUR_STYLES = [
    (FLC.FACE_LANDMARKS_FACE_OVAL, (200, 200, 200)),
    (FLC.FACE_LANDMARKS_LIPS, (80, 80, 255)),
    (FLC.FACE_LANDMARKS_LEFT_EYE, (80, 255, 80)),
    (FLC.FACE_LANDMARKS_RIGHT_EYE, (80, 255, 80)),
    (FLC.FACE_LANDMARKS_LEFT_EYEBROW, (80, 255, 255)),
    (FLC.FACE_LANDMARKS_RIGHT_EYEBROW, (80, 255, 255)),
    (FLC.FACE_LANDMARKS_LEFT_IRIS, (255, 200, 0)),
    (FLC.FACE_LANDMARKS_RIGHT_IRIS, (255, 200, 0)),
]


def draw_faces(frame, result, show_mesh):
    h, w = frame.shape[:2]
    for landmarks in result.face_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        if show_mesh:
            for c in FLC.FACE_LANDMARKS_TESSELATION:
                cv2.line(frame, pts[c.start], pts[c.end], (90, 90, 90), 1)
        for connections, color in CONTOUR_STYLES:
            for c in connections:
                cv2.line(frame, pts[c.start], pts[c.end], color, 1)


def draw_blendshapes(frame, result):
    if not result.face_blendshapes:
        return
    # 첫 번째 얼굴의 상위 블렌드쉐이프 (예: eyeBlinkLeft, jawOpen, mouthSmileLeft ...)
    top = sorted(result.face_blendshapes[0], key=lambda c: c.score, reverse=True)
    y = 60
    for cat in top[:TOP_BLENDSHAPES]:
        bar = int(cat.score * 150)
        cv2.rectangle(frame, (10, y - 12), (10 + bar, y + 2), (0, 200, 255), -1)
        cv2.putText(frame, f"{cat.category_name} {cat.score:.2f}", (170, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22


def main():
    # 경로에 한글이 있으면 model_asset_path 로딩이 실패할 수 있어 버퍼로 전달
    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")

    show_mesh = True
    start = time.monotonic()
    prev = start
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            draw_faces(frame, result, show_mesh)
            draw_blendshapes(frame, result)

            now = time.monotonic()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            cv2.putText(frame, f"FPS {fps:.1f}  Faces {len(result.face_landmarks)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Face Landmarker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                show_mesh = not show_mesh

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
