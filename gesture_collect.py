"""커스텀 제스처 데이터 수집 (웹캠 -> 손 랜드마크 -> CSV).

사용법:
    python gesture_collect.py none heart ok rock

    - 인자로 준 제스처 이름이 숫자키 1, 2, 3 ... 에 순서대로 매핑된다.
    - 숫자키 : 수집할 제스처 선택
    - SPACE  : 녹화 시작/정지 (녹화 중에는 손이 보이는 매 프레임이 저장됨)
    - q/ESC  : 종료

데이터는 gesture_data/landmarks.csv 에 이어서(append) 저장되므로
여러 번 나눠서 수집해도 된다.
"""
import csv
import sys
import time
from collections import Counter

import cv2
import mediapipe as mp

from gesture_common import (
    DATA_PATH, NUM_LANDMARKS, create_hand_landmarker, draw_hand, landmarks_to_array,
)

CAMERA_INDEX = 0


def load_counts():
    if not DATA_PATH.exists():
        return Counter()
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        return Counter(row["label"] for row in csv.DictReader(f))


def main():
    labels = sys.argv[1:]
    if not labels or len(labels) > 9:
        print(__doc__)
        print("제스처 이름을 1~9개 입력하세요. 예) python gesture_collect.py none heart ok")
        sys.exit(1)

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_file = not DATA_PATH.exists()
    counts = load_counts()

    header = ["label", "handedness"] + [
        f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
    ]

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"웹캠({CAMERA_INDEX})을 열 수 없습니다.")

    current = 0
    recording = False
    start = time.monotonic()

    with DATA_PATH.open("a", newline="", encoding="utf-8") as f, \
            create_hand_landmarker(num_hands=1) as landmarker:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # 거울 모드 (추론 코드와 동일하게)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                points = landmarks_to_array(result.hand_landmarks[0])
                handedness = result.handedness[0][0].category_name
                draw_hand(frame, points, (0, 0, 255) if recording else (0, 255, 0))
                if recording:
                    # 원본 좌표를 저장하고 정규화는 학습 때 수행
                    writer.writerow([labels[current], handedness]
                                    + [f"{v:.6f}" for v in points.reshape(-1)])
                    counts[labels[current]] += 1

            # 상태 표시
            status = "REC" if recording else "PAUSE"
            color = (0, 0, 255) if recording else (200, 200, 200)
            cv2.putText(frame, f"[{status}] {current + 1}: {labels[current]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            for i, name in enumerate(labels):
                mark = ">" if i == current else " "
                cv2.putText(frame, f"{mark}{i + 1} {name}: {counts[name]}", (10, 65 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            cv2.putText(frame, "1-9: select  SPACE: rec on/off  q: quit",
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)

            cv2.imshow("Gesture Collect", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                recording = not recording
            elif ord("1") <= key <= ord("9") and key - ord("1") < len(labels):
                current = key - ord("1")
                recording = False  # 제스처를 바꾸면 녹화는 자동 정지

    cap.release()
    cv2.destroyAllWindows()
    print(f"저장 위치: {DATA_PATH}")
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
