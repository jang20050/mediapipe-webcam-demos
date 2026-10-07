# MediaPipe 웹캠 비전 실습 강의 정리

> Google 공식 개발자 문서에서 모델을 받아 **웹캠 기반 실시간 손 / 제스처 / 얼굴 인식** 파이썬 프로그램을 만드는 실습

---

## 1. 강의 개요

| 항목 | 내용 |
|---|---|
| 주제 | MediaPipe Tasks(Python)로 웹캠 실시간 비전 AI 만들기 |
| 자료 출처 | **Google for Developers** (개발자 문서) – Google AI Edge / MediaPipe Solutions |
| 사용 도구 | Python, MediaPipe, OpenCV, **Claude Code + Claude in Chrome** |
| 결과물 | 손 랜드마크, 제스처 인식, 얼굴 랜드마크 웹캠 데모 3종 |

---

## 2. 작업 흐름 (핵심)

```
Google for Developers (개발자 문서)
        │   문서 URL만 전달
        ▼
Claude in Chrome (크롬 확장)
        │   Claude가 문서 페이지를 열고 모델(.task) 다운로드 링크를 찾아 직접 다운로드
        ▼
Claude Code
        │   모델을 프로젝트 폴더로 옮기고, 웹캠 기반 파이썬 코드를 작성 · 동작 검증
        ▼
GitHub 업로드
```

**포인트:** 사람은 문서 링크와 "라이브러리 받고 웹캠 코드 짜줘"라는 요청만 하면 된다.
**Claude in Chrome을 이용해서 Claude가 알아서 다운받고 코드까지 만들어준다.**

### 실제로 사용한 요청 예시

```
https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
해당하는 라이브러리 받고 파이썬 실행 코드 webcam 기반으로 짜줘 클로인크롬써서 다운받아
```

### Claude가 수행한 단계

1. 크롬에서 개발자 문서 페이지 열기
2. 페이지의 **Models** 섹션에서 `.task` 모델 다운로드 링크 찾기
3. 크롬으로 모델 다운로드 → `Downloads` 폴더 → 프로젝트 폴더로 복사
4. 설치된 라이브러리(`mediapipe`, `opencv-python`) 확인
5. 웹캠 실시간 처리 코드 작성
6. 모델 로딩 · 추론 스모크 테스트로 동작 확인

---

## 3. 폴더 구성

```
midea/
├── README.md                      # 강의 정리 (이 문서)
├── .gitignore
├── hand_landmarker.task           # 손 랜드마크 모델 (7.8MB)
├── hand_landmarker_webcam.py      # 손 랜드마크 웹캠 데모
├── gesture_recognizer.task        # 제스처 인식 모델 (8.4MB)
├── gesture_recognizer_webcam.py   # 제스처 인식 웹캠 데모
├── face_landmarker.task           # 얼굴 랜드마크 모델 (3.7MB)
└── face_landmarker_webcam.py      # 얼굴 랜드마크 웹캠 데모
```

---

## 4. 실습별 정리

### 4-1. Hand Landmarker (손 랜드마크)

- 문서: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
- 모델: `hand_landmarker.task`
- 코드: `hand_landmarker_webcam.py`
- 기능
  - 손 **21개 관절 점** 검출 (손목 0번 ~ 새끼손가락 끝 20번)
  - 최대 2개 손, **왼손 / 오른손 구분** + 신뢰도 표시
  - 관절을 선으로 연결해 손 뼈대 시각화

### 4-2. Gesture Recognizer (제스처 인식)

- 문서: https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer
- 모델: `gesture_recognizer.task`
- 코드: `gesture_recognizer_webcam.py`
- 기능
  - 손 랜드마크 + **제스처 분류**
  - 인식 가능한 기본 제스처 7종 (+ None)

| 제스처 | 의미 |
|---|---|
| `Closed_Fist` | 주먹 ✊ |
| `Open_Palm` | 손바닥 ✋ |
| `Pointing_Up` | 검지 위로 ☝️ |
| `Thumb_Up` | 엄지 척 👍 |
| `Thumb_Down` | 엄지 아래 👎 |
| `Victory` | 브이 ✌️ |
| `ILoveYou` | 사랑해 🤟 |

### 4-3. Face Landmarker (얼굴 랜드마크)

- 문서: https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker
- 모델: `face_landmarker.task`
- 코드: `face_landmarker_webcam.py`
- 기능
  - 얼굴 **478개 점** 메쉬 (홍채 포함)
  - 부위별 색 구분: 얼굴 외곽, 입술, 눈, 눈썹, 홍채
  - **블렌드쉐이프(표정 값)** 상위 5개 막대그래프
    - 예: `jawOpen`(입 벌림), `eyeBlinkLeft`(눈 깜빡임), `mouthSmileLeft`(미소)
  - `m` 키로 메쉬 표시 켜기 / 끄기

---

## 5. 공통 코드 구조

세 코드 모두 같은 패턴으로 되어 있다.

```python
# 1) 옵션 설정 – 모델 + 실행 모드(VIDEO)
options = vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)

# 2) 웹캠 열기
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 3) 프레임마다: BGR → RGB → mp.Image → 추론 → 그리기
with vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ok, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        draw_hands(frame, result)
        cv2.imshow("Hand Landmarker", frame)
```

| 단계 | 설명 |
|---|---|
| Running mode | `IMAGE`(사진 1장), `VIDEO`(프레임 + 타임스탬프), `LIVE_STREAM`(비동기 콜백) 중 **VIDEO** 사용 |
| 색 변환 | OpenCV는 BGR, MediaPipe는 RGB → `cv2.cvtColor` 필요 |
| 타임스탬프 | VIDEO 모드는 매 프레임 증가하는 ms 값 필요 |
| 좌표 | 결과 좌표는 0~1 정규화 값 → 화면 크기(w, h)를 곱해 픽셀로 변환 |
| 거울 모드 | `cv2.flip(frame, 1)`로 좌우 반전 |

### ⚠️ 팁: 한글 경로 문제

폴더 경로에 한글(예: `바탕 화면`)이 있으면 `model_asset_path`로 모델 로딩이 실패할 수 있다.
→ 파일을 바이트로 읽어서 `model_asset_buffer`로 전달하면 해결.

```python
mp_python.BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes())
```

---

## 6. 설치 및 실행

```bash
pip install mediapipe opencv-python

python hand_landmarker_webcam.py
python gesture_recognizer_webcam.py
python face_landmarker_webcam.py
```

| 키 | 동작 |
|---|---|
| `q` / `ESC` | 종료 |
| `m` | 얼굴 메쉬 토글 (face 전용) |

- 웹캠이 안 열리면 코드 상단의 `CAMERA_INDEX = 0`을 `1`로 변경

---

## 7. 정리

- **Google for Developers** 개발자 문서에는 모델 다운로드 링크와 사용법이 모두 있다.
- **Claude in Chrome**을 쓰면 Claude가 직접 문서를 열고 모델을 찾아 다운로드한다.
- **Claude Code**가 다운로드한 모델로 웹캠 코드를 작성하고 검증, GitHub 업로드까지 처리한다.
- MediaPipe Tasks는 *옵션 설정 → 모델 생성 → 프레임별 추론 → 결과 그리기*의 동일한 패턴이라, 다른 비전 태스크(포즈, 객체 검출 등)에도 그대로 응용할 수 있다.

## 모델 출처

[Google AI Edge – MediaPipe Solutions](https://developers.google.com/edge/mediapipe/solutions/vision) 공식 모델 (Apache 2.0)
