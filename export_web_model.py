"""학습한 PyTorch 제스처 모델을 웹용 JSON으로 변환하고 docs(웹) 폴더를 준비한다.

실행: python export_web_model.py   (run_web.py가 자동으로 호출하므로 보통 직접 실행할 필요 없음)

출력:
    docs/model.json            - 레이어 가중치 + 제스처 이름 + 검증용 샘플
    docs/hand_landmarker.task  - 브라우저에서 쓸 손 랜드마크 모델 (복사)
"""
import csv
import json
import shutil

import numpy as np
import torch
from torch import nn

from gesture_common import (
    BASE_DIR, CUSTOM_MODEL_PATH, DATA_PATH, HAND_MODEL_PATH, NUM_LANDMARKS, normalize,
)
from gesture_train import GestureMLP

WEB_DIR = BASE_DIR / "docs"  # GitHub Pages가 main 브랜치의 docs/ 폴더를 배포
WEB_MODEL_PATH = WEB_DIR / "model.json"
SAMPLES_PER_LABEL = 2  # 브라우저 계산이 파이썬과 같은지 확인하는 샘플 수


def export():
    if not CUSTOM_MODEL_PATH.exists():
        raise SystemExit(f"학습된 모델이 없습니다: {CUSTOM_MODEL_PATH.name}\n"
                         "gesture_studio.py에서 먼저 학습하세요.")

    ckpt = torch.load(CUSTOM_MODEL_PATH, weights_only=True)
    labels = ckpt["labels"]
    model = GestureMLP(len(labels))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Linear 레이어만 순서대로 (Dropout은 추론 시 무시, 마지막 레이어 외에는 ReLU)
    layers = [
        {"W": np.round(m.weight.detach().numpy(), 6).tolist(),
         "b": np.round(m.bias.detach().numpy(), 6).tolist()}
        for m in model.net if isinstance(m, nn.Linear)
    ]

    samples = []
    if DATA_PATH.exists():
        taken = {name: 0 for name in labels}
        with DATA_PATH.open(newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if taken.get(row[0], SAMPLES_PER_LABEL) >= SAMPLES_PER_LABEL:
                    continue
                points = np.array(row[2:], dtype=np.float32).reshape(NUM_LANDMARKS, 3)
                x = torch.from_numpy(normalize(points.copy(), row[1])).unsqueeze(0)
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1)[0]
                samples.append({"handedness": row[1], "points": points.tolist(),
                                "probs": probs.tolist()})
                taken[row[0]] += 1

    WEB_DIR.mkdir(exist_ok=True)
    WEB_MODEL_PATH.write_text(
        json.dumps({"labels": labels, "layers": layers, "samples": samples}),
        encoding="utf-8")

    web_hand_model = WEB_DIR / HAND_MODEL_PATH.name
    if not web_hand_model.exists() or web_hand_model.stat().st_size != HAND_MODEL_PATH.stat().st_size:
        shutil.copyfile(HAND_MODEL_PATH, web_hand_model)

    print(f"웹 모델 저장: {WEB_MODEL_PATH}  (제스처: {', '.join(labels)})")
    return WEB_MODEL_PATH


if __name__ == "__main__":
    export()
