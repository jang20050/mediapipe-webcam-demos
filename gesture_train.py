"""커스텀 제스처 분류기 학습 (CSV 랜드마크 -> PyTorch MLP).

GUI(gesture_studio.py)의 [학습] 탭에서 사용하며, 단독 실행도 가능하다.
    python gesture_train.py

입력: gesture_data/landmarks.csv
출력: gesture_custom_model.pt
"""
import csv
import math

import numpy as np
import torch
from torch import nn

from gesture_common import CUSTOM_MODEL_PATH, DATA_PATH, NUM_FEATURES, NUM_LANDMARKS, normalize

MIN_SAMPLES_PER_CLASS = 10


class GestureMLP(nn.Module):
    def __init__(self, num_classes, hidden=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NUM_FEATURES, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_dataset():
    if not DATA_PATH.exists():
        raise ValueError("수집된 데이터가 없습니다. 먼저 [수집] 탭에서 데이터를 모으세요.")

    features, names = [], []
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            points = np.array(row[2:], dtype=np.float32).reshape(NUM_LANDMARKS, 3)
            features.append(normalize(points, row[1]))
            names.append(row[0])

    labels = sorted(set(names))
    if len(labels) < 2:
        raise ValueError("제스처가 2종류 이상 필요합니다. ('none' 제스처를 함께 모으는 것을 추천)")
    y = np.array([labels.index(n) for n in names], dtype=np.int64)
    few = [n for i, n in enumerate(labels) if (y == i).sum() < MIN_SAMPLES_PER_CLASS]
    if few:
        raise ValueError(f"데이터가 {MIN_SAMPLES_PER_CLASS}개 미만인 제스처가 있습니다: {', '.join(few)}")
    return np.stack(features), y, labels


def split_per_class(y, val_ratio, rng):
    """클래스별로 같은 비율로 train/val 분리."""
    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])
    return np.array(train_idx), np.array(val_idx)


def augment(x):
    """학습 데이터 증강: xy 평면 랜덤 회전(±15도) + 작은 노이즈."""
    pts = x.view(-1, NUM_LANDMARKS, 3).clone()
    theta = (torch.rand(pts.shape[0]) - 0.5) * 2 * math.radians(15)
    cos, sin = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
    px, py = pts[..., 0].clone(), pts[..., 1].clone()
    pts[..., 0] = cos * px - sin * py
    pts[..., 1] = sin * px + cos * py
    pts += torch.randn_like(pts) * 0.01
    return pts.view(-1, NUM_FEATURES)


def train_model(epochs=300, batch_size=64, lr=1e-3, val_ratio=0.2, seed=0,
                log=print, progress=None):
    """학습 후 CUSTOM_MODEL_PATH에 저장.

    log(str): 진행 메시지 출력 함수
    progress(epoch, epochs, val_acc): 매 epoch 호출 (GUI 진행바용)
    반환: {"labels", "best_acc", "class_acc"}
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    X, y, labels = load_dataset()
    log(f"샘플 {len(X)}개, 제스처 {len(labels)}개")
    for i, name in enumerate(labels):
        log(f"  {name}: {(y == i).sum()}")

    train_idx, val_idx = split_per_class(y, val_ratio, rng)
    X_train, y_train = torch.from_numpy(X[train_idx]), torch.from_numpy(y[train_idx])
    X_val, y_val = torch.from_numpy(X[val_idx]), torch.from_numpy(y[val_idx])

    model = GestureMLP(len(labels))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_state = -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0.0
        for i in range(0, len(perm), batch_size):
            batch = perm[i:i + batch_size]
            loss = loss_fn(model(augment(X_train[batch])), y_train[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)

        model.eval()
        with torch.no_grad():
            val_acc = (model(X_val).argmax(1) == y_val).float().mean().item()
        if val_acc >= best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 25 == 0 or epoch == 1:
            log(f"epoch {epoch:4d}  loss {total_loss / len(perm):.4f}  acc {val_acc:.3f}")
        if progress:
            progress(epoch, epochs, val_acc)

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(X_val).argmax(1)

    class_acc = {}
    log(f"\n최고 검증 정확도: {best_acc:.3f}")
    log("제스처별 검증 정확도:")
    for i, name in enumerate(labels):
        mask = y_val == i
        class_acc[name] = (pred[mask] == i).float().mean().item()
        log(f"  {name}: {class_acc[name]:.3f}  ({mask.sum().item()}개)")

    torch.save({"state_dict": best_state, "labels": labels}, CUSTOM_MODEL_PATH)
    log(f"\n모델 저장: {CUSTOM_MODEL_PATH.name}")
    return {"labels": labels, "best_acc": best_acc, "class_acc": class_acc}


if __name__ == "__main__":
    try:
        train_model()
    except ValueError as e:
        print(e)
