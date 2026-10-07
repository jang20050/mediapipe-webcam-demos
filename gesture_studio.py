"""커스텀 제스처 스튜디오 (GUI) - 수집 / 학습 / 추론을 한 화면에서.

실행: python gesture_studio.py

[수집] 제스처 추가 -> 목록에서 선택 -> 녹화 (Space) -> 3초 후 자동 녹화, 목표 개수 도달 시 자동 정지
[학습] 학습 시작 -> 진행바/로그 확인 -> gesture_custom_model.pt 저장
[추론] 탭을 열면 최신 모델을 자동으로 불러와 실시간 인식
"""
import csv
import queue
import threading
import time
import tkinter as tk
from collections import Counter
from tkinter import messagebox, ttk

import cv2
import mediapipe as mp
import torch

from gesture_common import (
    CUSTOM_MODEL_PATH, DATA_PATH, NUM_LANDMARKS, create_hand_landmarker, draw_hand,
    landmarks_to_array, normalize,
)
from gesture_train import GestureMLP, train_model

FONT = "Malgun Gothic"
VIEW_W = 640
COUNTDOWN_SEC = 3
TAB_COLLECT, TAB_TRAIN, TAB_INFER = range(3)
HEADER = ["label", "handedness"] + [
    f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in ("x", "y", "z")
]


def read_rows():
    if not DATA_PATH.exists():
        return []
    with DATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        return list(reader)


def write_rows(rows, append):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    need_header = not append or not DATA_PATH.exists()
    with DATA_PATH.open("a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(HEADER)
        writer.writerows(rows)


class GestureStudio:
    def __init__(self, root):
        self.root = root
        self.counts = Counter(row[0] for row in read_rows())
        self.labels = sorted(self.counts)

        # 수집 상태
        self.buffer = []
        self.rec_label = None
        self.recording = False
        self.countdown_end = None

        # 추론 상태
        self.model = None
        self.model_labels = []
        self.model_mtime = None

        # 학습 스레드 -> UI 메시지
        self.messages = queue.Queue()
        self.training = False

        self.landmarker = create_hand_landmarker(num_hands=2)
        self.start = time.monotonic()
        self.prev = self.start
        self.cap = None
        self.photo = None

        self.build_ui()
        self.open_camera(0)
        self.refresh_list()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.tick()

    # ------------------------------------------------------------------ UI
    def build_ui(self):
        self.root.title("Gesture Studio - 커스텀 제스처 수집 / 학습 / 추론")
        style = ttk.Style()
        style.configure(".", font=(FONT, 10))
        style.configure("Rec.TButton", font=(FONT, 12, "bold"), padding=8)
        style.configure("Big.TLabel", font=(FONT, 28, "bold"))
        # Space는 녹화 단축키로 쓰므로 버튼 기본 동작(포커스된 버튼 클릭)을 끈다
        self.root.unbind_class("TButton", "<Key-space>")

        main = ttk.Frame(self.root, padding=8)
        main.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(main, width=VIEW_W, height=480, bg="black", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="n")
        self.canvas_img = self.canvas.create_image(0, 0, anchor="nw")

        side = ttk.Frame(main, width=300)
        side.grid(row=0, column=1, sticky="ns", padx=(10, 0))

        cam_row = ttk.Frame(side)
        cam_row.pack(fill="x", pady=(0, 6))
        ttk.Label(cam_row, text="카메라 번호").pack(side="left")
        self.cam_var = tk.StringVar(value="0")
        cam_box = ttk.Combobox(cam_row, textvariable=self.cam_var, values=["0", "1", "2", "3"],
                               width=4, state="readonly")
        cam_box.pack(side="left", padx=6)
        cam_box.bind("<<ComboboxSelected>>", lambda e: self.open_camera(int(self.cam_var.get())))

        self.tabs = ttk.Notebook(side)
        self.tabs.pack(fill="both", expand=True)
        self.build_collect_tab()
        self.build_train_tab()
        self.build_infer_tab()

        self.status = ttk.Label(main, text="", anchor="w")
        self.status.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self.root.bind_all("<space>", self.on_space)

    def build_collect_tab(self):
        tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(tab, text="  1. 수집  ")

        ttk.Label(tab, text="제스처 목록 (선택 후 녹화)").pack(anchor="w")
        self.listbox = tk.Listbox(tab, height=9, font=(FONT, 11), exportselection=False,
                                  activestyle="none")
        self.listbox.pack(fill="x", pady=(2, 6))

        add_row = ttk.Frame(tab)
        add_row.pack(fill="x")
        self.name_var = tk.StringVar()
        entry = ttk.Entry(add_row, textvariable=self.name_var)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda e: self.add_label())
        ttk.Button(add_row, text="추가", command=self.add_label).pack(side="left", padx=(4, 0))

        ttk.Button(tab, text="선택한 제스처 데이터 삭제", command=self.delete_label).pack(
            fill="x", pady=(4, 10))

        target_row = ttk.Frame(tab)
        target_row.pack(fill="x")
        ttk.Label(target_row, text="목표 개수").pack(side="left")
        self.target_var = tk.IntVar(value=300)
        ttk.Spinbox(target_row, from_=50, to=5000, increment=50, width=7,
                    textvariable=self.target_var).pack(side="left", padx=6)

        self.rec_btn = ttk.Button(tab, text="● 녹화 시작  (Space)", style="Rec.TButton",
                                  command=self.toggle_record)
        self.rec_btn.pack(fill="x", pady=10)

        self.collect_info = ttk.Label(tab, text="", font=(FONT, 11, "bold"))
        self.collect_info.pack(anchor="w")
        ttk.Label(tab, wraplength=270, foreground="#666", text=(
            "· 녹화를 누르면 3초 뒤 시작, 목표 개수가 되면 자동 정지\n"
            "· 손 각도·거리·위치를 조금씩 바꿔가며 녹화\n"
            "· 'none'(아무 동작 아님) 제스처도 꼭 함께 수집"
        )).pack(anchor="w", pady=(8, 0))

    def build_train_tab(self):
        tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(tab, text="  2. 학습  ")

        row = ttk.Frame(tab)
        row.pack(fill="x")
        ttk.Label(row, text="Epochs").pack(side="left")
        self.epochs_var = tk.IntVar(value=300)
        ttk.Spinbox(row, from_=50, to=2000, increment=50, width=7,
                    textvariable=self.epochs_var).pack(side="left", padx=6)
        self.train_btn = ttk.Button(row, text="학습 시작", command=self.start_training)
        self.train_btn.pack(side="right")

        self.progress = ttk.Progressbar(tab, maximum=100)
        self.progress.pack(fill="x", pady=8)

        self.log = tk.Text(tab, height=20, width=40, font=(FONT, 9), wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

    def build_infer_tab(self):
        tab = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(tab, text="  3. 추론  ")

        ttk.Label(tab, text="인식 결과").pack(anchor="w")
        self.result_label = ttk.Label(tab, text="-", style="Big.TLabel")
        self.result_label.pack(anchor="w", pady=(0, 8))

        ttk.Label(tab, text="제스처별 확률").pack(anchor="w")
        self.prob_frame = ttk.Frame(tab)
        self.prob_frame.pack(fill="x", pady=(2, 0))
        self.prob_bars = []  # [(Progressbar, 값 Label)] - model_labels 순서

        th_row = ttk.Frame(tab)
        th_row.pack(fill="x", pady=(12, 0))
        ttk.Label(th_row, text="최소 확률").pack(side="left")
        self.threshold_var = tk.DoubleVar(value=0.6)
        th_value = ttk.Label(th_row, text="0.60", width=5)
        ttk.Scale(th_row, from_=0.3, to=0.95, variable=self.threshold_var,
                  command=lambda v: th_value.config(text=f"{float(v):.2f}")).pack(
            side="left", fill="x", expand=True, padx=6)
        th_value.pack(side="left")
        ttk.Label(tab, wraplength=270, foreground="#666",
                  text="이 값보다 확률이 낮으면 '?'로 표시합니다.").pack(anchor="w", pady=(4, 0))

        self.model_info = ttk.Label(tab, text="", wraplength=270, foreground="#666")
        self.model_info.pack(anchor="w", pady=(12, 0))

    # ------------------------------------------------------------------ 카메라 루프
    def open_camera(self, index):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if self.cap.isOpened():
            self.set_status(f"카메라 {index} 연결됨")
        else:
            self.set_status(f"카메라 {index}를 열 수 없습니다. 다른 번호를 선택하세요.")

    def tick(self):
        self.process_messages()
        ok, frame = self.cap.read() if self.cap else (False, None)
        if ok:
            frame = cv2.flip(frame, 1)  # 거울 모드
            h, w = frame.shape[:2]
            if w != VIEW_W:
                frame = cv2.resize(frame, (VIEW_W, int(h * VIEW_W / w)))
            if int(self.canvas["height"]) != frame.shape[0]:
                self.canvas.config(height=frame.shape[0])

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - self.start) * 1000)
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            overlays = []  # (x, y, text, color, size, anchor) - 한글 표시를 위해 Canvas에 그림
            tab = self.tabs.index("current")
            if tab == TAB_COLLECT:
                self.handle_collect(frame, result, overlays)
            elif tab == TAB_INFER:
                self.handle_infer(frame, result, overlays)
            else:
                for landmarks in result.hand_landmarks:
                    draw_hand(frame, landmarks_to_array(landmarks))

            now = time.monotonic()
            fps = 1.0 / max(now - self.prev, 1e-6)
            self.prev = now
            overlays.append((VIEW_W - 10, 10, f"FPS {fps:.0f}", "#ffff00", 11, "ne"))
            self.show(frame, overlays)
        self.root.after(5, self.tick)

    def show(self, frame, overlays):
        ok, buf = cv2.imencode(".ppm", frame)
        self.photo = tk.PhotoImage(data=buf.tobytes())
        self.canvas.itemconfig(self.canvas_img, image=self.photo)
        self.canvas.delete("overlay")
        for x, y, text, color, size, anchor in overlays:
            font = (FONT, size, "bold")
            self.canvas.create_text(x + 2, y + 2, text=text, fill="black", font=font,
                                    anchor=anchor, tags="overlay")
            self.canvas.create_text(x, y, text=text, fill=color, font=font,
                                    anchor=anchor, tags="overlay")

    # ------------------------------------------------------------------ 수집
    def refresh_list(self):
        selected = self.selected_label()
        self.listbox.delete(0, "end")
        for name in self.labels:
            self.listbox.insert("end", f"{name}   ({self.counts[name]}개)")
        if selected in self.labels:
            self.listbox.selection_set(self.labels.index(selected))

    def selected_label(self):
        sel = self.listbox.curselection()
        return self.labels[sel[0]] if sel and sel[0] < len(self.labels) else None

    def add_label(self):
        name = self.name_var.get().strip()
        if not name:
            return
        if name not in self.labels:
            self.labels.append(name)
        self.name_var.set("")
        self.refresh_list()
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(self.labels.index(name))
        self.set_status(f"'{name}' 선택됨 - 녹화 버튼 또는 Space로 수집을 시작하세요.")

    def delete_label(self):
        name = self.selected_label()
        if not name or self.recording or self.countdown_end:
            return
        if self.counts[name] and not messagebox.askyesno(
                "데이터 삭제", f"'{name}' 데이터 {self.counts[name]}개를 삭제할까요?\n되돌릴 수 없습니다."):
            return
        write_rows([row for row in read_rows() if row[0] != name], append=False)
        self.listbox.selection_clear(0, "end")
        self.labels.remove(name)
        self.counts.pop(name, None)
        self.refresh_list()
        self.set_status(f"'{name}' 삭제됨")

    def toggle_record(self):
        if self.recording or self.countdown_end:
            self.stop_record()
            return
        name = self.selected_label()
        if not name:
            messagebox.showinfo("제스처 선택", "목록에서 제스처를 선택하거나 새로 추가하세요.")
            return
        if self.counts[name] >= self.get_int(self.target_var, 300):
            messagebox.showinfo("목표 도달", f"'{name}'은 이미 목표 개수에 도달했습니다.\n목표 개수를 늘려주세요.")
            return
        self.rec_label = name
        self.countdown_end = time.monotonic() + COUNTDOWN_SEC
        self.rec_btn.config(text="■ 정지  (Space)")
        self.listbox.config(state="disabled")

    def stop_record(self):
        self.recording = False
        self.countdown_end = None
        if self.buffer:
            write_rows(self.buffer, append=True)
            self.set_status(f"'{self.rec_label}' {len(self.buffer)}개 저장됨 "
                            f"(총 {self.counts[self.rec_label]}개)")
            self.buffer = []
        self.rec_btn.config(text="● 녹화 시작  (Space)")
        self.listbox.config(state="normal")
        self.refresh_list()

    def handle_collect(self, frame, result, overlays):
        now = time.monotonic()
        if self.countdown_end:
            remaining = self.countdown_end - now
            if remaining > 0:
                overlays.append((VIEW_W // 2, frame.shape[0] // 2, str(int(remaining) + 1),
                                 "#ffffff", 72, "center"))
                overlays.append((VIEW_W // 2, frame.shape[0] // 2 + 70,
                                 f"'{self.rec_label}' 준비", "#ffffff", 16, "center"))
            else:
                self.countdown_end = None
                self.recording = True

        color = (0, 0, 255) if self.recording else (0, 255, 0)
        for landmarks in result.hand_landmarks:
            draw_hand(frame, landmarks_to_array(landmarks), color)

        target = self.get_int(self.target_var, 300)
        if self.recording:
            if result.hand_landmarks:
                points = landmarks_to_array(result.hand_landmarks[0])
                handedness = result.handedness[0][0].category_name
                self.buffer.append([self.rec_label, handedness]
                                   + [f"{v:.6f}" for v in points.reshape(-1)])
                self.counts[self.rec_label] += 1
            else:
                overlays.append((VIEW_W // 2, 40, "손이 보이지 않습니다", "#ff6060", 16, "n"))
            overlays.append((10, 10, f"● REC  {self.rec_label}  {self.counts[self.rec_label]}/{target}",
                             "#ff4040", 14, "nw"))
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), (0, 0, 255), 4)
            if self.counts[self.rec_label] >= target:
                self.stop_record()
                self.set_status(f"'{self.rec_label}' 목표 {target}개 도달 - 자동 정지")

        name = self.rec_label if (self.recording or self.countdown_end) else self.selected_label()
        self.collect_info.config(text=f"{name}: {self.counts[name]} / {target}" if name else "")

    def on_space(self, event):
        if isinstance(event.widget, (tk.Entry, ttk.Entry, tk.Spinbox, ttk.Spinbox, tk.Text)):
            return None
        if self.tabs.index("current") == TAB_COLLECT:
            self.toggle_record()
            return "break"
        return None

    # ------------------------------------------------------------------ 학습
    def start_training(self):
        if self.training:
            return
        if self.recording or self.countdown_end:
            self.stop_record()
        self.training = True
        self.train_btn.config(state="disabled")
        self.progress["value"] = 0
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")
        self.set_status("학습 중...")
        epochs = self.get_int(self.epochs_var, 300)
        threading.Thread(target=self.train_worker, args=(epochs,), daemon=True).start()

    def train_worker(self, epochs):
        try:
            result = train_model(
                epochs=epochs,
                log=lambda s: self.messages.put(("log", s)),
                progress=lambda e, n, acc: self.messages.put(("progress", e * 100 / n)),
            )
            self.messages.put(("done", result))
        except Exception as e:  # noqa: BLE001 - UI에 그대로 보여준다
            self.messages.put(("error", str(e)))

    def process_messages(self):
        while True:
            try:
                kind, payload = self.messages.get_nowait()
            except queue.Empty:
                return
            if kind == "log":
                self.log.config(state="normal")
                self.log.insert("end", payload + "\n")
                self.log.see("end")
                self.log.config(state="disabled")
            elif kind == "progress":
                self.progress["value"] = payload
            elif kind == "done":
                self.training = False
                self.train_btn.config(state="normal")
                self.model_mtime = None  # 추론 탭에서 새 모델 다시 로드
                self.set_status(f"학습 완료 - 검증 정확도 {payload['best_acc']:.1%}. "
                                "[3. 추론] 탭에서 확인하세요.")
            elif kind == "error":
                self.training = False
                self.train_btn.config(state="normal")
                self.set_status("학습 실패")
                messagebox.showerror("학습 실패", payload)

    # ------------------------------------------------------------------ 추론
    def ensure_model(self):
        if not CUSTOM_MODEL_PATH.exists():
            self.model = None
            return False
        mtime = CUSTOM_MODEL_PATH.stat().st_mtime
        if mtime != self.model_mtime:
            ckpt = torch.load(CUSTOM_MODEL_PATH, weights_only=True)
            self.model_labels = ckpt["labels"]
            self.model = GestureMLP(len(self.model_labels))
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.model_mtime = mtime
            self.model_info.config(text="모델 제스처: " + ", ".join(self.model_labels))
            self.build_prob_rows()
        return True

    def build_prob_rows(self):
        for child in self.prob_frame.winfo_children():
            child.destroy()
        self.prob_bars = []
        for i, name in enumerate(self.model_labels):
            ttk.Label(self.prob_frame, text=name, width=10).grid(row=i, column=0, sticky="w")
            bar = ttk.Progressbar(self.prob_frame, maximum=1.0, length=130)
            bar.grid(row=i, column=1, padx=4, pady=1)
            value = ttk.Label(self.prob_frame, text="", width=5)
            value.grid(row=i, column=2, sticky="w")
            self.prob_bars.append((bar, value))

    def set_probs(self, probs=None):
        for i, (bar, value) in enumerate(self.prob_bars):
            p = float(probs[i]) if probs is not None else 0.0
            bar["value"] = p
            value.config(text=f"{p:.2f}" if probs is not None else "")

    def handle_infer(self, frame, result, overlays):
        if not self.ensure_model():
            overlays.append((VIEW_W // 2, 40, "학습된 모델이 없습니다. [2. 학습]을 먼저 진행하세요.",
                             "#ff6060", 13, "n"))
            self.result_label.config(text="-")
            return

        threshold = self.threshold_var.get()
        first = None
        for landmarks, handed in zip(result.hand_landmarks, result.handedness):
            points = landmarks_to_array(landmarks)
            x = torch.from_numpy(normalize(points, handed[0].category_name)).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(self.model(x), dim=1)[0]
            score, idx = probs.max(0)
            name = self.model_labels[idx] if score >= threshold else "?"

            pts = draw_hand(frame, points)
            x0 = min(p[0] for p in pts)
            y0 = min(p[1] for p in pts)
            overlays.append((x0, max(y0 - 8, 30), f"{name} {score:.2f}", "#00ffff", 18, "sw"))
            if first is None:
                first = (name, probs)

        if first is None:
            self.result_label.config(text="-")
            self.set_probs(None)
            return
        name, probs = first
        self.result_label.config(text=name)
        self.set_probs(probs)

    # ------------------------------------------------------------------ 기타
    @staticmethod
    def get_int(var, default):
        """Spinbox에 숫자가 아닌 값이 입력돼도 죽지 않도록."""
        try:
            return max(1, int(var.get()))
        except (tk.TclError, ValueError):
            return default

    def set_status(self, text):
        self.status.config(text=text)

    def on_close(self):
        if self.recording or self.countdown_end:
            self.stop_record()
        if self.cap:
            self.cap.release()
        self.landmarker.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.resizable(False, False)
    GestureStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
