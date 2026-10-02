# -*- coding: utf-8 -*-
"""
CCTV 인물 마스킹 도구 v2 (CPU 전용, 딥러닝 미사용)
 - OpenCV CSRT/KCF 추적 + 이상 감지(크기/위치 급변, 색 히스토그램, 화면 이탈)
 - 추적 놓치면 자동 정지 + 마지막 위치 계속 마스킹(HOLD)
 - 구간(시작점/끝점) 선택 후 해당 구간만 저장
 - 저장: 마스킹 대상 없으면 무인코딩 복사, 있으면 ffmpeg libx264(원본 비트레이트 맞춤)
필요 패키지: pip install opencv-contrib-python numpy pillow imageio-ffmpeg
"""
import os
import sys
import time
import math
import queue
import shutil
import tempfile
import threading
import subprocess
import colorsys
from contextlib import contextmanager

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import cv2
import numpy as np
from PIL import Image, ImageTk

# ===================== 튜닝 파라미터 =====================
TRACK_MAX_WIDTH = 640       # 추적은 이 폭 이하로 축소한 영상에서 수행 (CPU 부하 감소)
BOX_PAD_RATIO = 0.10        # 마스크를 추적 박스보다 사방 10% 크게
HOLD_EXTRA_PAD = 0.15       # 추적을 놓쳐 마지막 위치 유지(HOLD) 시 추가 여유
EDGE_SNAP_RATIO = 0.02      # 가장자리 근접 시 가장자리까지 마스크 확장
MAX_AREA_CHANGE = 1.8       # 프레임 간 박스 면적 변화 허용 배율
MAX_JUMP_RATIO = 0.5        # 프레임 간 중심 이동 허용치 (이전 박스 대각선 대비)
HIST_DIST_THRESHOLD = 0.65  # 색 히스토그램 바타차리야 거리 임계값
HIST_FAIL_FRAMES = 3        # 외형 불일치 연속 프레임 수
HIST_UPDATE_RATE = 0.05     # 기준 히스토그램 갱신 비율
MIN_VISIBLE_RATIO = 0.35    # 화면 내 비율이 이보다 작으면 화면 밖 이탈
MIN_BOX_SIZE = 8            # 최소 선택 크기 (원본 픽셀)
LIST_REFRESH_EVERY = 10     # 재생 중 인물 목록/타임라인 갱신 주기(프레임)

X264_PRESET = "veryfast"    # ultrafast(가장 빠름, 용량↑) ~ medium(느림, 용량↓)
QUALITY_OPTIONS = ["원본 비트레이트", "CRF 23 (표준)", "CRF 28 (용량 우선)"]
COPY_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".ts", ".m4v")
NO_WINDOW = 0x08000000 if os.name == "nt" else 0   # Windows에서 ffmpeg 콘솔창 숨김


# ===================== 유틸 =====================
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def find_ffmpeg():
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def make_color(i):
    h = (i * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def bgr_to_hex(c):
    b, g, r = c
    return f"#{r:02x}{g:02x}{b:02x}"


def fmt_time(sec):
    ms = int(max(0.0, sec) * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def create_tracker(name):
    name = name.upper()
    candidates = []
    for mod in (cv2, getattr(cv2, "legacy", None)):
        if mod is None:
            continue
        fn = getattr(mod, f"Tracker{name}_create", None)
        if fn is not None:
            candidates.append(fn)
    cls = getattr(cv2, f"Tracker{name}", None)
    if cls is not None and hasattr(cls, "create"):
        candidates.append(cls.create)
    for fn in candidates:
        try:
            return fn()
        except Exception:
            continue
    raise RuntimeError(
        f"{name} 추적기를 만들 수 없습니다. 'opencv-contrib-python'이 설치되어 있는지 확인하세요.")


def calc_hist(img, box):
    x, y, w, h = box
    H, W = img.shape[:2]
    x1 = int(max(0, x + w * 0.15))
    y1 = int(max(0, y + h * 0.15))
    x2 = int(min(W, x + w * 0.85))
    y2 = int(min(H, y + h * 0.85))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    hsv = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
    return hist


def mask_rect(b, W, H):
    x, y, w, h, hold = b
    pad = BOX_PAD_RATIO + (HOLD_EXTRA_PAD if hold else 0.0)
    x1, y1 = x - w * pad, y - h * pad
    x2, y2 = x + w * (1 + pad), y + h * (1 + pad)
    snap = max(6.0, EDGE_SNAP_RATIO * max(W, H))
    if x1 < snap:
        x1 = 0
    if y1 < snap:
        y1 = 0
    if x2 > W - snap:
        x2 = W
    if y2 > H - snap:
        y2 = H
    x1, y1 = int(max(0, x1)), int(max(0, y1))
    x2, y2 = int(min(W, x2)), int(min(H, y2))
    return x1, y1, x2, y2


def draw_mask(img, b, color, style, mask_on=True, border=True,
              thick=2, label=None, font_scale=0.6):
    H, W = img.shape[:2]
    x1, y1, x2, y2 = mask_rect(b, W, H)
    rw, rh = x2 - x1, y2 - y1
    if rw < 2 or rh < 2:
        return
    if mask_on:
        roi = img[y1:y2, x1:x2]
        if style == "모자이크":
            bs = max(6, min(rw, rh) // 8)
            small = cv2.resize(roi, (max(1, rw // bs), max(1, rh // bs)),
                               interpolation=cv2.INTER_LINEAR)
            img[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        elif style == "블러":
            k = max(31, (min(rw, rh) // 3) | 1)
            img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        else:
            img[y1:y2, x1:x2] = 0
    if border:
        cv2.rectangle(img, (x1, y1), (x2 - 1, y2 - 1), color, thick)
    if label:
        t = max(1, thick // 2)
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, t)
        ty = y1 + th + 4 if y1 < th + 6 else y1 - 4
        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + base), color, -1)
        cv2.putText(img, label, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), t, cv2.LINE_AA)


def person_segments(p):
    """추적된 프레임을 연속 구간 [(시작, 끝), ...]으로 압축"""
    ks = sorted(p.boxes)
    if not ks:
        return []
    segs = []
    s = prev = ks[0]
    for k in ks[1:]:
        if k != prev + 1:
            segs.append((s, prev))
            s = k
        prev = k
    segs.append((s, prev))
    return segs


# ===================== 데이터 =====================
class Person:
    def __init__(self, pid, color):
        self.pid = pid
        self.color = color
        self.boxes = {}                 # frame_idx -> (x, y, w, h, hold) [원본 좌표]
        self.status = "tracking"        # tracking / lost / ended
        self.tracker = None
        self.tracker_idx = None
        self.ref_hist = None
        self.hist_fail = 0
        self.last_reason = ""

    def range_str(self):
        if not self.boxes:
            return "-"
        return f"{min(self.boxes)}~{max(self.boxes)}"


# ===================== 타임라인(탐색 바) =====================
class Timeline(tk.Canvas):
    """클릭/드래그 탐색 + 선택 구간(파랑) + 인물별 추적 구간(색 막대) 표시"""
    PAD = 8

    def __init__(self, master, on_seek):
        super().__init__(master, height=30, width=200, bg="#1e1e1e",
                         highlightthickness=0, cursor="hand2")
        self.on_seek = on_seek
        self.total = 1
        self.cur = 0
        self.rin = None
        self.rout = None
        self.cover = []
        self.bind("<Configure>", lambda e: self.redraw())
        self.bind("<ButtonPress-1>", self._seek)
        self.bind("<B1-Motion>", self._seek)

    def _x(self, idx):
        w = max(1, self.winfo_width() - 2 * self.PAD)
        return self.PAD + w * idx / max(1, self.total - 1)

    def _idx(self, x):
        w = max(1, self.winfo_width() - 2 * self.PAD)
        r = min(max((x - self.PAD) / w, 0.0), 1.0)
        return int(round(r * (self.total - 1)))

    def _seek(self, e):
        self.on_seek(self._idx(e.x))

    def set_total(self, n):
        self.total = max(1, n)
        self.redraw()

    def set_range(self, a, b):
        self.rin, self.rout = a, b
        self.redraw()

    def set_cover(self, cover):
        self.cover = cover
        self.redraw()

    def set_pos(self, idx):
        self.cur = idx
        self._draw_head()

    def _draw_head(self):
        self.delete("head")
        x = self._x(self.cur)
        self.create_line(x, 0, x, self.winfo_height(), fill="#ffffff", width=2, tags="head")

    def redraw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4:
            return
        P = self.PAD
        if self.rin is not None or self.rout is not None:
            a = 0 if self.rin is None else self.rin
            b = self.total - 1 if self.rout is None else self.rout
            self.create_rectangle(self._x(a), 1, self._x(b), h - 1, fill="#1e3a8a", outline="")
            if self.rin is not None:
                self.create_line(self._x(a), 0, self._x(a), h, fill="#60a5fa", width=2)
            if self.rout is not None:
                self.create_line(self._x(b), 0, self._x(b), h, fill="#60a5fa", width=2)
        y1, y2 = 9, h - 9
        self.create_rectangle(P, y1, w - P, y2, fill="#3f3f46", outline="")
        if self.cover:
            lane = (y2 - y1) / len(self.cover)
            for i, (col, segs) in enumerate(self.cover):
                ly1 = y1 + i * lane
                ly2 = ly1 + max(1.0, lane)
                for s, e in segs:
                    xs = self._x(s)
                    self.create_rectangle(xs, ly1, max(self._x(e), xs + 1), ly2,
                                          fill=col, outline="")
        self._draw_head()


# ===================== 로그 창 =====================
class LogWindow:
    def __init__(self, master):
        self.win = tk.Toplevel(master)
        self.win.title("응답 로그")
        self.win.geometry("600x260+30+30")
        self.win.attributes("-topmost", True)
        self.text = ScrolledText(self.win, state="disabled", wrap="word",
                                 font=("Consolas", 9))
        self.text.pack(fill="both", expand=True)
        bar = ttk.Frame(self.win)
        bar.pack(fill="x")
        ttk.Button(bar, text="로그 지우기", command=self.clear).pack(side="right")
        self.win.protocol("WM_DELETE_WINDOW", self.win.withdraw)
        self.q = queue.Queue()
        self.win.after(100, self._poll)

    def write(self, s):
        self.q.put(s)

    def _poll(self):
        chunks = []
        try:
            while True:
                chunks.append(self.q.get_nowait())
        except queue.Empty:
            pass
        if chunks:
            self.text.configure(state="normal")
            self.text.insert("end", "".join(chunks))
            lines = int(self.text.index("end-1c").split(".")[0])
            if lines > 5000:
                self.text.delete("1.0", f"{lines - 4000}.0")
            self.text.see("end")
            self.text.configure(state="disabled")
        self.win.after(100, self._poll)

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def show(self):
        self.win.deiconify()
        self.win.lift()


class TeeStream:
    def __init__(self, orig, logwin):
        self.orig = orig
        self.logwin = logwin

    def write(self, s):
        if self.orig is not None:
            try:
                self.orig.write(s)
            except Exception:
                pass
        self.logwin.write(s)

    def flush(self):
        if self.orig is not None:
            try:
                self.orig.flush()
            except Exception:
                pass


# ===================== 메인 앱 =====================
class App:
    def __init__(self, root):
        self.root = root
        root.title("CCTV 인물 마스킹")
        self.logwin = LogWindow(root)
        sys.stdout = TeeStream(sys.__stdout__, self.logwin)
        sys.stderr = TeeStream(sys.__stderr__, self.logwin)

        self.cap = None
        self.path = None
        self.total = 0
        self.fps = 30.0
        self.W = self.H = 0
        self.src_bps = 0
        self.tscale = 1.0
        self.cur_idx = -1
        self.cur_frame = None
        self.prev_idx = -1
        self.prev_frame = None
        self.next_read_idx = 0
        self.playing = False
        self._after_id = None
        self.persons = []
        self.next_pid = 1
        self.disp = (1.0, 0, 0)
        self.photo = None
        self.img_item = None
        self.drag_start = None
        self.drag_rect = None
        self.exporting = False
        self.cancel_export = False
        self._frame_counter = 0
        self.range_in = None
        self.range_out = None
        self.ffmpeg = find_ffmpeg()

        self.tracker_name = tk.StringVar(value="CSRT")
        self.mask_style = tk.StringVar(value="모자이크")
        self.quality = tk.StringVar(value=QUALITY_OPTIONS[0])
        self.preview_mask = tk.BooleanVar(value=True)
        self.export_border = tk.BooleanVar(value=True)

        self._build_ui()
        self._bind_keys()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        log(f"OpenCV {cv2.__version__} / 프로그램 시작")
        if self.ffmpeg:
            log(f"ffmpeg: {self.ffmpeg}")
        else:
            log("경고: ffmpeg를 찾지 못했습니다. 'pip install imageio-ffmpeg' 설치를 권장합니다. "
                "(없으면 OpenCV mp4v로 저장되어 느리고 용량이 큽니다)")
        root.after(200, self.open_file_first)

    # ---------- UI ----------
    def _build_ui(self):
        root = self.root
        root.minsize(600, 420)
        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)

        side = ttk.Frame(main, padding=4)
        side.pack(side="right", fill="y")
        ttk.Label(side, text="추적 인물 목록").pack(anchor="w")
        self.listbox = tk.Listbox(side, width=30, height=10, exportselection=False,
                                  activestyle="none")
        self.listbox.pack(fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)
        ttk.Button(side, text="선택 해제 (새 인물 모드)", takefocus=False,
                   command=self.clear_selection).pack(fill="x", pady=(4, 0))
        ttk.Button(side, text="추적 종료 (현재 프레임까지)", takefocus=False,
                   command=self.end_selected).pack(fill="x", pady=(2, 0))
        ttk.Button(side, text="인물 삭제", takefocus=False,
                   command=self.delete_selected).pack(fill="x", pady=(2, 0))
        help_txt = ("[사용법]\n"
                    "- 일시정지 후 드래그: 새 인물 추가\n"
                    "- 목록에서 인물 선택 후 드래그:\n"
                    "  해당 인물 위치 재지정(재추적)\n"
                    "- Space: 재생/정지\n"
                    "- ←/→: 1프레임 이동\n"
                    "- Shift+←/→: 1초 이동\n"
                    "- Home/End: 처음/끝\n"
                    "- I / O: 구간 시작점 / 끝점\n"
                    "[탐색 바]\n"
                    "- 파란 배경: 저장 구간\n"
                    "- 색 막대: 인물별 추적 구간")
        ttk.Label(side, text=help_txt, justify="left",
                  foreground="#555").pack(anchor="w", pady=(8, 0))

        left = ttk.Frame(main)
        left.pack(side="left", fill="both", expand=True)

        row4 = ttk.Frame(left, padding=(4, 2))
        row4.pack(side="bottom", fill="x")
        row3 = ttk.Frame(left, padding=(4, 2))
        row3.pack(side="bottom", fill="x")
        row2 = ttk.Frame(left, padding=(4, 2))
        row2.pack(side="bottom", fill="x")
        row1 = ttk.Frame(left, padding=(4, 2))
        row1.pack(side="bottom", fill="x")
        seek = ttk.Frame(left, padding=(4, 2))
        seek.pack(side="bottom", fill="x")

        self.canvas = tk.Canvas(left, bg="black", highlightthickness=0,
                                width=800, height=450, cursor="crosshair")
        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # 탐색 바 (라벨을 먼저 pack해야 창을 줄여도 안 잘림)
        self.lbl_time = ttk.Label(seek, text="-", width=36, anchor="e")
        self.lbl_time.pack(side="right")
        self.timeline = Timeline(seek, self.on_timeline_seek)
        self.timeline.pack(side="left", fill="x", expand=True)

        def B(parent, text, cmd, w=None):
            b = ttk.Button(parent, text=text, command=cmd, takefocus=False, width=w)
            b.pack(side="left", padx=1)
            return b

        B(row1, "⏮ 처음", lambda: self.goto(0))
        B(row1, "-5초", lambda: self.step(-int(round(5 * self.fps)), pause=False))
        B(row1, "◀ -1F", lambda: self.step(-1))
        self.play_btn = B(row1, "▶ 재생", self.toggle_play, w=8)
        B(row1, "+1F ▶", lambda: self.step(1))
        B(row1, "+5초", lambda: self.step(int(round(5 * self.fps)), pause=False))
        B(row1, "끝 ⏭", lambda: self.goto(self.total - 1))

        ttk.Label(row2, text="구간:").pack(side="left")
        B(row2, "시작점 [I]", self.set_in)
        B(row2, "끝점 [O]", self.set_out)
        B(row2, "구간 시작으로", self.goto_in)
        B(row2, "구간 해제", self.clear_range)
        self.lbl_range = ttk.Label(row2, text="전체", foreground="#1d4ed8")
        self.lbl_range.pack(side="left", padx=6)

        ttk.Label(row3, text="화면 크기:").pack(side="left")
        for s in (0.25, 0.5, 1.0):
            B(row3, f"{s}x", lambda s=s: self.set_view_scale(s), w=5)
        ttk.Separator(row3, orient="vertical").pack(side="left", fill="y", padx=6)
        B(row3, "영상 열기", self.open_file)
        B(row3, "구간 저장", self.export)
        B(row3, "저장 취소", self.cancel_save)
        B(row3, "로그 창", self.logwin.show)

        ttk.Label(row4, text="추적기:").pack(side="left")
        cb = ttk.Combobox(row4, textvariable=self.tracker_name, width=6,
                          state="readonly", values=["CSRT", "KCF", "MIL"])
        cb.pack(side="left", padx=(0, 6))
        cb.bind("<<ComboboxSelected>>",
                lambda e: log(f"추적기 변경: {self.tracker_name.get()} (이후 새로 시작하는 추적부터 적용)"))
        ttk.Label(row4, text="마스킹:").pack(side="left")
        cb2 = ttk.Combobox(row4, textvariable=self.mask_style, width=8,
                           state="readonly", values=["모자이크", "블러", "검정 채움"])
        cb2.pack(side="left", padx=(0, 6))
        cb2.bind("<<ComboboxSelected>>", lambda e: self.render())
        ttk.Label(row4, text="저장 화질:").pack(side="left")
        ttk.Combobox(row4, textvariable=self.quality, width=16, state="readonly",
                     values=QUALITY_OPTIONS).pack(side="left", padx=(0, 6))
        ttk.Checkbutton(row4, text="미리보기 마스킹", variable=self.preview_mask,
                        command=self.render, takefocus=False).pack(side="left")
        ttk.Checkbutton(row4, text="저장 시 테두리", variable=self.export_border,
                        takefocus=False).pack(side="left", padx=(6, 0))

    def _bind_keys(self):
        r = self.root
        r.bind("<space>", lambda e: self.toggle_play())
        r.bind("<Left>", lambda e: self.step(-1))
        r.bind("<Right>", lambda e: self.step(1))
        r.bind("<Shift-Left>", lambda e: self.step(-int(round(self.fps)), pause=False))
        r.bind("<Shift-Right>", lambda e: self.step(int(round(self.fps)), pause=False))
        r.bind("<Home>", lambda e: self.goto(0))
        r.bind("<End>", lambda e: self.goto(self.total - 1))
        for k in ("i", "I"):
            r.bind(k, lambda e: self.set_in())
        for k in ("o", "O"):
            r.bind(k, lambda e: self.set_out())

    @contextmanager
    def dialog_guard(self):
        try:
            self.logwin.win.attributes("-topmost", False)
        except tk.TclError:
            pass
        try:
            yield
        finally:
            try:
                self.logwin.win.attributes("-topmost", True)
            except tk.TclError:
                pass

    # ---------- 파일 ----------
    def open_file_first(self):
        if not self.open_file():
            log("영상이 선택되지 않아 종료합니다.")
            self.root.after(300, self.on_close)

    def open_file(self):
        if self.exporting:
            log("저장 중에는 다른 영상을 열 수 없습니다.")
            return self.cap is not None
        self.pause()
        with self.dialog_guard():
            path = filedialog.askopenfilename(
                parent=self.root, title="재생할 CCTV 영상 선택",
                filetypes=[("동영상", "*.mp4 *.avi *.mkv *.mov *.wmv *.m4v *.mpg *.mpeg *.ts *.dav"),
                           ("모든 파일", "*.*")])
        if not path:
            return False
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            with self.dialog_guard():
                messagebox.showerror("오류", f"영상을 열 수 없습니다:\n{path}", parent=self.root)
            log(f"열기 실패: {path}")
            return False
        if self.cap is not None:
            self.cap.release()
        self.cap = cap
        self.path = path
        self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 < fps < 240 else 30.0
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.tscale = min(1.0, TRACK_MAX_WIDTH / max(1, self.W))

        # 원본 비트레이트 추정 (저장 시 같은 비트레이트로 인코딩 -> 용량 유사)
        bps = 0.0
        prop = getattr(cv2, "CAP_PROP_BITRATE", None)
        if prop is not None:
            try:
                v = cap.get(prop)
                if v and v > 0:
                    bps = v * 1000.0
            except Exception:
                pass
        if bps <= 0 and self.total > 0:
            dur = self.total / self.fps
            if dur > 0:
                bps = os.path.getsize(path) * 8 / dur
        self.src_bps = bps

        self.persons = []
        self.next_pid = 1
        self.cur_idx = -1
        self.cur_frame = None
        self.prev_idx = -1
        self.prev_frame = None
        self.next_read_idx = 0
        if self.total <= 0:
            self.total = 10 ** 9
        self.timeline.set_total(self.total)
        self.range_in = self.range_out = None
        self._range_changed(quiet=True)
        self.refresh_list()
        self.root.title(f"CCTV 인물 마스킹 - {os.path.basename(path)}")
        log(f"영상 열기: {path}")
        log(f"  해상도 {self.W}x{self.H}, {self.fps:.2f}fps, {self.total}프레임, "
            f"약 {self.src_bps / 1000:.0f}kbps, 추적 배율 {self.tscale:.2f}")
        self.goto(0)
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        for s in (1.0, 0.5, 0.25):
            if self.W * s + 280 <= sw * 0.9 and self.H * s + 250 <= sh * 0.85:
                break
        self.set_view_scale(s)
        return True

    # ---------- 프레임 읽기/표시 ----------
    def read_frame(self, idx):
        if self.cap is None or idx < 0 or idx >= self.total:
            return None
        sequential = (idx == self.next_read_idx)
        if not sequential:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.next_read_idx = -1
            if sequential and 0 < idx < self.total:
                self.total = idx
                self.timeline.set_total(self.total)
                if self.range_out is not None and self.range_out >= self.total:
                    self.range_out = self.total - 1
                    self._range_changed(quiet=True)
                log(f"실제 프레임 수 보정: {self.total}")
            return None
        self.next_read_idx = idx + 1
        return frame

    def to_small(self, frame):
        if self.tscale < 1.0:
            return cv2.resize(frame, None, fx=self.tscale, fy=self.tscale,
                              interpolation=cv2.INTER_AREA)
        return frame

    def show_frame(self, idx):
        frame = self.read_frame(idx)
        if frame is None:
            return None
        self.prev_idx, self.prev_frame = self.cur_idx, self.cur_frame
        self.cur_idx, self.cur_frame = idx, frame
        lost = self.process_tracking(idx, frame)
        self.timeline.set_pos(idx)
        self.lbl_time.configure(
            text=f"{idx + 1}/{self.total}F  {fmt_time(idx / self.fps)} / {fmt_time(self.total / self.fps)}")
        self.render()
        self._frame_counter += 1
        if lost or not self.playing or self._frame_counter % LIST_REFRESH_EVERY == 0:
            self.refresh_list()
        return lost

    def render(self):
        if self.cur_frame is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return
        s = min(cw / self.W, ch / self.H)
        dw, dh = max(1, int(self.W * s)), max(1, int(self.H * s))
        ox, oy = (cw - dw) // 2, (ch - dh) // 2
        self.disp = (s, ox, oy)
        img = self.cur_frame.copy()
        thick = max(1, int(round(2 / s)))
        fscale = max(0.5, 0.55 / s)
        for p in self.persons:
            b = p.boxes.get(self.cur_idx)
            if b is None:
                continue
            lab = f"ID{p.pid}" + (" HOLD" if b[4] else "")
            draw_mask(img, b, p.color, self.mask_style.get(),
                      mask_on=self.preview_mask.get(), border=True,
                      thick=thick, label=lab, font_scale=fscale)
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        disp = cv2.resize(img, (dw, dh), interpolation=interp)
        self.photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)))
        if self.img_item is None:
            self.img_item = self.canvas.create_image(ox, oy, anchor="nw", image=self.photo)
        else:
            self.canvas.coords(self.img_item, ox, oy)
            self.canvas.itemconfigure(self.img_item, image=self.photo)
        self.canvas.tag_lower(self.img_item)

    def canvas_to_frame(self, cx, cy):
        s, ox, oy = self.disp
        fx = (cx - ox) / s
        fy = (cy - oy) / s
        return min(max(fx, 0), self.W), min(max(fy, 0), self.H)

    def set_view_scale(self, s):
        if self.cap is None:
            return
        try:
            self.root.state("normal")
        except tk.TclError:
            pass
        self.root.update_idletasks()
        extra_w = self.root.winfo_width() - self.canvas.winfo_width()
        extra_h = self.root.winfo_height() - self.canvas.winfo_height()
        cw, ch = int(self.W * s), int(self.H * s)
        self.root.geometry(f"{cw + extra_w}x{ch + extra_h}")
        log(f"화면 크기 {s}x 적용 ({cw}x{ch})")

    # ---------- 구간 ----------
    def get_range(self):
        a = 0 if self.range_in is None else self.range_in
        b = self.total - 1 if self.range_out is None else self.range_out
        return a, min(b, self.total - 1)

    def _range_changed(self, quiet=False):
        if self.range_in is None and self.range_out is None:
            txt = "전체"
        else:
            a, b = self.get_range()
            txt = (f"{a}F ~ {b}F  ({fmt_time(a / self.fps)} ~ {fmt_time((b + 1) / self.fps)}, "
                   f"{b - a + 1}프레임)")
        self.lbl_range.configure(text=txt)
        self.timeline.set_range(self.range_in, self.range_out)
        if not quiet:
            log(f"구간: {txt}")

    def set_in(self):
        if self.cur_frame is None:
            return
        self.range_in = self.cur_idx
        if self.range_out is not None and self.range_out < self.range_in:
            self.range_out = None
            log("끝점이 시작점보다 앞이어서 끝점을 해제했습니다.")
        self._range_changed()

    def set_out(self):
        if self.cur_frame is None:
            return
        self.range_out = self.cur_idx
        if self.range_in is not None and self.range_in > self.range_out:
            self.range_in = None
            log("시작점이 끝점보다 뒤여서 시작점을 해제했습니다.")
        self._range_changed()

    def clear_range(self):
        self.range_in = self.range_out = None
        self._range_changed()

    def goto_in(self):
        self.pause()
        self.goto(self.get_range()[0])

    # ---------- 추적 ----------
    def tracker_init(self, p, small, box, idx):
        ts = self.tscale
        x, y, w, h = box
        tb = (int(round(x * ts)), int(round(y * ts)),
              max(4, int(round(w * ts))), max(4, int(round(h * ts))))
        p.tracker = create_tracker(self.tracker_name.get())
        p.tracker.init(small, tb)
        p.tracker_idx = idx
        p.hist_fail = 0
        if p.ref_hist is None:
            p.ref_hist = calc_hist(small, tb)

    def tracker_step(self, p, small, f):
        ts = self.tscale
        try:
            ok, tb = p.tracker.update(small)
        except cv2.error as e:
            return None, f"추적기 오류: {e}"
        p.tracker_idx = f
        if not ok:
            return None, "추적기가 대상을 찾지 못함"
        px, py, pw, ph = p.boxes[f - 1][:4]
        x, y, w, h = [float(v) / ts for v in tb]
        if w < 2 or h < 2:
            return None, "박스 크기 비정상"
        vx1, vy1 = max(0.0, x), max(0.0, y)
        vx2, vy2 = min(float(self.W), x + w), min(float(self.H), y + h)
        vis = max(0.0, vx2 - vx1) * max(0.0, vy2 - vy1) / (w * h)
        if vis < MIN_VISIBLE_RATIO:
            return None, "화면 밖으로 이탈"
        ratio = (w * h) / max(1.0, pw * ph)
        if ratio > MAX_AREA_CHANGE or ratio < 1.0 / MAX_AREA_CHANGE:
            return None, f"박스 크기 급변 (x{ratio:.2f})"
        jump = math.hypot((x + w / 2) - (px + pw / 2), (y + h / 2) - (py + ph / 2)) \
            / max(1.0, math.hypot(pw, ph))
        if jump > MAX_JUMP_RATIO:
            return None, f"위치 급변 ({jump:.2f})"
        hist = calc_hist(small, tb)
        if hist is not None and p.ref_hist is not None:
            d = cv2.compareHist(p.ref_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if d > HIST_DIST_THRESHOLD:
                p.hist_fail += 1
                if p.hist_fail >= HIST_FAIL_FRAMES:
                    return None, f"외형 불일치 (히스토그램 거리 {d:.2f}) - 배경/다른 사람으로 드리프트 의심"
            else:
                p.hist_fail = 0
                p.ref_hist = cv2.addWeighted(p.ref_hist, 1 - HIST_UPDATE_RATE,
                                             hist, HIST_UPDATE_RATE, 0)
        return (vx1, vy1, vx2 - vx1, vy2 - vy1), ""

    def near_edge(self, box):
        x, y, w, h = box[:4]
        m = max(6.0, EDGE_SNAP_RATIO * 2 * max(self.W, self.H))
        return x < m or y < m or x + w > self.W - m or y + h > self.H - m

    def process_tracking(self, f, frame):
        lost = []
        small = None
        for p in self.persons:
            if p.status == "ended" or f in p.boxes or (f - 1) not in p.boxes:
                continue
            prev = p.boxes[f - 1]
            if p.status == "lost":
                p.boxes[f] = (prev[0], prev[1], prev[2], prev[3], True)
                continue
            if small is None:
                small = self.to_small(frame)
            if p.tracker is None or p.tracker_idx != f - 1:
                prev_img = self.prev_frame if self.prev_idx == f - 1 else self.read_frame(f - 1)
                if prev_img is None:
                    continue
                try:
                    self.tracker_init(p, self.to_small(prev_img), prev[:4], f - 1)
                except Exception as e:
                    log(f"ID{p.pid} 추적기 초기화 실패: {e}")
                    continue
            box, reason = self.tracker_step(p, small, f)
            if box is None:
                p.status = "lost"
                if self.near_edge(prev):
                    reason += " / 가장자리 근처: 화면 밖으로 나갔을 가능성"
                p.last_reason = reason
                p.boxes[f] = (prev[0], prev[1], prev[2], prev[3], True)
                lost.append(p)
            else:
                p.boxes[f] = (box[0], box[1], box[2], box[3], False)
        return lost

    def handle_lost(self, lost):
        if not lost:
            return
        self.pause()
        for p in lost:
            log(f"[추적 놓침] ID{p.pid} @ {self.cur_idx}F: {p.last_reason}")
        log("  -> 마지막 위치를 확대해 계속 마스킹(HOLD)합니다. 목록에서 선택된 인물을 드래그해 "
            "위치를 다시 지정하거나, 화면 밖으로 나간 경우 '추적 종료'를 누르세요.")
        self.refresh_list()
        idx = self.persons.index(lost[0])
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(idx)
        self.listbox.see(idx)

    # ---------- 재생/탐색 ----------
    def toggle_play(self):
        if self.playing:
            self.pause()
            log(f"일시정지 @ {self.cur_idx}F")
        else:
            self.play()

    def play(self):
        if self.cap is None or self.playing:
            return
        if self.cur_idx >= self.total - 1:
            log("영상 끝입니다.")
            return
        if self.range_out is not None and self.cur_idx >= self.range_out:
            log("구간 끝점입니다. '구간 시작으로'를 누르거나 다른 위치로 이동 후 재생하세요.")
            return
        lost = [p for p in self.persons if p.status == "lost"]
        if lost:
            log("주의: 놓친 상태 인물 " + ", ".join(f"ID{p.pid}" for p in lost) +
                " - 마지막 위치 마스크(HOLD)를 유지한 채 재생합니다.")
        self.playing = True
        self.play_btn.configure(text="⏸ 정지")
        log(f"재생 @ {self.cur_idx}F")
        self._tick()

    def pause(self):
        self.playing = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None
        if hasattr(self, "play_btn"):
            self.play_btn.configure(text="▶ 재생")

    def _tick(self):
        self._after_id = None
        if not self.playing:
            return
        t0 = time.perf_counter()
        nxt = self.cur_idx + 1
        if self.range_out is not None and nxt > self.range_out:
            self.pause()
            self.refresh_list()
            log(f"구간 끝점({self.range_out}F)에 도달했습니다.")
            return
        if nxt >= self.total:
            self.pause()
            self.refresh_list()
            log("영상 끝에 도달했습니다.")
            return
        lost = self.show_frame(nxt)
        if lost is None:
            self.pause()
            self.refresh_list()
            log("영상 끝에 도달했습니다 (더 읽을 프레임 없음).")
            return
        if lost:
            self.handle_lost(lost)
            return
        elapsed = (time.perf_counter() - t0) * 1000
        delay = max(1, int(1000 / self.fps - elapsed))
        self._after_id = self.root.after(delay, self._tick)

    def step(self, n, pause=True):
        if self.cap is None:
            return
        if pause:
            self.pause()
        self.goto(self.cur_idx + n)

    def goto(self, idx):
        if self.cap is None:
            return
        idx = max(0, min(idx, self.total - 1))
        if idx == self.cur_idx:
            return
        if idx > self.cur_idx + 1:
            gap = [p for p in self.persons if p.status == "tracking" and p.boxes
                   and min(p.boxes) <= self.cur_idx and max(p.boxes) < idx - 1]
            if gap:
                log("주의: 추적되지 않은 구간을 건너뜁니다 (" +
                    ", ".join(f"ID{p.pid}" for p in gap) +
                    "). 마스킹을 완성하려면 해당 구간도 재생해야 합니다.")
        lost = self.show_frame(idx)
        if lost is None:
            log(f"{idx}F 프레임을 읽을 수 없습니다.")
            return
        self.handle_lost(lost)

    def on_timeline_seek(self, idx):
        if self.cap is None:
            return
        if self.playing:
            self.pause()
        self.goto(idx)

    # ---------- 마우스 선택 ----------
    def selected_person(self):
        sel = self.listbox.curselection()
        if not sel or sel[0] >= len(self.persons):
            return None
        return self.persons[sel[0]]

    def on_press(self, e):
        if self.cur_frame is None:
            return
        if self.playing:
            log("재생 중에는 선택할 수 없습니다. 일시정지(Space) 후 드래그하세요.")
            return
        self.drag_start = (e.x, e.y)
        p = self.selected_person()
        color = bgr_to_hex(p.color if p else make_color(self.next_pid))
        self.drag_rect = self.canvas.create_rectangle(e.x, e.y, e.x, e.y,
                                                      outline=color, width=2, dash=(4, 2))

    def on_drag(self, e):
        if self.drag_start and self.drag_rect:
            x0, y0 = self.drag_start
            self.canvas.coords(self.drag_rect, x0, y0, e.x, e.y)

    def on_release(self, e):
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        self.drag_start = None
        if self.drag_rect:
            self.canvas.delete(self.drag_rect)
            self.drag_rect = None
        fx0, fy0 = self.canvas_to_frame(x0, y0)
        fx1, fy1 = self.canvas_to_frame(e.x, e.y)
        x, y = min(fx0, fx1), min(fy0, fy1)
        w, h = abs(fx1 - fx0), abs(fy1 - fy0)
        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
            log("선택 영역이 너무 작습니다. 인물 전체를 드래그하세요.")
            return
        box = (x, y, w, h)
        p = self.selected_person()
        try:
            if p is None:
                p = Person(self.next_pid, make_color(self.next_pid))
                self.next_pid += 1
                self.assign_box(p, box)
                self.persons.append(p)
                log(f"ID{p.pid} 추가 @ {self.cur_idx}F  박스=({int(x)},{int(y)},{int(w)},{int(h)})")
            else:
                self.assign_box(p, box)
                log(f"ID{p.pid} 위치 재지정 @ {self.cur_idx}F -> 이후 결과를 지우고 다시 추적합니다.")
                self.listbox.selection_clear(0, "end")
        except Exception as ex:
            log(f"추적 시작 실패: {ex}")
            return
        self.refresh_list()
        self.render()

    def assign_box(self, p, box):
        f = self.cur_idx
        for k in [k for k in p.boxes if k >= f]:
            del p.boxes[k]
        p.boxes[f] = (box[0], box[1], box[2], box[3], False)
        p.status = "tracking"
        p.ref_hist = None
        p.last_reason = ""
        self.tracker_init(p, self.to_small(self.cur_frame), box, f)

    # ---------- 인물 목록 ----------
    def refresh_list(self):
        sel = self.selected_person()
        self.listbox.delete(0, "end")
        names = {"tracking": "추적", "lost": "놓침", "ended": "종료"}
        for i, p in enumerate(self.persons):
            self.listbox.insert("end", f"ID{p.pid} [{names[p.status]}] {p.range_str()}F")
            self.listbox.itemconfig(i, foreground=bgr_to_hex(p.color),
                                    background="#222" if p.status != "lost" else "#552222")
        self.listbox.configure(bg="#222")
        if sel in self.persons:
            self.listbox.selection_set(self.persons.index(sel))
        self.timeline.set_cover([(bgr_to_hex(p.color), person_segments(p)) for p in self.persons])

    def on_list_select(self, e):
        p = self.selected_person()
        if p:
            log(f"ID{p.pid} 선택됨 - 일시정지 상태에서 드래그하면 현재 프레임 위치로 재지정됩니다.")

    def clear_selection(self):
        self.listbox.selection_clear(0, "end")
        log("새 인물 모드: 드래그하면 새 인물이 추가됩니다.")

    def end_selected(self):
        p = self.selected_person()
        if p is None:
            log("목록에서 인물을 먼저 선택하세요.")
            return
        f = self.cur_idx
        for k in [k for k in p.boxes if k > f]:
            del p.boxes[k]
        p.status = "ended"
        p.tracker = None
        log(f"ID{p.pid} 추적 종료 ({f}F까지 마스킹)")
        self.listbox.selection_clear(0, "end")
        self.refresh_list()
        self.render()

    def delete_selected(self):
        p = self.selected_person()
        if p is None:
            log("목록에서 인물을 먼저 선택하세요.")
            return
        with self.dialog_guard():
            ok = messagebox.askyesno("삭제", f"ID{p.pid}의 추적 결과를 모두 삭제할까요?",
                                     parent=self.root)
        if ok:
            self.persons.remove(p)
            log(f"ID{p.pid} 삭제")
            self.refresh_list()
            self.render()

    # ---------- 저장 ----------
    def _quality_args(self):
        q = self.quality.get()
        if q.startswith("CRF"):
            return ["-crf", q.split()[1]]
        bps = int(self.src_bps * 0.95)          # 오디오 몫을 대략 제외
        if bps < 100000:
            log("원본 비트레이트를 알 수 없어 CRF 23으로 저장합니다.")
            return ["-crf", "23"]
        return ["-b:v", str(bps), "-maxrate", str(int(bps * 1.5)), "-bufsize", str(int(bps * 2))]

    def cancel_save(self):
        if self.exporting:
            self.cancel_export = True
            log("저장 취소 요청...")
        else:
            log("진행 중인 저장이 없습니다.")

    def export(self):
        if self.cap is None:
            return
        if self.exporting:
            log("이미 저장 중입니다.")
            return
        self.pause()
        a, b = self.get_range()
        n = b - a + 1

        snap = []
        for p in self.persons:
            bx = {k: v for k, v in p.boxes.items() if a <= k <= b}
            if bx:
                snap.append((p.pid, p.color, bx))

        incomplete = [p for p in self.persons
                      if p.status != "ended" and p.boxes and min(p.boxes) <= b and max(p.boxes) < b]
        if incomplete:
            msg = (f"다음 인물은 구간 끝({b}F)까지 추적되지 않았습니다:\n" +
                   "\n".join(f" ID{p.pid}: {p.range_str()}F (상태: {p.status})" for p in incomplete) +
                   "\n\n추적되지 않은 프레임은 마스킹되지 않습니다. 그래도 저장할까요?")
            with self.dialog_guard():
                if not messagebox.askyesno("확인", msg, parent=self.root):
                    return

        mode = "encode"
        if not snap:
            if self.ffmpeg:
                with self.dialog_guard():
                    ans = messagebox.askyesnocancel(
                        "저장 방식",
                        "구간 안에 마스킹할 인물이 없습니다.\n\n"
                        "[예] 인코딩 없이 원본 그대로 잘라내기 (매우 빠름, 화질/용량 원본과 동일)\n"
                        "     ※ 키프레임 단위로 잘려 시작이 조금 앞당겨질 수 있음\n"
                        "[아니오] 재인코딩 (프레임 단위로 정확히 자름)",
                        parent=self.root)
                if ans is None:
                    return
                mode = "copy" if ans else "encode"
            else:
                log("ffmpeg가 없어 무인코딩 복사를 할 수 없습니다. OpenCV로 재인코딩합니다.")

        base = os.path.splitext(os.path.basename(self.path))[0]
        src_ext = os.path.splitext(self.path)[1].lower()
        if mode == "copy":
            ext = src_ext if src_ext in COPY_EXTS else ".mkv"
            suffix = "_cut"
        else:
            ext = ".mp4"
            suffix = "_masked" if snap else "_cut"
        if a > 0 or b < self.total - 1:
            suffix += f"_{a}-{b}"
        with self.dialog_guard():
            out = filedialog.asksaveasfilename(
                parent=self.root, title="구간 저장", defaultextension=ext,
                initialdir=os.path.dirname(self.path), initialfile=f"{base}{suffix}{ext}",
                filetypes=[("MP4", "*.mp4"), ("MKV", "*.mkv"), ("AVI", "*.avi"), ("모든 파일", "*.*")])
        if not out:
            return
        if os.path.abspath(out) == os.path.abspath(self.path):
            log("원본 파일에 덮어쓸 수 없습니다.")
            return

        self.exporting = True
        self.cancel_export = False
        log(f"저장 구간: {a}F ~ {b}F ({n}프레임, {fmt_time(n / self.fps)}), 방식: "
            + ("무인코딩 복사" if mode == "copy" else
               ("ffmpeg libx264" if self.ffmpeg else "OpenCV mp4v")))
        if mode == "copy":
            target, args = self._copy_worker, (self.path, out, a, b, self.fps, self.total)
        elif self.ffmpeg:
            target = self._ffmpeg_encode_worker
            args = (self.path, out, snap, self.mask_style.get(), self.export_border.get(),
                    self.fps, self.W, self.H, a, b, self._quality_args())
        else:
            target = self._cv_encode_worker
            args = (self.path, out, snap, self.mask_style.get(), self.export_border.get(),
                    self.fps, self.W, self.H, a, b)
        threading.Thread(target=target, args=args, daemon=True).start()

    def _finish_log(self, out, t0, frames=None):
        try:
            size = fmt_size(os.path.getsize(out))
        except OSError:
            size = "?"
        extra = f"{frames}프레임, " if frames is not None else ""
        log(f"저장 완료: {out} ({extra}{size}, {time.time() - t0:.1f}초)")

    def _copy_worker(self, src, out, a, b, fps, total):
        """마스킹 없음: 스트림 복사 (디코딩/인코딩 없음)"""
        t0 = time.time()
        try:
            cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
            if a > 0:
                cmd += ["-ss", f"{a / fps:.3f}"]
            cmd += ["-i", src]
            if b < total - 1:
                cmd += ["-t", f"{(b - a + 1) / fps:.3f}"]
            cmd += ["-map", "0:v:0", "-map", "0:a?", "-c", "copy",
                    "-avoid_negative_ts", "make_zero", out]
            log("ffmpeg 무인코딩 복사 실행 중...")
            r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               creationflags=NO_WINDOW)
            if r.returncode != 0:
                err = r.stderr.decode(errors="replace")[-800:]
                log(f"복사 실패 (코드 {r.returncode}): {err}")
                log("  -> 원본 코덱이 출력 형식과 맞지 않을 수 있습니다. 확장자를 .mkv로 바꿔 다시 시도하세요.")
            else:
                self._finish_log(out, t0)
        except Exception as e:
            log(f"복사 중 오류: {e}")
        finally:
            self.exporting = False

    def _ffmpeg_encode_worker(self, src, out, snap, style, border, fps, W, H, a, b, qargs):
        """OpenCV로 디코딩+마스킹 -> 원시 프레임을 ffmpeg(libx264)에 파이프로 전달"""
        n = b - a + 1
        t0 = time.time()
        cap = cv2.VideoCapture(src)
        errf = tempfile.TemporaryFile()
        proc = None
        done = 0
        try:
            if a > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            cmd = [self.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
                   "-framerate", f"{fps:.6f}", "-i", "-"]
            if a > 0:
                cmd += ["-ss", f"{a / fps:.3f}"]
            cmd += ["-t", f"{n / fps:.3f}", "-i", src,
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-preset", X264_PRESET] + qargs + [
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k"]
            if out.lower().endswith((".mp4", ".m4v", ".mov")):
                cmd += ["-movflags", "+faststart"]
            cmd.append(out)
            log("인코딩 옵션: libx264 " + X264_PRESET + " " + " ".join(qargs))
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                    stderr=errf, creationflags=NO_WINDOW)
            thick = max(2, W // 400)
            last_pct = -1
            broken = False
            for i in range(n):
                if self.cancel_export:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != W or frame.shape[0] != H:
                    frame = cv2.resize(frame, (W, H))
                idx = a + i
                for pid, color, boxes in snap:
                    bx = boxes.get(idx)
                    if bx is not None:
                        draw_mask(frame, bx, color, style, mask_on=True,
                                  border=border, thick=thick, label=None)
                try:
                    proc.stdin.write(np.ascontiguousarray(frame).tobytes())
                except (BrokenPipeError, OSError):
                    broken = True
                    break
                done += 1
                pct = int(done * 100 / n)
                if pct // 5 != last_pct // 5:
                    last_pct = pct
                    log(f"  저장 진행 {pct}% ({done}/{n}F)")
            try:
                proc.stdin.close()
            except Exception:
                pass
            ret = proc.wait()
            if self.cancel_export:
                log("저장이 취소되었습니다.")
                try:
                    os.remove(out)
                except OSError:
                    pass
            elif ret != 0 or broken:
                errf.seek(0)
                err = errf.read().decode(errors="replace")[-800:]
                log(f"ffmpeg 오류 (코드 {ret}): {err}")
            else:
                self._finish_log(out, t0, done)
        except Exception as e:
            log(f"저장 중 오류: {e}")
            if proc is not None:
                try:
                    proc.kill()
                except Exception:
                    pass
        finally:
            cap.release()
            errf.close()
            self.exporting = False

    def _cv_encode_worker(self, src, out, snap, style, border, fps, W, H, a, b):
        """ffmpeg가 없을 때의 대체 저장 (느리고 용량 큼, 소리 없음)"""
        n = b - a + 1
        t0 = time.time()
        cap = cv2.VideoCapture(src)
        writer = None
        done = 0
        try:
            if a > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            fourcc = cv2.VideoWriter_fourcc(*("XVID" if out.lower().endswith(".avi") else "mp4v"))
            writer = cv2.VideoWriter(out, fourcc, fps, (W, H))
            if not writer.isOpened():
                log("VideoWriter를 열 수 없습니다 (코덱/경로 확인).")
                return
            thick = max(2, W // 400)
            last_pct = -1
            for i in range(n):
                if self.cancel_export:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != W or frame.shape[0] != H:
                    frame = cv2.resize(frame, (W, H))
                idx = a + i
                for pid, color, boxes in snap:
                    bx = boxes.get(idx)
                    if bx is not None:
                        draw_mask(frame, bx, color, style, mask_on=True,
                                  border=border, thick=thick, label=None)
                writer.write(frame)
                done += 1
                pct = int(done * 100 / n)
                if pct // 5 != last_pct // 5:
                    last_pct = pct
                    log(f"  저장 진행 {pct}% ({done}/{n}F)")
            writer.release()
            writer = None
            if self.cancel_export:
                log("저장이 취소되었습니다.")
                try:
                    os.remove(out)
                except OSError:
                    pass
            else:
                self._finish_log(out, t0, done)
                log("  (OpenCV 저장: 소리 없음)")
        except Exception as e:
            log(f"저장 중 오류: {e}")
        finally:
            if writer is not None:
                writer.release()
            cap.release()
            self.exporting = False

    # ---------- 종료 ----------
    def on_close(self):
        if self.exporting:
            with self.dialog_guard():
                if not messagebox.askyesno("종료", "저장 중입니다. 그래도 종료할까요?",
                                           parent=self.root):
                    return
            self.cancel_export = True
        self.pause()
        if self.cap is not None:
            self.cap.release()
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        self.root.destroy()


def main():
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
