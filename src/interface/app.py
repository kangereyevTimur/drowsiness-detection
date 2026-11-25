from collections import deque
import csv
import subprocess
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk

import customtkinter as ctk
import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.session_storage import SessionStorage
from src.detection.drowsiness_detector import DrowsinessEvaluator
from src.detection.eye_detector import (
    BASE_CNN_PATH,
    DEVICE,
    FINETUNED_CNN_PATH,
    EyeCNN,
)
from src.detection.multi_person import MultiPersonTracker

LANDMARKER_PATH = ROOT / "models" / "face_landmarker.task"
DIAG_DIR = ROOT / "results" / "diagnostics"
SNAPSHOT_DIR = ROOT / "results" / "snapshots"
REPORTS_DIR = ROOT / "results" / "reports"
CONFUSION_MATRIX_PNG = ROOT / "results" / "cnn_finetuned" / "webcam_test_confusion_matrices.png"

DIAG_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_SECONDS = 3.5

# Палитра Obsidian & Electric Lavender (в стиле Web Tactics)
BG_VOID = "#06060A"
BG_SURFACE = "#0D0D14"
BG_CARD = "#12121C"
BG_ELEVATED = "#181826"
BORDER_HAIRLINE = "#222234"
BORDER_VIOLET = "#4C3585"
TEXT_WHITE = "#FFFFFF"
TEXT_LAVENDER = "#C4B5FD"
TEXT_MUTED = "#8E92A4"
TEXT_DIM = "#5E6278"
ACCENT_VIOLET = "#8B5CF6"
ACCENT_VIOLET_HOVER = "#7C3AED"
STATUS_GREEN = "#22C55E"
STATUS_AMBER = "#F59E0B"
STATUS_RED = "#EF4444"

FACE_MESH_KEYPOINTS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    1, 4, 5, 195, 197, 6, 168, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
]

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class DrowsinessApp(ctk.CTk):
    """
    Интерфейс системы определения сонливости и усталости в тёмном минималистичном стиле
    Obsidian & Electric Lavender (без смайликов, с выносом детальных метрик за кнопку).
    """

    def __init__(self):
        super().__init__()

        self.title("DROWSINESS AI — Fatigue Monitoring System")
        self.geometry("1340x860")
        self.minsize(1160, 760)
        self.configure(fg_color=BG_VOID)

        # Состояние работы
        self.is_running = False
        self.is_video_file = False
        self.video_source_name = "ОЖИДАНИЕ"
        self.cap = None
        self.landmarker = None
        self.storage = SessionStorage()

        self.multi_tracker = MultiPersonTracker(window_seconds=60.0)
        self.max_faces = 4

        self.session_start_dt = None
        self.app_start_mono = 0.0
        self.calibration_start_mono = 0.0
        self.last_timestamp_ms = -1
        self.last_sound_time = 0.0
        self.last_voice_time = 0.0
        self.perclos_samples = []
        self.last_status = None
        self.last_annotated_frame = None
        self.metrics_visible = False

        # Буферы для графика
        self.chart_history_len = 120
        self.score_history = deque([0.0] * self.chart_history_len, maxlen=self.chart_history_len)
        self.perclos_history = deque([0.0] * self.chart_history_len, maxlen=self.chart_history_len)
        self.fps_history = deque(maxlen=25)
        self.last_frame_mono = time.monotonic()

        self.csv_file = None
        self.csv_writer = None
        self.current_log_path = None

        # Модели
        self.available_models = {}
        if FINETUNED_CNN_PATH.exists():
            self.available_models["EyeCNN Fine-Tuned"] = FINETUNED_CNN_PATH
        if BASE_CNN_PATH.exists():
            self.available_models["EyeCNN Baseline"] = BASE_CNN_PATH

        self.current_model_path = FINETUNED_CNN_PATH if FINETUNED_CNN_PATH.exists() else BASE_CNN_PATH
        self.model = EyeCNN().to(DEVICE)
        self._load_cnn_weights(self.current_model_path)

        # Переключатели отрисовки на видео
        self.show_boxes_var = ctk.BooleanVar(value=True)
        self.show_mesh_var = ctk.BooleanVar(value=False)

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(120, self._bring_to_front)

    def _bring_to_front(self):
        try:
            self.lift()
            self.attributes("-topmost", True)
            self.focus_force()
            self.after(400, lambda: self.attributes("-topmost", False))
        except Exception:
            pass

    def _load_cnn_weights(self, path: Path):
        if path.exists():
            self.model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            self.model.eval()
            self.current_model_path = path

    def _on_model_select(self, choice: str):
        path = self.available_models.get(choice)
        if path and path.exists():
            self._load_cnn_weights(path)

    def _on_sensitivity_select(self, choice: str):
        mapping = {
            "Стандартный": 1.0,
            "Высокая чувств.": 1.25,
            "Мягкий": 0.80,
        }
        self.multi_tracker.set_sensitivity(mapping.get(choice, 1.0))

    def _on_faces_mode_select(self, choice: str):
        self.max_faces = 4 if "4" in choice else 1
        if self.is_running and self.landmarker is not None:
            self.landmarker.close()
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_faces=self.max_faces,
            )
            self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
            self.multi_tracker.reset()

    # =========================================================================
    # ПОСТРОЕНИЕ ИНТЕРФЕЙСА (OBSIDIAN + ELECTRIC LAVENDER)
    # =========================================================================
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- ВЕРХНЯЯ НАВИГАЦИЯ ---
        navbar = ctk.CTkFrame(self, height=60, corner_radius=0, fg_color=BG_VOID)
        navbar.grid(row=0, column=0, sticky="ew")
        navbar.grid_columnconfigure(5, weight=1)

        divider = ctk.CTkFrame(self, height=1, corner_radius=0, fg_color=BORDER_HAIRLINE)
        divider.grid(row=0, column=0, sticky="ews")

        ctk.CTkLabel(
            navbar,
            text="DROWSINESS AI",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_WHITE,
        ).grid(row=0, column=0, padx=(36, 10), pady=16, sticky="w")

        ctk.CTkLabel(
            navbar,
            text=str(DEVICE).upper(),
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=BG_ELEVATED,
            text_color=TEXT_LAVENDER,
            corner_radius=4,
            padx=7,
            pady=2,
        ).grid(row=0, column=1, padx=(0, 26), pady=16, sticky="w")

        self.nav_buttons = {}
        nav_items = [
            ("monitor", "МОНИТОРИНГ"),
            ("history", "ИСТОРИЯ СЕССИЙ"),
            ("models", "МОДЕЛИ"),
        ]
        for idx, (key, label) in enumerate(nav_items, start=2):
            btn = ctk.CTkButton(
                navbar,
                text=label,
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color="transparent",
                hover_color=BG_CARD,
                text_color=TEXT_WHITE if key == "monitor" else TEXT_MUTED,
                height=32,
                width=120,
                corner_radius=16,
                command=lambda k=key: self.switch_page(k),
            )
            btn.grid(row=0, column=idx, padx=4, pady=14, sticky="w")
            self.nav_buttons[key] = btn

        # Селекторы справа
        right_nav = ctk.CTkFrame(navbar, fg_color="transparent")
        right_nav.grid(row=0, column=6, padx=36, pady=12, sticky="e")

        self.faces_menu = ctk.CTkOptionMenu(
            right_nav,
            values=["До 4 человек", "1 человек"],
            width=128,
            height=30,
            fg_color=BG_CARD,
            button_color=BG_ELEVATED,
            button_hover_color=BORDER_HAIRLINE,
            text_color=TEXT_LAVENDER,
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._on_faces_mode_select,
        )
        self.faces_menu.pack(side="left", padx=5)

        self.sens_menu = ctk.CTkOptionMenu(
            right_nav,
            values=["Стандартный", "Высокая чувств.", "Мягкий"],
            width=140,
            height=30,
            fg_color=BG_CARD,
            button_color=BG_ELEVATED,
            button_hover_color=BORDER_HAIRLINE,
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(size=11),
            command=self._on_sensitivity_select,
        )
        self.sens_menu.pack(side="left", padx=5)

        model_names = list(self.available_models.keys()) or ["EyeCNN"]
        self.model_menu = ctk.CTkOptionMenu(
            right_nav,
            values=model_names,
            width=155,
            height=30,
            fg_color=BG_CARD,
            button_color=BG_ELEVATED,
            button_hover_color=BORDER_HAIRLINE,
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(size=11),
            command=self._on_model_select,
        )
        self.model_menu.pack(side="left", padx=5)

        self.alert_mode_menu = ctk.CTkOptionMenu(
            right_nav,
            values=["Голос и звук", "Только звук", "Без звука"],
            width=125,
            height=30,
            fg_color=BG_CARD,
            button_color=BG_ELEVATED,
            button_hover_color=BORDER_HAIRLINE,
            text_color=TEXT_WHITE,
            font=ctk.CTkFont(size=11),
        )
        self.alert_mode_menu.pack(side="left", padx=(5, 0))

        # --- КОНТЕЙНЕР СТРАНИЦ ---
        self.pages_container = ctk.CTkFrame(self, fg_color=BG_VOID, corner_radius=0)
        self.pages_container.grid(row=1, column=0, sticky="nsew")
        self.pages_container.grid_columnconfigure(0, weight=1)
        self.pages_container.grid_rowconfigure(0, weight=1)

        self.page_monitor = ctk.CTkScrollableFrame(
            self.pages_container,
            fg_color=BG_VOID,
            corner_radius=0,
            scrollbar_button_color=BG_ELEVATED,
            scrollbar_button_hover_color=BORDER_VIOLET,
        )
        self.page_history = ctk.CTkFrame(self.pages_container, fg_color=BG_VOID, corner_radius=0)
        self.page_models = ctk.CTkFrame(self.pages_container, fg_color=BG_VOID, corner_radius=0)

        self._build_monitor_page()
        self._build_history_page()
        self._build_models_page()

        self.switch_page("monitor")

    def switch_page(self, page_key: str):
        for p in (self.page_monitor, self.page_history, self.page_models):
            p.grid_forget()

        for k, btn in self.nav_buttons.items():
            if k == page_key:
                btn.configure(text_color=TEXT_WHITE, fg_color=BG_ELEVATED)
            else:
                btn.configure(text_color=TEXT_MUTED, fg_color="transparent")

        if page_key == "monitor":
            self.page_monitor.grid(row=0, column=0, sticky="nsew")
        elif page_key == "history":
            self.refresh_history_table()
            self.page_history.grid(row=0, column=0, sticky="nsew")
        elif page_key == "models":
            self.page_models.grid(row=0, column=0, sticky="nsew")

    # =========================================================================
    # СТРАНИЦА 1: МОНИТОРИНГ (HERO SPLIT + BENTO CARDS 01//..04//)
    # =========================================================================
    def _build_monitor_page(self):
        self.page_monitor.grid_columnconfigure(0, weight=1)

        # 1. ГЛАВНЫЙ HERO-БЛОК (СЛЕВА ТИПОГРАФИКА И CTA, СПРАВА ВИТРИНА С ВИДЕО)
        hero = ctk.CTkFrame(self.page_monitor, fg_color="transparent")
        hero.grid(row=0, column=0, padx=44, pady=(26, 22), sticky="ew")
        hero.grid_columnconfigure(0, weight=5)
        hero.grid_columnconfigure(1, weight=7)

        # --- Левая колонка Hero ---
        hero_left = ctk.CTkFrame(hero, fg_color="transparent")
        hero_left.grid(row=0, column=0, padx=(0, 28), sticky="nsew")

        ctk.CTkLabel(
            hero_left,
            text="EYECNN  —  MEDIAPIPE  —  LIVE TELEMETRY",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_DIM,
            anchor="w",
        ).pack(fill="x", pady=(6, 12))

        ctk.CTkLabel(
            hero_left,
            text="КОНТРОЛЬ\nВНИМАНИЯ",
            font=ctk.CTkFont(size=34, weight="bold"),
            text_color=TEXT_WHITE,
            justify="left",
            anchor="w",
        ).pack(fill="x")

        ctk.CTkLabel(
            hero_left,
            text="В РЕАЛЬНОМ\nВРЕМЕНИ",
            font=ctk.CTkFont(size=34, weight="bold", slant="italic"),
            text_color=TEXT_LAVENDER,
            justify="left",
            anchor="w",
        ).pack(fill="x", pady=(0, 14))

        self.lbl_state = ctk.CTkLabel(
            hero_left,
            text="СТАТУС: ОЖИДАНИЕ ЗАПУСКА",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_LAVENDER,
            anchor="w",
        )
        self.lbl_state.pack(fill="x", pady=(2, 4))

        self.lbl_reasons = ctk.CTkLabel(
            hero_left,
            text=(
                "Нейросетевой анализ состояния глаз, частоты морганий, "
                "наклона головы и зевания по видеопотоку без лишнего визуального шума."
            ),
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
            justify="left",
            anchor="w",
            wraplength=400,
        )
        self.lbl_reasons.pack(fill="x", pady=(0, 18))

        # Компактные индикаторы Score и PERCLOS
        meters_box = ctk.CTkFrame(
            hero_left,
            fg_color=BG_SURFACE,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_HAIRLINE,
        )
        meters_box.pack(fill="x", pady=(0, 20))

        row_s = ctk.CTkFrame(meters_box, fg_color="transparent")
        row_s.pack(fill="x", padx=14, pady=(10, 2))
        ctk.CTkLabel(
            row_s, text="ИНДЕКС УСТАЛОСТИ (SCORE)", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED
        ).pack(side="left")
        self.lbl_score_val = ctk.CTkLabel(
            row_s, text="0.0 / 100", font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_WHITE
        )
        self.lbl_score_val.pack(side="right")

        self.bar_score = ctk.CTkProgressBar(
            meters_box, height=6, corner_radius=3, fg_color=BG_ELEVATED, progress_color=ACCENT_VIOLET
        )
        self.bar_score.pack(fill="x", padx=14, pady=(2, 10))
        self.bar_score.set(0.0)

        row_p = ctk.CTkFrame(meters_box, fg_color="transparent")
        row_p.pack(fill="x", padx=14, pady=(0, 2))
        ctk.CTkLabel(
            row_p, text="ДОЛЯ ЗАКРЫТИЯ ГЛАЗ (PERCLOS)", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED
        ).pack(side="left")
        self.lbl_perclos_val = ctk.CTkLabel(
            row_p, text="0.0%", font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_LAVENDER
        )
        self.lbl_perclos_val.pack(side="right")

        self.bar_perclos = ctk.CTkProgressBar(
            meters_box, height=6, corner_radius=3, fg_color=BG_ELEVATED, progress_color=TEXT_LAVENDER
        )
        self.bar_perclos.pack(fill="x", padx=14, pady=(2, 12))
        self.bar_perclos.set(0.0)

        # Главные кнопки-пилюли (как GET AN ESTIMATE + BOOK A CALL на фото)
        cta_row = ctk.CTkFrame(hero_left, fg_color="transparent")
        cta_row.pack(fill="x", pady=(2, 0))

        self.btn_cam_primary = ctk.CTkButton(
            cta_row,
            text="ЗАПУСТИТЬ КАМЕРУ",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=TEXT_WHITE,
            hover_color="#E5E7EB",
            text_color=BG_VOID,
            corner_radius=20,
            height=40,
            width=175,
            command=self._toggle_camera_stream,
        )
        self.btn_cam_primary.pack(side="left", padx=(0, 10))

        self.btn_video_secondary = ctk.CTkButton(
            cta_row,
            text="ОТКРЫТЬ ВИДЕО",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=BG_SURFACE,
            hover_color=BG_ELEVATED,
            text_color=TEXT_WHITE,
            border_width=1,
            border_color=BORDER_HAIRLINE,
            corner_radius=20,
            height=40,
            width=155,
            command=self.open_video_file,
        )
        self.btn_video_secondary.pack(side="left")

        # --- Правая колонка Hero: Витрина видеопотока с фиолетовой подсветкой ---
        self.video_showcase = ctk.CTkFrame(
            hero,
            fg_color=BG_SURFACE,
            corner_radius=18,
            border_width=1,
            border_color=BORDER_VIOLET,
        )
        self.video_showcase.grid(row=0, column=1, sticky="nsew")
        self.video_showcase.grid_rowconfigure(0, weight=1)
        self.video_showcase.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            self.video_showcase,
            text="ВИДЕОПОТОК НЕ АКТИВЕН\n\nНажмите «ЗАПУСТИТЬ КАМЕРУ» или «ОТКРЫТЬ ВИДЕО» слева",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_MUTED,
            fg_color="#090910",
            corner_radius=14,
            height=385,
        )
        self.video_label.grid(row=0, column=0, padx=14, pady=(14, 10), sticky="nsew")

        # Нижняя полоска внутри витрины (таймер, FPS, источник + кнопка метрик)
        showcase_footer = ctk.CTkFrame(self.video_showcase, fg_color="transparent")
        showcase_footer.grid(row=1, column=0, padx=16, pady=(0, 14), sticky="ew")
        showcase_footer.grid_columnconfigure(0, weight=1)

        self.lbl_timer_fps = ctk.CTkLabel(
            showcase_footer,
            text="00:00   /   0.0 FPS   /   ОЖИДАНИЕ",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_LAVENDER,
            fg_color=BG_ELEVATED,
            corner_radius=12,
            padx=12,
            pady=4,
        )
        self.lbl_timer_fps.grid(row=0, column=0, sticky="w")

        self.btn_toggle_metrics_top = ctk.CTkButton(
            showcase_footer,
            text="ПОКАЗАТЬ МЕТРИКИ",
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=BG_ELEVATED,
            hover_color=BORDER_VIOLET,
            text_color=TEXT_WHITE,
            border_width=1,
            border_color=BORDER_VIOLET,
            corner_radius=14,
            height=28,
            width=155,
            command=self.toggle_metrics_panel,
        )
        self.btn_toggle_metrics_top.grid(row=0, column=1, sticky="e")

        # 2. СКРЫТАЯ ПАНЕЛЬ ДЕТАЛЬНЫХ МЕТРИК (ОТКРЫВАЕТСЯ ПО КНОПКЕ)
        self.metrics_drawer = ctk.CTkFrame(
            self.page_monitor,
            fg_color=BG_SURFACE,
            corner_radius=16,
            border_width=1,
            border_color=BORDER_VIOLET,
        )
        self._build_hidden_metrics_drawer()

        # 3. СЕКЦИЯ ИНСТРУМЕНТОВ ("OUR SERVICES" — 01 // 02 // 03 // 04)
        services_sec = ctk.CTkFrame(self.page_monitor, fg_color="transparent")
        services_sec.grid(row=2, column=0, padx=44, pady=(10, 30), sticky="ew")
        services_sec.grid_columnconfigure((0, 1, 2, 3), weight=1)

        sec_hdr = ctk.CTkFrame(services_sec, fg_color="transparent")
        sec_hdr.grid(row=0, column=0, columnspan=4, pady=(0, 14), sticky="ew")
        ctk.CTkLabel(
            sec_hdr,
            text="ИНСТРУМЕНТЫ",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=TEXT_WHITE,
        ).pack(side="left")
        ctk.CTkLabel(
            sec_hdr,
            text="  СИСТЕМЫ",
            font=ctk.CTkFont(size=18, weight="bold", slant="italic"),
            text_color=TEXT_LAVENDER,
        ).pack(side="left")

        self.btn_card_cam = self._create_service_card(
            services_sec,
            col=0,
            index_tag="01 //",
            title="ВЕБ-КАМЕРА",
            desc="Запуск потокового анализа глаз и мимики в реальном времени.",
            btn_text="ЗАПУСТИТЬ",
            command=self._toggle_camera_stream,
        )
        self.btn_card_metrics = self._create_service_card(
            services_sec,
            col=1,
            index_tag="02 //",
            title="ДЕТАЛЬНЫЕ МЕТРИКИ",
            desc="Раскрыть 7 признаков, кропы глаз 64x64 и живой график.",
            btn_text="ОТКРЫТЬ",
            command=self.toggle_metrics_panel,
        )
        self._create_service_card(
            services_sec,
            col=2,
            index_tag="03 //",
            title="КАЛИБРОВКА ПОЗЫ",
            desc="Сбросить базовый EAR глаз и нейтральный наклон головы.",
            btn_text="КАЛИБРОВАТЬ",
            command=self.recalibrate,
        )
        self._create_service_card(
            services_sec,
            col=3,
            index_tag="04 //",
            title="ОТЧЁТ СЕССИИ PNG",
            desc="Сформировать 4-панельный график по текущей или последней сессии.",
            btn_text="СОЗДАТЬ ОТЧЁТ",
            command=self.export_and_open_report_png,
        )

    def _create_service_card(self, parent, col: int, index_tag: str, title: str, desc: str, btn_text: str, command):
        card = ctk.CTkFrame(
            parent,
            fg_color=BG_SURFACE,
            corner_radius=14,
            border_width=1,
            border_color=BORDER_HAIRLINE,
        )
        card.grid(
            row=1,
            column=col,
            padx=(0, 8) if col == 0 else ((8, 0) if col == 3 else 8),
            sticky="nsew",
        )

        ctk.CTkLabel(
            card,
            text=index_tag,
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_LAVENDER,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(14, 4))

        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_WHITE,
            anchor="w",
        ).pack(fill="x", padx=16, pady=(0, 6))

        ctk.CTkLabel(
            card,
            text=desc,
            font=ctk.CTkFont(size=11),
            text_color=TEXT_MUTED,
            justify="left",
            anchor="w",
            wraplength=250,
        ).pack(fill="x", padx=16, pady=(0, 14))

        btn = ctk.CTkButton(
            card,
            text=btn_text,
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=BG_ELEVATED,
            hover_color=BORDER_VIOLET,
            text_color=TEXT_WHITE,
            border_width=1,
            border_color=BORDER_HAIRLINE,
            corner_radius=14,
            height=30,
            command=command,
        )
        btn.pack(anchor="w", padx=16, pady=(0, 14))
        return btn

    # =========================================================================
    # СКРЫТАЯ ПАНЕЛЬ ДЕТАЛЬНЫХ МЕТРИК
    # =========================================================================
    def _build_hidden_metrics_drawer(self):
        self.metrics_drawer.grid_columnconfigure((0, 1, 2), weight=1)

        hdr = ctk.CTkFrame(self.metrics_drawer, fg_color="transparent")
        hdr.grid(row=0, column=0, columnspan=3, padx=20, pady=(16, 8), sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hdr,
            text="ДЕТАЛЬНАЯ БИОМЕТРИЧЕСКАЯ ТЕЛЕМЕТРИЯ (7 ПРИЗНАКОВ)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_LAVENDER,
        ).grid(row=0, column=0, sticky="w")

        tools = ctk.CTkFrame(hdr, fg_color="transparent")
        tools.grid(row=0, column=1, sticky="e")

        ctk.CTkCheckBox(
            tools,
            text="Рамки глаз и рта",
            variable=self.show_boxes_var,
            font=ctk.CTkFont(size=11),
            text_color=TEXT_WHITE,
            fg_color=ACCENT_VIOLET,
            hover_color=ACCENT_VIOLET_HOVER,
            checkbox_width=16,
            checkbox_height=16,
        ).pack(side="left", padx=8)

        ctk.CTkCheckBox(
            tools,
            text="Точки лица (Face Mesh)",
            variable=self.show_mesh_var,
            font=ctk.CTkFont(size=11),
            text_color=TEXT_WHITE,
            fg_color=ACCENT_VIOLET,
            hover_color=ACCENT_VIOLET_HOVER,
            checkbox_width=16,
            checkbox_height=16,
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            tools,
            text="СНИМОК КАДРА",
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color=BG_ELEVATED,
            hover_color=BORDER_VIOLET,
            text_color=TEXT_WHITE,
            border_width=1,
            border_color=BORDER_HAIRLINE,
            corner_radius=12,
            height=26,
            width=120,
            command=self.take_snapshot,
        ).pack(side="left", padx=(8, 0))

        # Сетка 6 показателей
        metrics_grid = ctk.CTkFrame(self.metrics_drawer, fg_color="transparent")
        metrics_grid.grid(row=1, column=0, columnspan=2, padx=(16, 8), pady=6, sticky="nsew")
        metrics_grid.grid_columnconfigure((0, 1, 2), weight=1)

        self.lbl_blinks = self._create_metric_tile(metrics_grid, 0, 0, "ЧАСТОТА МОРГАНИЙ", "0 (0.0 / мин)")
        self.lbl_blink_dur = self._create_metric_tile(metrics_grid, 0, 1, "ДЛИТЕЛЬНОСТЬ МОРГАНИЯ", "0 мс")
        self.lbl_closure = self._create_metric_tile(metrics_grid, 0, 2, "НЕПРЕРЫВНОЕ ЗАКРЫТИЕ", "0.0 с")
        self.lbl_ear = self._create_metric_tile(metrics_grid, 1, 0, "АПЕРТУРА ГЛАЗ (EAR)", "0.00 (100%)")
        self.lbl_head = self._create_metric_tile(metrics_grid, 1, 1, "УГЛЫ ГОЛОВЫ (P / Y / R)", "0° / 0° / 0°")
        self.lbl_yawn = self._create_metric_tile(metrics_grid, 1, 2, "ИНДЕКС ЗЕВАНИЯ (MAR)", "0.00  |  0 зевков")

        # Кропы глаз 64x64 и счётчик тревог
        crops_box = ctk.CTkFrame(
            self.metrics_drawer,
            fg_color=BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_HAIRLINE,
        )
        crops_box.grid(row=1, column=2, padx=(8, 16), pady=10, sticky="nsew")
        crops_box.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(
            crops_box,
            text="ТЕНЗОРЫ EYECNN (64x64) И ТРЕВОГИ",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_MUTED,
        ).grid(row=0, column=0, columnspan=3, padx=12, pady=(8, 4), sticky="w")

        l_frame = ctk.CTkFrame(crops_box, fg_color="transparent")
        l_frame.grid(row=1, column=0, padx=6, pady=6)
        self.lbl_crop_l = ctk.CTkLabel(
            l_frame, text="L", width=58, height=58, fg_color=BG_VOID, corner_radius=6
        )
        self.lbl_crop_l.pack()
        self.lbl_prob_l = ctk.CTkLabel(l_frame, text="L: --", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED)
        self.lbl_prob_l.pack(pady=(2, 0))

        r_frame = ctk.CTkFrame(crops_box, fg_color="transparent")
        r_frame.grid(row=1, column=1, padx=6, pady=6)
        self.lbl_crop_r = ctk.CTkLabel(
            r_frame, text="R", width=58, height=58, fg_color=BG_VOID, corner_radius=6
        )
        self.lbl_crop_r.pack()
        self.lbl_prob_r = ctk.CTkLabel(r_frame, text="R: --", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED)
        self.lbl_prob_r.pack(pady=(2, 0))

        al_frame = ctk.CTkFrame(crops_box, fg_color="transparent")
        al_frame.grid(row=1, column=2, padx=6, pady=6)
        ctk.CTkLabel(al_frame, text="ТРЕВОГ", font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED).pack()
        self.lbl_alerts = ctk.CTkLabel(
            al_frame, text="0", font=ctk.CTkFont(size=22, weight="bold"), text_color=TEXT_LAVENDER
        )
        self.lbl_alerts.pack(pady=(2, 0))

        # Живой график телеметрии
        chart_frame = ctk.CTkFrame(
            self.metrics_drawer,
            fg_color=BG_CARD,
            corner_radius=12,
            border_width=1,
            border_color=BORDER_HAIRLINE,
        )
        chart_frame.grid(row=2, column=0, columnspan=3, padx=16, pady=(4, 16), sticky="ew")
        chart_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            chart_frame,
            text="ДИНАМИКА ИНДЕКСА СОНЛИВОСТИ (SCORE) И PERCLOS В РЕАЛЬНОМ ВРЕМЕНИ",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_MUTED,
        ).grid(row=0, column=0, padx=12, pady=(8, 2), sticky="w")

        self.chart_canvas = tk.Canvas(
            chart_frame,
            height=82,
            bg=BG_VOID,
            highlightthickness=1,
            highlightbackground=BORDER_HAIRLINE,
        )
        self.chart_canvas.grid(row=1, column=0, padx=12, pady=(4, 10), sticky="ew")
        self.chart_canvas.bind("<Configure>", lambda e: self._draw_live_chart())

    def _create_metric_tile(self, parent, row: int, col: int, title: str, init_val: str):
        tile = ctk.CTkFrame(
            parent,
            fg_color=BG_CARD,
            corner_radius=10,
            border_width=1,
            border_color=BORDER_HAIRLINE,
        )
        tile.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(
            tile, text=title, font=ctk.CTkFont(size=10, weight="bold"), text_color=TEXT_MUTED
        ).pack(anchor="w", padx=12, pady=(8, 2))
        val_lbl = ctk.CTkLabel(
            tile, text=init_val, font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_WHITE
        )
        val_lbl.pack(anchor="w", padx=12, pady=(0, 8))
        return val_lbl

    def toggle_metrics_panel(self):
        self.metrics_visible = not self.metrics_visible
        if self.metrics_visible:
            self.metrics_drawer.grid(row=1, column=0, padx=44, pady=(0, 18), sticky="ew")
            self.btn_toggle_metrics_top.configure(text="СКРЫТЬ МЕТРИКИ")
            self.btn_card_metrics.configure(text="СКРЫТЬ")
            self._draw_live_chart()
        else:
            self.metrics_drawer.grid_forget()
            self.btn_toggle_metrics_top.configure(text="ПОКАЗАТЬ МЕТРИКИ")
            self.btn_card_metrics.configure(text="ОТКРЫТЬ")

    def _draw_live_chart(self):
        if not self.metrics_visible:
            return
        c = self.chart_canvas
        w = max(200, c.winfo_width())
        h = max(50, c.winfo_height())
        c.delete("all")

        for pct in (25, 50, 75):
            y = h - int((pct / 100.0) * (h - 12)) - 6
            c.create_line(0, y, w, y, fill="#161624", width=1)

        y_thr = h - int((65.0 / 100.0) * (h - 12)) - 6
        c.create_line(0, y_thr, w, y_thr, fill="#7F1D1D", dash=(4, 4), width=1)

        n = len(self.score_history)
        if n < 2:
            return
        step_x = w / float(n - 1)

        perclos_pts = []
        for i, val in enumerate(self.perclos_history):
            x = i * step_x
            y = h - int((min(100.0, max(0.0, val)) / 100.0) * (h - 12)) - 6
            perclos_pts.extend([x, y])
        if len(perclos_pts) >= 4:
            c.create_line(*perclos_pts, fill="#38BDF8", width=1.6, smooth=True)

        score_pts = []
        for i, val in enumerate(self.score_history):
            x = i * step_x
            y = h - int((min(100.0, max(0.0, val)) / 100.0) * (h - 12)) - 6
            score_pts.extend([x, y])
        if len(score_pts) >= 4:
            c.create_line(*score_pts, fill=TEXT_LAVENDER, width=2.2, smooth=True)

    # =========================================================================
    # СТРАНИЦА 2: ИСТОРИЯ СЕССИЙ
    # =========================================================================
    def _build_history_page(self):
        self.page_history.grid_columnconfigure(0, weight=1)
        self.page_history.grid_rowconfigure(2, weight=1)

        hdr = ctk.CTkFrame(self.page_history, fg_color="transparent")
        hdr.grid(row=0, column=0, padx=44, pady=(24, 12), sticky="ew")
        ctk.CTkLabel(
            hdr, text="ИСТОРИЯ ", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_WHITE
        ).pack(side="left")
        ctk.CTkLabel(
            hdr, text="СЕССИЙ И ОТЧЁТЫ", font=ctk.CTkFont(size=24, weight="bold", slant="italic"), text_color=TEXT_LAVENDER
        ).pack(side="left")

        summary_frame = ctk.CTkFrame(
            self.page_history,
            corner_radius=14,
            fg_color=BG_SURFACE,
            border_width=1,
            border_color=BORDER_HAIRLINE,
        )
        summary_frame.grid(row=1, column=0, padx=44, pady=(0, 14), sticky="ew")
        summary_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.sum_sessions = self._create_metric_tile(summary_frame, 0, 0, "ВСЕГО СЕССИЙ", "0")
        self.sum_duration = self._create_metric_tile(summary_frame, 0, 1, "СУММАРНОЕ ВРЕМЯ", "0.0 мин")
        self.sum_blinks = self._create_metric_tile(summary_frame, 0, 2, "ВСЕГО МОРГАНИЙ", "0")
        self.sum_perclos = self._create_metric_tile(summary_frame, 0, 3, "СРЕДНИЙ PERCLOS", "0.0%")
        self.sum_alerts = self._create_metric_tile(summary_frame, 0, 4, "СИГНАЛОВ ТРЕВОГИ", "0")

        table_frame = ctk.CTkFrame(
            self.page_history,
            corner_radius=14,
            fg_color=BG_SURFACE,
            border_width=1,
            border_color=BORDER_HAIRLINE,
        )
        table_frame.grid(row=2, column=0, padx=44, pady=(0, 14), sticky="nsew")
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Treeview",
            background=BG_SURFACE,
            foreground=TEXT_WHITE,
            rowheight=30,
            fieldbackground=BG_SURFACE,
            borderwidth=0,
            font=("Helvetica", 12),
        )
        style.configure(
            "Treeview.Heading",
            background=BG_ELEVATED,
            foreground=TEXT_LAVENDER,
            font=("Helvetica", 11, "bold"),
            relief="flat",
        )
        style.map("Treeview", background=[("selected", BORDER_VIOLET)], foreground=[("selected", TEXT_WHITE)])

        cols = (
            "id", "start_time", "duration", "blinks", "blink_rate",
            "perclos_avg", "perclos_max", "alerts", "t_norm", "t_elev", "t_high", "model"
        )
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings")

        headings = {
            "id": ("№", 45),
            "start_time": ("Дата и время", 165),
            "duration": ("Длит. (с)", 80),
            "blinks": ("Морганий", 80),
            "blink_rate": ("Частота/мин", 95),
            "perclos_avg": ("Ср. PERCLOS", 95),
            "perclos_max": ("Пик PERCLOS", 95),
            "alerts": ("Тревог", 70),
            "t_norm": ("Норма (с)", 85),
            "t_elev": ("Повыш. (с)", 85),
            "t_high": ("Высок. (с)", 85),
            "model": ("Модель CNN", 150),
        }
        for col_id, (text, width) in headings.items():
            self.tree.heading(col_id, text=text)
            self.tree.column(col_id, width=width, anchor="center")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)

        self.tree.grid(row=0, column=0, padx=(10, 0), pady=10, sticky="nsew")
        scrollbar.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="ns")

        bottom_bar = ctk.CTkFrame(self.page_history, fg_color="transparent")
        bottom_bar.grid(row=3, column=0, padx=44, pady=(0, 22), sticky="ew")

        self.lbl_report_status = ctk.CTkLabel(
            bottom_bar,
            text="База данных: results/sessions.db (SQLite) и results/sessions_history.csv",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.lbl_report_status.pack(side="left")

        ctk.CTkButton(
            bottom_bar,
            text="ОБНОВИТЬ",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=BG_CARD,
            hover_color=BG_ELEVATED,
            text_color=TEXT_WHITE,
            border_width=1,
            border_color=BORDER_HAIRLINE,
            corner_radius=18,
            height=36,
            width=120,
            command=self.refresh_history_table,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            bottom_bar,
            text="ПАПКА ОТЧЁТОВ",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=BG_CARD,
            hover_color=BG_ELEVATED,
            text_color=TEXT_WHITE,
            border_width=1,
            border_color=BORDER_HAIRLINE,
            corner_radius=18,
            height=36,
            width=145,
            command=lambda: subprocess.Popen(["open", str(REPORTS_DIR)]),
        ).pack(side="right", padx=8)

        ctk.CTkButton(
            bottom_bar,
            text="СГЕНЕРИРОВАТЬ PNG-ОТЧЁТ",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=TEXT_WHITE,
            hover_color="#E5E7EB",
            text_color=BG_VOID,
            corner_radius=18,
            height=36,
            width=205,
            command=self.export_and_open_report_png,
        ).pack(side="right", padx=8)

        self.refresh_history_table()

    # =========================================================================
    # СТРАНИЦА 3: СРАВНЕНИЕ МОДЕЛЕЙ И МЕТОДИКА
    # =========================================================================
    def _build_models_page(self):
        self.page_models.grid_columnconfigure((0, 1), weight=1)
        self.page_models.grid_rowconfigure(2, weight=1)

        hdr = ctk.CTkFrame(self.page_models, fg_color="transparent")
        hdr.grid(row=0, column=0, columnspan=2, padx=44, pady=(24, 10), sticky="ew")
        ctk.CTkLabel(
            hdr, text="СРАВНЕНИЕ ", font=ctk.CTkFont(size=24, weight="bold"), text_color=TEXT_WHITE
        ).pack(side="left")
        ctk.CTkLabel(
            hdr, text="МОДЕЛЕЙ И МЕТОДИКА", font=ctk.CTkFont(size=24, weight="bold", slant="italic"), text_color=TEXT_LAVENDER
        ).pack(side="left")

        cards_row = ctk.CTkFrame(self.page_models, fg_color="transparent")
        cards_row.grid(row=1, column=0, columnspan=2, padx=44, pady=(0, 14), sticky="ew")
        cards_row.grid_columnconfigure((0, 1, 2), weight=1)

        models_info = [
            (
                "01 // EYECNN BASELINE (MRL)",
                "Webcam Test: 74.36%",
                "Macro F1: 0.7431  |  MRL Test: 96.85%",
                "Обучена только на инфракрасных снимках MRL Eye Dataset. Давала ложные срабатывания при прищуре на веб-камере.",
            ),
            (
                "02 // MOBILENETV3 SMALL",
                "Webcam Test: 58.97%",
                "Macro F1: 0.5846  |  Transfer Learning",
                "Архитектура ImageNet переобучилась под инфракрасный домен и показала низкую устойчивость к теням от дневного света.",
            ),
            (
                "03 // EYECNN FINE-TUNED + REPLAY",
                "Webcam Test: 100.00% (234 кадра)",
                "Macro F1: 1.0000  |  MRL Test: 97.50%",
                "Дообучена на собственном датасете со стабилизацией кропа 1:1 и защитой от забывания MRL Experience Replay.",
            ),
        ]

        for idx, (title, acc, f1, desc) in enumerate(models_info):
            card = ctk.CTkFrame(
                cards_row,
                corner_radius=14,
                fg_color=BG_SURFACE if idx < 2 else BG_CARD,
                border_width=1,
                border_color=BORDER_HAIRLINE if idx < 2 else BORDER_VIOLET,
            )
            card.grid(row=0, column=idx, padx=8, pady=4, sticky="nsew")
            ctk.CTkLabel(
                card, text=title, font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_LAVENDER
            ).pack(anchor="w", padx=16, pady=(14, 4))
            ctk.CTkLabel(
                card, text=acc, font=ctk.CTkFont(size=16, weight="bold"), text_color=TEXT_WHITE
            ).pack(anchor="w", padx=16, pady=2)
            ctk.CTkLabel(
                card, text=f1, font=ctk.CTkFont(size=11, weight="bold"), text_color=TEXT_MUTED
            ).pack(anchor="w", padx=16, pady=2)
            ctk.CTkLabel(
                card, text=desc, font=ctk.CTkFont(size=12), text_color=TEXT_MUTED, justify="left", wraplength=330
            ).pack(anchor="w", padx=16, pady=(6, 14))

        cm_frame = ctk.CTkFrame(
            self.page_models,
            corner_radius=14,
            fg_color=BG_SURFACE,
            border_width=1,
            border_color=BORDER_HAIRLINE,
        )
        cm_frame.grid(row=2, column=0, padx=(44, 8), pady=(0, 24), sticky="nsew")
        cm_frame.grid_rowconfigure(1, weight=1)
        cm_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            cm_frame,
            text="МАТРИЦЫ ОШИБОК (CONFUSION MATRICES)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_LAVENDER,
        ).grid(row=0, column=0, padx=16, pady=(14, 4), sticky="w")

        if CONFUSION_MATRIX_PNG.exists():
            pil_cm = Image.open(CONFUSION_MATRIX_PNG)
            cm_img = ctk.CTkImage(light_image=pil_cm, dark_image=pil_cm, size=(540, 205))
            lbl_cm = ctk.CTkLabel(cm_frame, image=cm_img, text="")
            lbl_cm.image = cm_img
            lbl_cm.grid(row=1, column=0, padx=14, pady=10)
        else:
            ctk.CTkLabel(
                cm_frame,
                text="График матриц ошибок не найден в results/cnn_finetuned/",
                text_color=TEXT_MUTED,
            ).grid(row=1, column=0, padx=14, pady=14)

        algo_frame = ctk.CTkFrame(
            self.page_models,
            corner_radius=14,
            fg_color=BG_SURFACE,
            border_width=1,
            border_color=BORDER_HAIRLINE,
        )
        algo_frame.grid(row=2, column=1, padx=(8, 44), pady=(0, 24), sticky="nsew")

        ctk.CTkLabel(
            algo_frame,
            text="СЕМЬ ПРИЗНАКОВ ОЦЕНКИ СОНЛИВОСТИ",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_LAVENDER,
        ).pack(anchor="w", padx=18, pady=(14, 8))

        features_text = (
            "1. EyeCNN + EAR (Апертура глаз): гибридный детектор отличает закрытие век\n"
            "   от прищуривания благодаря стабилизированному квадратному кропу 1:1.\n\n"
            "2. PERCLOS (% закрытия за 60 с): международный стандарт оценки усталости\n"
            "   (Норма < 15%, Повышенная сонливость 15–28%, Высокая тревога >= 28%).\n\n"
            "3. Непрерывное закрытие глаз (Closure Duration): детекция микросна\n"
            "   при смыкании век дольше 1.0 с (предупреждение) и >= 2.0 с (тревога).\n\n"
            "4. Частота и длительность морганий (Blink Rate & Duration): фиксация\n"
            "   медленных морганий (> 270–380 мс) и частого моргания (> 26 раз/мин).\n\n"
            "5. Положение головы 3D (Head Pose — Pitch, Yaw, Roll): контроль наклона\n"
            "   головы вперёд и вбок с автоадаптацией нейтральной осанки.\n\n"
            "6. Детектор зевания (Yawning MAR): отношение раскрытия губ с фильтром\n"
            "   длительности (>= 1.6 с), исключающим ложные срабатывания при речи."
        )
        ctk.CTkLabel(
            algo_frame,
            text=features_text,
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
            justify="left",
        ).pack(anchor="w", padx=18, pady=6)

    # =========================================================================
    # ДЕЙСТВИЯ И УПРАВЛЕНИЕ ПОТОКОМ
    # =========================================================================
    def _toggle_camera_stream(self):
        if self.is_running:
            self.stop_monitoring()
        else:
            self.start_monitoring(video_path=None)

    def take_snapshot(self):
        if self.last_annotated_frame is None:
            return
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = SNAPSHOT_DIR / f"snapshot_{ts_str}.png"
        cv2.imwrite(str(out_path), self.last_annotated_frame)
        self.lbl_state.configure(text=f"СНИМОК СОХРАНЁН: {out_path.name}", text_color=TEXT_LAVENDER)

    def export_and_open_report_png(self):
        from src.analytics.report_generator import generate_session_report_png

        out_png = generate_session_report_png(self.current_log_path)
        if out_png and out_png.exists():
            self.lbl_report_status.configure(
                text=f"График отчёта сохранён: {out_png.relative_to(ROOT)}",
                text_color=STATUS_GREEN,
            )
            self.lbl_state.configure(
                text=f"ОТЧЁТ СФОРМИРОВАН: {out_png.name}",
                text_color=STATUS_GREEN,
            )
            subprocess.Popen(
                ["open", str(out_png)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            self.lbl_report_status.configure(
                text="Не найден лог сессии в results/diagnostics/ для построения графика.",
                text_color=STATUS_RED,
            )

    def refresh_history_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        sessions = self.storage.get_recent_sessions(limit=100)
        for s in sessions:
            self.tree.insert(
                "",
                "end",
                values=(
                    s["id"],
                    s["start_time"],
                    f"{s['duration_sec']:.1f}",
                    s["total_blinks"],
                    f"{s['avg_blink_rate']:.1f}",
                    f"{s['avg_perclos']:.1f}%",
                    f"{s['max_perclos']:.1f}%",
                    s["alerts_count"],
                    f"{s['time_normal_sec']:.1f}",
                    f"{s['time_elevated_sec']:.1f}",
                    f"{s['time_high_sec']:.1f}",
                    s["model_name"],
                ),
            )

        stats = self.storage.get_summary_stats()
        self.sum_sessions.configure(text=str(stats["total_sessions"]))
        self.sum_duration.configure(text=f"{stats['total_duration_sec'] / 60.0:.1f} мин")
        self.sum_blinks.configure(text=str(stats["total_blinks"]))
        self.sum_perclos.configure(text=f"{stats['mean_perclos']:.1f}%")
        self.sum_alerts.configure(text=str(stats["total_alerts"]))

    def open_video_file(self):
        if self.is_running:
            self.stop_monitoring()

        file_path = filedialog.askopenfilename(
            title="Выберите видеофайл для анализа сонливости",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if file_path:
            self.start_monitoring(video_path=Path(file_path))

    def start_monitoring(self, video_path: Path | None = None):
        if self.is_running:
            return

        if video_path is not None:
            self.cap = cv2.VideoCapture(str(video_path))
            self.is_video_file = True
            self.video_source_name = video_path.name.upper()
        else:
            self.cap = cv2.VideoCapture(0)
            self.is_video_file = False
            self.video_source_name = "LIVE CAM"

        if not self.cap.isOpened():
            self.lbl_state.configure(text="ОШИБКА ОТКРЫТИЯ ВИДЕОПОТОКА", text_color=STATUS_RED)
            return

        if not self.is_video_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=self.max_faces,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        self.multi_tracker.reset()
        self._on_sensitivity_select(self.sens_menu.get())
        self.perclos_samples.clear()
        self.score_history = deque([0.0] * self.chart_history_len, maxlen=self.chart_history_len)
        self.perclos_history = deque([0.0] * self.chart_history_len, maxlen=self.chart_history_len)
        self.fps_history.clear()
        self.last_status = None

        self.session_start_dt = datetime.now()
        self.app_start_mono = time.monotonic()
        self.calibration_start_mono = self.app_start_mono
        self.last_frame_mono = self.app_start_mono
        self.last_timestamp_ms = -1
        self.is_running = True

        session_tag = self.session_start_dt.strftime("%Y%m%d_%H%M%S")
        self.current_log_path = DIAG_DIR / f"gui_session_{session_tag}.csv"
        self.csv_file = open(self.current_log_path, mode="w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "timestamp_ms", "level", "score", "perclos", "blink_rate", "avg_blink_ms",
            "closure_sec", "left_cnn_prob", "right_cnn_prob", "left_ear", "right_ear",
            "left_closed", "right_closed", "both_closed", "pitch_deg", "yaw_deg", "roll_deg",
            "mar", "total_yawns"
        ])

        self.btn_cam_primary.configure(
            text="ОСТАНОВИТЬ", fg_color=STATUS_RED, hover_color="#DC2626", text_color=TEXT_WHITE
        )
        self.btn_card_cam.configure(text="ОСТАНОВИТЬ")
        self._update_frame()

    def recalibrate(self):
        if not self.is_running:
            return
        now = time.monotonic()
        self.calibration_start_mono = now
        self.multi_tracker.recalibrate_all(now)
        self._on_sensitivity_select(self.sens_menu.get())

    def stop_monitoring(self):
        if not self.is_running:
            return

        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.landmarker is not None:
            self.landmarker.close()
            self.landmarker = None
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

        self.btn_cam_primary.configure(
            text="ЗАПУСТИТЬ КАМЕРУ", fg_color=TEXT_WHITE, hover_color="#E5E7EB", text_color=BG_VOID
        )
        self.btn_card_cam.configure(text="ЗАПУСТИТЬ")
        self.lbl_state.configure(text="СТАТУС: МОНИТОРИНГ ЗАВЕРШЁН", text_color=TEXT_LAVENDER)
        self.video_showcase.configure(border_color=BORDER_VIOLET)
        self.video_label.configure(
            image=None,
            text="МОНИТОРИНГ ЗАВЕРШЁН\n\nСессия сохранена в историю. Можно построить PNG-отчёт ниже.",
        )

        elapsed = time.monotonic() - self.app_start_mono
        if elapsed >= 2.0 and self.last_status is not None:
            avg_p = float(np.mean(self.perclos_samples)) if self.perclos_samples else 0.0
            max_p = float(np.max(self.perclos_samples)) if self.perclos_samples else 0.0
            st_dur = self.last_status.get("state_durations", {})
            self.storage.save_session(
                {
                    "start_time": self.session_start_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_sec": round(elapsed, 1),
                    "total_blinks": self.last_status.get("total_blinks", 0),
                    "avg_blink_rate": round(self.last_status.get("blink_rate_per_min", 0.0), 1),
                    "avg_blink_ms": round(self.last_status.get("avg_blink_duration_ms", 0.0), 0),
                    "avg_perclos": round(avg_p, 1),
                    "max_perclos": round(max_p, 1),
                    "alerts_count": self.last_status.get("alerts_count", 0),
                    "time_normal_sec": round(st_dur.get("NORMAL", 0.0), 1),
                    "time_elevated_sec": round(st_dur.get("ELEVATED", 0.0), 1),
                    "time_high_sec": round(st_dur.get("HIGH", 0.0), 1),
                    "model_name": self.current_model_path.name,
                }
            )
            self.refresh_history_table()

    def _trigger_alert_feedback(self):
        mode = self.alert_mode_menu.get()
        if "Без звука" in mode:
            return

        now = time.monotonic()
        if (now - self.last_sound_time) >= 2.5:
            self.last_sound_time = now
            subprocess.Popen(
                ["afplay", "/System/Library/Sounds/Ping.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        if "Голос" in mode and (now - self.last_voice_time) >= 9.0:
            self.last_voice_time = now
            subprocess.Popen(
                ["say", "Внимание! Высокий уровень сонливости!"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    # =========================================================================
    # ОБРАБОТКА КАДРОВ
    # =========================================================================
    def _update_frame(self):
        if not self.is_running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_monitoring()
            return

        if not self.is_video_file:
            frame = cv2.flip(frame, 1)

        now = time.monotonic()
        dt_frame = max(0.001, now - self.last_frame_mono)
        self.last_frame_mono = now
        self.fps_history.append(1.0 / dt_frame)
        fps_val = float(np.mean(self.fps_history))

        total_elapsed = int(now - self.app_start_mono)
        mm, ss = divmod(total_elapsed, 60)
        self.lbl_timer_fps.configure(
            text=f"{mm:02d}:{ss:02d}   /   {fps_val:.1f} FPS   /   {self.video_source_name}"
        )

        h, w = frame.shape[:2]
        cal_limit = 2.0 if self.is_video_file else CALIBRATION_SECONDS

        ts = max(int((now - self.app_start_mono) * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = ts

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        people = (
            self.multi_tracker.process_frame(frame, res.face_landmarks, self.model, now, cal_limit)
            if res.face_landmarks
            else []
        )

        self.lbl_timer_fps.configure(
            text=f"{mm:02d}:{ss:02d}   /   {fps_val:.1f} FPS   /   ЛЮДЕЙ: {len(people)}   /   {self.video_source_name}"
        )

        if people:
            any_high_alert = False
            high_ids = []

            for p in people:
                lm = p["landmarks"]
                status = p["status"]
                pid = p["person_id"]
                fx1, fy1, fx2, fy2 = p["face_box"]
                is_cal = p["is_calibrating"]
                rem_cal = p["rem_cal"]

                if self.show_mesh_var.get():
                    for idx in FACE_MESH_KEYPOINTS:
                        px = int(lm[idx].x * w)
                        py = int(lm[idx].y * h)
                        cv2.circle(frame, (px, py), 2, (250, 180, 196), -1, cv2.LINE_AA)

                # Индивидуальная рамка и бейдж над каждым человеком
                if is_cal:
                    f_col = (245, 180, 195)
                    tag = f"ID #{pid} | CAL {rem_cal:.1f}s"
                elif status["level"] == DrowsinessEvaluator.LEVEL_HIGH:
                    f_col = (60, 60, 240)
                    tag = f"ID #{pid} | HIGH | {status['score']:.0f}"
                    any_high_alert = True
                    high_ids.append(f"#{pid}")
                elif status["level"] == DrowsinessEvaluator.LEVEL_ELEVATED:
                    f_col = (0, 185, 245)
                    tag = f"ID #{pid} | ELEVATED | {status['score']:.0f}"
                else:
                    f_col = (110, 210, 90)
                    tag = f"ID #{pid} | OK | {status['score']:.0f}"

                if len(people) > 1 or self.show_boxes_var.get():
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), f_col, 2)
                    cv2.rectangle(frame, (fx1, max(0, fy1 - 24)), (min(w, fx1 + 215), fy1), f_col, -1)
                    cv2.putText(
                        frame,
                        tag,
                        (fx1 + 6, max(16, fy1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        (15, 15, 20),
                        1,
                        cv2.LINE_AA,
                    )

                if self.show_boxes_var.get():
                    for crop_data, name, is_c, is_s, prob, ear_v in [
                        (p["left_crop"], "L", p["l_closed"], p["l_sq"], p["prob_l"], p["l_ear"]),
                        (p["right_crop"], "R", p["r_closed"], p["r_sq"], p["prob_r"], p["r_ear"]),
                    ]:
                        box = crop_data[0]
                        if is_c:
                            col, st = (60, 60, 240), "CLOSED"
                        elif is_s:
                            col, st = (0, 185, 245), "SQUINT"
                        else:
                            col, st = (245, 180, 195), "OPEN"
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), col, 2)
                        if len(people) == 1:
                            cv2.putText(
                                frame,
                                f"{name}:{st} P:{prob:.2f} EAR:{ear_v:.2f}",
                                (box[0], max(22, box[1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.45,
                                col,
                                1,
                                cv2.LINE_AA,
                            )

                    yawn = status["yawn"]
                    m_box = yawn.get("mouth_box")
                    if m_box is not None and yawn["is_mouth_open"]:
                        m_col = (245, 110, 195) if yawn["is_yawning"] else (0, 185, 245)
                        m_label = (
                            f"YAWN ({yawn['yawn_duration_sec']:.1f}s)"
                            if yawn["is_yawning"]
                            else f"MAR:{yawn['mar']:.2f}"
                        )
                        cv2.rectangle(frame, (m_box[0], m_box[1]), (m_box[2], m_box[3]), m_col, 2)
                        cv2.putText(
                            frame,
                            m_label,
                            (m_box[0], min(h - 10, m_box[3] + 18)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.46,
                            m_col,
                            2,
                            cv2.LINE_AA,
                        )

            if any_high_alert:
                alert_text = (
                    f"DROWSINESS ALERT ({', '.join(high_ids)})"
                    if len(people) > 1
                    else "DROWSINESS ALERT"
                )
                cv2.rectangle(frame, (w // 2 - 255, 14), (w // 2 + 255, 70), (30, 25, 210), -1)
                cv2.putText(
                    frame,
                    alert_text,
                    (w // 2 - 220, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.82,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                self._trigger_alert_feedback()

            # Выбираем человека с наивысшим риском для главной панели телеметрии и CSV-лога
            primary = max(people, key=lambda item: (item["status"]["score"], -item["person_id"]))
            p_status = primary["status"]
            self.last_status = p_status
            if not primary["is_calibrating"]:
                self.perclos_samples.append(p_status["perclos"])

            self.score_history.append(p_status["score"])
            self.perclos_history.append(p_status["perclos"])
            if self.metrics_visible:
                self._draw_live_chart()

            if self.csv_writer is not None:
                pose = p_status["pose"]
                yawn = p_status["yawn"]
                self.csv_writer.writerow([
                    ts, p_status["level"], f"{p_status['score']:.2f}", f"{p_status['perclos']:.2f}",
                    f"{p_status['blink_rate_per_min']:.2f}", f"{p_status['avg_blink_duration_ms']:.1f}",
                    f"{p_status['closure_duration_sec']:.2f}", f"{primary['prob_l']:.4f}", f"{primary['prob_r']:.4f}",
                    f"{primary['l_ear']:.4f}", f"{primary['r_ear']:.4f}", int(primary["l_closed"]), int(primary["r_closed"]),
                    int(primary["both_closed"]), f"{pose['pitch_deg']:.1f}", f"{pose['yaw_deg']:.1f}", f"{pose['roll_deg']:.1f}",
                    f"{yawn['mar']:.3f}", yawn["total_yawns"]
                ])

            self._update_telemetry_widgets(
                p_status,
                primary["is_calibrating"],
                primary["rem_cal"],
                primary["left_crop"][2],
                primary["right_crop"][2],
                primary["prob_l"],
                primary["prob_r"],
                primary["l_closed"],
                primary["r_closed"],
                person_id=primary["person_id"],
                total_people=len(people),
            )
        else:
            cv2.putText(
                frame,
                "FACE NOT DETECTED",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (196, 181, 253),
                2,
                cv2.LINE_AA,
            )

        self.last_annotated_frame = frame

        disp_w, disp_h = 660, 372
        rgb_disp = cv2.cvtColor(cv2.resize(frame, (disp_w, disp_h)), cv2.COLOR_BGR2RGB)
        ctk_img = ctk.CTkImage(
            light_image=Image.fromarray(rgb_disp),
            dark_image=Image.fromarray(rgb_disp),
            size=(disp_w, disp_h),
        )
        self.video_label.configure(image=ctk_img, text="")
        self.video_label.image = ctk_img

        self.after(24, self._update_frame)

    def _update_telemetry_widgets(
        self,
        status: dict,
        is_calibrating: bool,
        rem_cal: float,
        crop_l_bgr,
        crop_r_bgr,
        prob_l: float,
        prob_r: float,
        l_closed: bool,
        r_closed: bool,
        person_id: int = 1,
        total_people: int = 1,
    ):
        score = status["score"]
        perclos = status["perclos"]

        self.lbl_score_val.configure(text=f"{score:.1f} / 100")
        self.bar_score.set(min(1.0, max(0.0, score / 100.0)))

        self.lbl_perclos_val.configure(text=f"{perclos:.1f}%")
        self.bar_perclos.set(min(1.0, max(0.0, perclos / 40.0)))

        id_suffix = f" (ID #{person_id} из {total_people})" if total_people > 1 else ""

        if is_calibrating:
            self.lbl_state.configure(
                text=f"СТАТУС: КАЛИБРОВКА ({rem_cal:.1f} С){id_suffix}", text_color=TEXT_LAVENDER
            )
            self.bar_score.configure(progress_color=ACCENT_VIOLET)
            self.video_showcase.configure(border_color=BORDER_VIOLET)
        else:
            lvl = status["level"]
            if lvl == DrowsinessEvaluator.LEVEL_HIGH:
                self.lbl_state.configure(text=f"СТАТУС: ВЫСОКАЯ СОНЛИВОСТЬ{id_suffix}", text_color=STATUS_RED)
                self.bar_score.configure(progress_color=STATUS_RED)
                self.video_showcase.configure(border_color=STATUS_RED)
            elif lvl == DrowsinessEvaluator.LEVEL_ELEVATED:
                self.lbl_state.configure(text=f"СТАТУС: ПОВЫШЕННАЯ УСТАЛОСТЬ{id_suffix}", text_color=STATUS_AMBER)
                self.bar_score.configure(progress_color=STATUS_AMBER)
                self.video_showcase.configure(border_color=STATUS_AMBER)
            else:
                self.lbl_state.configure(text=f"СТАТУС: НОРМАЛЬНОЕ СОСТОЯНИЕ{id_suffix}", text_color=STATUS_GREEN)
                self.bar_score.configure(progress_color=STATUS_GREEN)
                self.video_showcase.configure(border_color=BORDER_VIOLET)

        reasons = status.get("reasons", [])
        self.lbl_reasons.configure(
            text=f"Активные факторы: {', '.join(reasons)}"
            if reasons
            else "Все 7 биометрических показателей находятся в пределах физиологической нормы."
        )

        # Обновление скрытой панели детальных метрик
        self.lbl_blinks.configure(text=f"{status['total_blinks']} ({status['blink_rate_per_min']:.1f} / мин)")
        self.lbl_blink_dur.configure(text=f"{status['avg_blink_duration_ms']:.0f} мс")
        self.lbl_closure.configure(text=f"{status['closure_duration_sec']:.1f} с")
        self.lbl_ear.configure(text=f"{status['mean_ear']:.2f} ({status['mean_ear_ratio']*100:.0f}%)")

        pose = status["pose"]
        self.lbl_head.configure(
            text=f"{pose['pitch_deg']:+.0f}° / {pose['yaw_deg']:+.0f}° / {pose['roll_deg']:+.0f}° [{pose['pose_status']}]"
        )

        yawn = status["yawn"]
        yawn_tag = " (Зевок)" if yawn["is_yawning"] else ""
        self.lbl_yawn.configure(text=f"MAR {yawn['mar']:.2f}  |  {yawn['total_yawns']}{yawn_tag}")
        self.lbl_alerts.configure(text=str(status["alerts_count"]))

        if self.metrics_visible:
            for crop_bgr, lbl_img, lbl_p, name, p_val, is_c in [
                (crop_l_bgr, self.lbl_crop_l, self.lbl_prob_l, "L", prob_l, l_closed),
                (crop_r_bgr, self.lbl_crop_r, self.lbl_prob_r, "R", prob_r, r_closed),
            ]:
                rgb_c = cv2.cvtColor(cv2.resize(crop_bgr, (58, 58)), cv2.COLOR_BGR2RGB)
                pil_c = Image.fromarray(rgb_c)
                c_img = ctk.CTkImage(light_image=pil_c, dark_image=pil_c, size=(58, 58))
                lbl_img.configure(image=c_img, text="")
                lbl_img.image = c_img

                st_txt = "CLOSED" if is_c else "OPEN"
                st_col = STATUS_RED if is_c else STATUS_GREEN
                lbl_p.configure(text=f"{name}: {st_txt} ({p_val:.2f})", text_color=st_col)

    def _on_close(self):
        if self.is_running:
            self.stop_monitoring()
        self.destroy()


def main():
    app = DrowsinessApp()
    print("[3/3] Графический интерфейс открыт и готов к работе.", flush=True)
    app.mainloop()


if __name__ == "__main__":
    main()
