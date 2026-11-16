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
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
    EyeCNN,
    StabilizedEyeTracker,
    calculate_ear,
    crop_square_eye,
)

LANDMARKER_PATH = ROOT / "models" / "face_landmarker.task"
DIAG_DIR = ROOT / "results" / "diagnostics"
SNAPSHOT_DIR = ROOT / "results" / "snapshots"
REPORTS_DIR = ROOT / "results" / "reports"
CONFUSION_MATRIX_PNG = ROOT / "results" / "cnn_finetuned" / "webcam_test_confusion_matrices.png"

DIAG_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_SECONDS = 3.5

# Светлая минималистичная палитра в стиле Figma Community
BG_WHITE = "#FFFFFF"
BG_HERO = "#ECEDEE"
BG_SUBTLE = "#F7F8F9"
BORDER_LIGHT = "#E2E4E8"
BORDER_BUTTON = "#D0D5DD"
TEXT_MAIN = "#111827"
TEXT_MUTED = "#6B7280"
TEXT_SOFT = "#9CA3AF"
ACCENT_BLUE = "#0D99FF"
ACCENT_BLUE_HOVER = "#0B84E0"
STATUS_GREEN = "#16A34A"
STATUS_AMBER = "#D97706"
STATUS_RED = "#DC2626"

FACE_MESH_KEYPOINTS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
    1, 4, 5, 195, 197, 6, 168, 70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
]

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")


class DrowsinessApp(ctk.CTk):
    """
    Минималистичный интерфейс системы определения сонливости и усталости
    в светлом стиле Figma Community (без эмодзи, с выносом детальных метрик за кнопку).
    """

    def __init__(self):
        super().__init__()

        self.title("Drowsiness Detection System")
        self.geometry("1320x860")
        self.minsize(1140, 760)
        self.configure(fg_color=BG_WHITE)

        # Состояние работы
        self.is_running = False
        self.is_video_file = False
        self.video_source_name = "Камера не активна"
        self.cap = None
        self.landmarker = None
        self.storage = SessionStorage()

        self.trackers = [StabilizedEyeTracker("Left"), StabilizedEyeTracker("Right")]
        self.evaluator = DrowsinessEvaluator(window_seconds=60.0)

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
            "Стандартный профиль": 1.0,
            "Высокая чувствительность": 1.25,
            "Мягкий профиль": 0.80,
        }
        self.evaluator.sensitivity_multiplier = mapping.get(choice, 1.0)

    # =========================================================================
    # ПОСТРОЕНИЕ ИНТЕРФЕЙСА (В СТИЛЕ FIGMA COMMUNITY)
    # =========================================================================
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- ВЕРХНЯЯ НАВИГАЦИОННАЯ ПАНЕЛЬ (TOP NAVBAR) ---
        navbar = ctk.CTkFrame(self, height=54, corner_radius=0, fg_color=BG_WHITE)
        navbar.grid(row=0, column=0, sticky="ew")
        navbar.grid_columnconfigure(5, weight=1)

        # Тонкая линия-разделитель под шапкой
        divider = ctk.CTkFrame(self, height=1, corner_radius=0, fg_color=BORDER_LIGHT)
        divider.grid(row=0, column=0, sticky="ews")

        brand_lbl = ctk.CTkLabel(
            navbar,
            text="Drowsiness AI",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_MAIN,
        )
        brand_lbl.grid(row=0, column=0, padx=(24, 6), pady=14, sticky="w")

        badge_device = ctk.CTkLabel(
            navbar,
            text=str(DEVICE).upper(),
            font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#EAECEF",
            text_color=TEXT_MUTED,
            corner_radius=4,
            padx=6,
            pady=1,
        )
        badge_device.grid(row=0, column=1, padx=(0, 20), pady=14, sticky="w")

        # Кнопки переключения разделов в верхней строке
        self.nav_buttons = {}
        nav_items = [
            ("monitor", "Мониторинг"),
            ("history", "История сессий"),
            ("models", "Сравнение моделей"),
        ]
        for idx, (key, label) in enumerate(nav_items, start=2):
            btn = ctk.CTkButton(
                navbar,
                text=label,
                font=ctk.CTkFont(size=13, weight="normal"),
                fg_color="transparent",
                hover_color=BG_SUBTLE,
                text_color=TEXT_MAIN if key == "monitor" else TEXT_MUTED,
                height=32,
                width=120,
                corner_radius=6,
                command=lambda k=key: self.switch_page(k),
            )
            btn.grid(row=0, column=idx, padx=4, pady=11, sticky="w")
            self.nav_buttons[key] = btn

        # Правая часть шапки: компактные селекторы
        right_nav = ctk.CTkFrame(navbar, fg_color="transparent")
        right_nav.grid(row=0, column=6, padx=24, pady=10, sticky="e")

        self.sens_menu = ctk.CTkOptionMenu(
            right_nav,
            values=["Стандартный профиль", "Высокая чувствительность", "Мягкий профиль"],
            width=185,
            height=30,
            fg_color=BG_SUBTLE,
            button_color="#E5E7EB",
            button_hover_color="#D1D5DB",
            text_color=TEXT_MAIN,
            font=ctk.CTkFont(size=12),
            command=self._on_sensitivity_select,
        )
        self.sens_menu.pack(side="left", padx=6)

        model_names = list(self.available_models.keys()) or ["EyeCNN"]
        self.model_menu = ctk.CTkOptionMenu(
            right_nav,
            values=model_names,
            width=165,
            height=30,
            fg_color=BG_SUBTLE,
            button_color="#E5E7EB",
            button_hover_color="#D1D5DB",
            text_color=TEXT_MAIN,
            font=ctk.CTkFont(size=12),
            command=self._on_model_select,
        )
        self.model_menu.pack(side="left", padx=6)

        self.alert_mode_menu = ctk.CTkOptionMenu(
            right_nav,
            values=["Голос и звук", "Только звук", "Без звука"],
            width=130,
            height=30,
            fg_color=BG_SUBTLE,
            button_color="#E5E7EB",
            button_hover_color="#D1D5DB",
            text_color=TEXT_MAIN,
            font=ctk.CTkFont(size=12),
        )
        self.alert_mode_menu.pack(side="left", padx=(6, 0))

        # --- КОНТЕЙНЕР СТРАНИЦ ---
        self.pages_container = ctk.CTkFrame(self, fg_color=BG_WHITE, corner_radius=0)
        self.pages_container.grid(row=1, column=0, sticky="nsew")
        self.pages_container.grid_columnconfigure(0, weight=1)
        self.pages_container.grid_rowconfigure(0, weight=1)

        self.page_monitor = ctk.CTkScrollableFrame(
            self.pages_container, fg_color=BG_WHITE, corner_radius=0
        )
        self.page_history = ctk.CTkFrame(self.pages_container, fg_color=BG_WHITE, corner_radius=0)
        self.page_models = ctk.CTkFrame(self.pages_container, fg_color=BG_WHITE, corner_radius=0)

        self._build_monitor_page()
        self._build_history_page()
        self._build_models_page()

        self.switch_page("monitor")

    def switch_page(self, page_key: str):
        for p in (self.page_monitor, self.page_history, self.page_models):
            p.grid_forget()

        for k, btn in self.nav_buttons.items():
            if k == page_key:
                btn.configure(
                    text_color=TEXT_MAIN,
                    font=ctk.CTkFont(size=13, weight="bold"),
                    fg_color=BG_SUBTLE,
                )
            else:
                btn.configure(
                    text_color=TEXT_MUTED,
                    font=ctk.CTkFont(size=13, weight="normal"),
                    fg_color="transparent",
                )

        if page_key == "monitor":
            self.page_monitor.grid(row=0, column=0, sticky="nsew")
        elif page_key == "history":
            self.refresh_history_table()
            self.page_history.grid(row=0, column=0, sticky="nsew")
        elif page_key == "models":
            self.page_models.grid(row=0, column=0, sticky="nsew")

    # =========================================================================
    # СТРАНИЦА 1: МОНИТОРИНГ (КАК НА ФОТО FIGMA COMMUNITY)
    # =========================================================================
    def _build_monitor_page(self):
        self.page_monitor.grid_columnconfigure(0, weight=1)

        # 1. ЦЕНТРАЛЬНЫЙ ЗАГОЛОВОК И СТРОКА СТАТУСА (HERO HEADER)
        hero_header = ctk.CTkFrame(self.page_monitor, fg_color="transparent")
        hero_header.grid(row=0, column=0, pady=(26, 18))

        title_row = ctk.CTkFrame(hero_header, fg_color="transparent")
        title_row.pack()

        ctk.CTkLabel(
            title_row,
            text="Система контроля ",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")

        ctk.CTkLabel(
            title_row,
            text="сонливости, ",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=TEXT_MUTED,
        ).pack(side="left")

        ctk.CTkLabel(
            title_row,
            text="внимания",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=ACCENT_BLUE,
        ).pack(side="left")

        ctk.CTkLabel(
            title_row,
            text=" и усталости",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")

        # Строка-капсула статуса (как поисковая строка на референсе)
        self.status_pill = ctk.CTkFrame(
            hero_header,
            fg_color=BG_SUBTLE,
            corner_radius=20,
            border_width=1,
            border_color=BORDER_LIGHT,
            height=38,
        )
        self.status_pill.pack(pady=(12, 0))

        self.lbl_pill_text = ctk.CTkLabel(
            self.status_pill,
            text="Статус: ожидание запуска видеопотока  |  Индекс сонливости: 0.0 из 100",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
            padx=22,
            pady=6,
        )
        self.lbl_pill_text.pack()

        # 2. ГЛАВНЫЙ СЕРЫЙ БАННЕР (КАК "Publish your Weave Workflow" НА ФОТО)
        self.hero_banner = ctk.CTkFrame(
            self.page_monitor,
            fg_color=BG_HERO,
            corner_radius=16,
            border_width=1,
            border_color=BORDER_LIGHT,
        )
        self.hero_banner.grid(row=1, column=0, padx=64, pady=(6, 22), sticky="ew")
        self.hero_banner.grid_columnconfigure(0, weight=4)
        self.hero_banner.grid_columnconfigure(1, weight=7)

        # Левая колонка внутри баннера (Описание состояния и главные индикаторы)
        banner_left = ctk.CTkFrame(self.hero_banner, fg_color="transparent")
        banner_left.grid(row=0, column=0, padx=(30, 18), pady=26, sticky="nsew")

        self.lbl_state = ctk.CTkLabel(
            banner_left,
            text="Готов к работе",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=TEXT_MAIN,
            anchor="w",
        )
        self.lbl_state.pack(fill="x", pady=(4, 8))

        self.lbl_reasons = ctk.CTkLabel(
            banner_left,
            text=(
                "Запустите веб-камеру или откройте видеофайл для автоматического "
                "анализа состояния глаз, мимики и наклона головы в реальном времени ->"
            ),
            font=ctk.CTkFont(size=13),
            text_color="#4B5563",
            justify="left",
            anchor="w",
            wraplength=340,
        )
        self.lbl_reasons.pack(fill="x", pady=(0, 18))

        # Компактная полоса Индекса сонливости
        score_row = ctk.CTkFrame(banner_left, fg_color="transparent")
        score_row.pack(fill="x", pady=(4, 2))
        ctk.CTkLabel(
            score_row,
            text="Индекс усталости (Score)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")
        self.lbl_score_val = ctk.CTkLabel(
            score_row,
            text="0.0 / 100",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MUTED,
        )
        self.lbl_score_val.pack(side="right")

        self.bar_score = ctk.CTkProgressBar(
            banner_left,
            height=8,
            corner_radius=4,
            fg_color="#D5D8DC",
            progress_color=STATUS_GREEN,
        )
        self.bar_score.pack(fill="x", pady=(2, 12))
        self.bar_score.set(0.0)

        # Компактная полоса PERCLOS
        perclos_row = ctk.CTkFrame(banner_left, fg_color="transparent")
        perclos_row.pack(fill="x", pady=(2, 2))
        ctk.CTkLabel(
            perclos_row,
            text="Доля закрытия глаз (PERCLOS)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(side="left")
        self.lbl_perclos_val = ctk.CTkLabel(
            perclos_row,
            text="0.0%",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=TEXT_MUTED,
        )
        self.lbl_perclos_val.pack(side="right")

        self.bar_perclos = ctk.CTkProgressBar(
            banner_left,
            height=8,
            corner_radius=4,
            fg_color="#D5D8DC",
            progress_color=ACCENT_BLUE,
        )
        self.bar_perclos.pack(fill="x", pady=(2, 20))
        self.bar_perclos.set(0.0)

        # Нижняя капсула таймера и FPS (как индикатор слайдов в левом нижнем углу баннера Figma)
        pill_bottom = ctk.CTkFrame(
            banner_left,
            fg_color=BG_WHITE,
            corner_radius=14,
            border_width=1,
            border_color="#D0D5DD",
        )
        pill_bottom.pack(anchor="w", side="bottom", pady=(10, 0))

        self.lbl_timer_fps = ctk.CTkLabel(
            pill_bottom,
            text="00:00  •  0.0 FPS  •  Ожидание",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_MAIN,
            padx=12,
            pady=4,
        )
        self.lbl_timer_fps.pack()

        # Правая колонка внутри баннера (Окно видеопотока)
        video_container = ctk.CTkFrame(
            self.hero_banner,
            fg_color=BG_WHITE,
            corner_radius=12,
            border_width=1,
            border_color="#DCE0E5",
        )
        video_container.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="nsew")
        video_container.grid_rowconfigure(0, weight=1)
        video_container.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            video_container,
            text="Предпросмотр видеопотока\n\nВыберите «Веб-камера» или «Видеофайл» в панели управления ниже",
            font=ctk.CTkFont(size=13),
            text_color=TEXT_MUTED,
            fg_color=BG_SUBTLE,
            corner_radius=10,
            height=350,
        )
        self.video_label.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")

        # 3. СЕКЦИЯ УПРАВЛЕНИЯ И ИНСТРУМЕНТОВ (В СТИЛЕ СЕТКИ "Skills" С КНОПКАМИ "Try it out")
        skills_section = ctk.CTkFrame(self.page_monitor, fg_color="transparent")
        skills_section.grid(row=2, column=0, padx=64, pady=(4, 16), sticky="ew")
        skills_section.grid_columnconfigure((0, 1), weight=1)

        # Заголовок секции + кнопка раскрытия метрик справа (как "Browse skills >")
        sec_header = ctk.CTkFrame(skills_section, fg_color="transparent")
        sec_header.grid(row=0, column=0, columnspan=2, pady=(0, 12), sticky="ew")
        sec_header.grid_columnconfigure(0, weight=1)

        title_sub = ctk.CTkFrame(sec_header, fg_color="transparent")
        title_sub.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            title_sub,
            text="Управление и диагностика",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=TEXT_MAIN,
        ).pack(anchor="w")
        ctk.CTkLabel(
            title_sub,
            text="Инструменты запуска видеопотока, калибровки, сохранения отчётов и просмотра детальной телеметрии.",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        ).pack(anchor="w", pady=(2, 0))

        self.btn_toggle_metrics_link = ctk.CTkButton(
            sec_header,
            text="Показать детальные метрики  >",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="transparent",
            hover_color=BG_SUBTLE,
            text_color=ACCENT_BLUE,
            width=210,
            height=30,
            anchor="e",
            command=self.toggle_metrics_panel,
        )
        self.btn_toggle_metrics_link.grid(row=0, column=1, sticky="e")

        # Четыре минималистичные карточки в 2 колонки (как карточки внизу на скриншоте Figma)
        self.btn_cam_action = self._create_action_row_card(
            skills_section,
            row=1,
            col=0,
            code_tag="CAM",
            title="мониторинг с веб-камеры",
            subtitle="Запуск потокового анализа глаз и мимики в реальном времени",
            btn_text="Запустить",
            command=self._toggle_camera_stream,
        )

        self.btn_file_action = self._create_action_row_card(
            skills_section,
            row=1,
            col=1,
            code_tag="VID",
            title="анализ готового видеофайла",
            subtitle="Проверка записи .mp4, .mov или .avi с расчётом всех показателей",
            btn_text="Выбрать файл",
            command=self.open_video_file,
        )

        self.btn_metrics_action = self._create_action_row_card(
            skills_section,
            row=2,
            col=0,
            code_tag="MET",
            title="детальные биометрические метрики",
            subtitle="Открыть панель всех 7 признаков, кропов глаз 64x64 и живого графика",
            btn_text="Открыть метрики",
            command=self.toggle_metrics_panel,
        )

        self.btn_report_action = self._create_action_row_card(
            skills_section,
            row=2,
            col=1,
            code_tag="REP",
            title="графический отчёт и калибровка",
            subtitle="Построение сводного PNG-графика сессии или сброс позы головы",
            btn_text="Создать отчёт",
            command=self.export_and_open_report_png,
            secondary_btn_text="Калибровка",
            secondary_command=self.recalibrate,
        )

        # 4. СКРЫТАЯ ПАНЕЛЬ ДЕТАЛЬНЫХ МЕТРИК (ПОЯВЛЯЕТСЯ ТОЛЬКО ПО КНОПКЕ)
        self.metrics_drawer = ctk.CTkFrame(
            self.page_monitor,
            fg_color=BG_SUBTLE,
            corner_radius=14,
            border_width=1,
            border_color=BORDER_LIGHT,
        )
        # По умолчанию НЕ отображается (спрятана за кнопку)
        self._build_hidden_metrics_drawer()

    def _create_action_row_card(
        self,
        parent,
        row: int,
        col: int,
        code_tag: str,
        title: str,
        subtitle: str,
        btn_text: str,
        command,
        secondary_btn_text: str | None = None,
        secondary_command=None,
    ):
        card = ctk.CTkFrame(parent, fg_color=BG_WHITE, corner_radius=10)
        card.grid(
            row=row,
            column=col,
            padx=(0, 18) if col == 0 else (18, 0),
            pady=10,
            sticky="ew",
        )
        card.grid_columnconfigure(1, weight=1)

        # Квадратная иконка-монограмма слева (как иконки [F] в Figma Skills)
        icon_box = ctk.CTkFrame(
            card,
            width=44,
            height=44,
            corner_radius=10,
            fg_color=BG_SUBTLE,
            border_width=1,
            border_color=BORDER_LIGHT,
        )
        icon_box.grid(row=0, column=0, rowspan=2, padx=(0, 14), pady=4, sticky="nw")
        icon_box.grid_propagate(False)

        ctk.CTkLabel(
            icon_box,
            text=code_tag,
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=TEXT_MAIN,
        ).place(relx=0.5, rely=0.5, anchor="center")

        # Текст карточки
        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_MAIN,
            anchor="w",
        ).grid(row=0, column=1, sticky="w")

        ctk.CTkLabel(
            card,
            text=subtitle,
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
            anchor="w",
            justify="left",
            wraplength=340,
        ).grid(row=1, column=1, sticky="w", pady=(2, 0))

        # Кнопки справа (в стиле кнопки "Try it out" — белая с тонкой рамкой)
        btn_box = ctk.CTkFrame(card, fg_color="transparent")
        btn_box.grid(row=0, column=2, rowspan=2, padx=(12, 0), sticky="e")

        if secondary_btn_text and secondary_command:
            sec_btn = ctk.CTkButton(
                btn_box,
                text=secondary_btn_text,
                font=ctk.CTkFont(size=12, weight="bold"),
                fg_color=BG_WHITE,
                hover_color=BG_SUBTLE,
                text_color=TEXT_MAIN,
                border_width=1,
                border_color=BORDER_BUTTON,
                corner_radius=8,
                width=96,
                height=32,
                command=secondary_command,
            )
            sec_btn.pack(side="left", padx=(0, 6))

        main_btn = ctk.CTkButton(
            btn_box,
            text=btn_text,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=BG_WHITE,
            hover_color=BG_SUBTLE,
            text_color=TEXT_MAIN,
            border_width=1,
            border_color=BORDER_BUTTON,
            corner_radius=8,
            width=120,
            height=32,
            command=command,
        )
        main_btn.pack(side="left")
        return main_btn

    # =========================================================================
    # СКРЫТАЯ ПАНЕЛЬ ДЕТАЛЬНЫХ МЕТРИК (ОТКРЫВАЕТСЯ ПО КНОПКЕ)
    # =========================================================================
    def _build_hidden_metrics_drawer(self):
        self.metrics_drawer.grid_columnconfigure((0, 1, 2), weight=1)

        # Шапка панели метрик
        hdr = ctk.CTkFrame(self.metrics_drawer, fg_color="transparent")
        hdr.grid(row=0, column=0, columnspan=3, padx=20, pady=(16, 8), sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            hdr,
            text="Детальная биометрическая телеметрия (7 признаков)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=TEXT_MAIN,
        ).grid(row=0, column=0, sticky="w")

        # Переключатели отрисовки и кнопка снимка
        tools_row = ctk.CTkFrame(hdr, fg_color="transparent")
        tools_row.grid(row=0, column=1, sticky="e")

        ctk.CTkCheckBox(
            tools_row,
            text="Рамки глаз и рта",
            variable=self.show_boxes_var,
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MAIN,
            checkbox_width=18,
            checkbox_height=18,
        ).pack(side="left", padx=8)

        ctk.CTkCheckBox(
            tools_row,
            text="Точки лица (Face Mesh)",
            variable=self.show_mesh_var,
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MAIN,
            checkbox_width=18,
            checkbox_height=18,
        ).pack(side="left", padx=8)

        ctk.CTkButton(
            tools_row,
            text="Сохранить снимок кадра",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=BG_WHITE,
            hover_color="#EAECEF",
            text_color=TEXT_MAIN,
            border_width=1,
            border_color=BORDER_BUTTON,
            height=28,
            width=155,
            command=self.take_snapshot,
        ).pack(side="left", padx=(8, 0))

        # Сетка 6 карточек показателей
        metrics_grid = ctk.CTkFrame(self.metrics_drawer, fg_color="transparent")
        metrics_grid.grid(row=1, column=0, columnspan=2, padx=(16, 8), pady=8, sticky="nsew")
        metrics_grid.grid_columnconfigure((0, 1, 2), weight=1)

        self.lbl_blinks = self._create_metric_tile(metrics_grid, 0, 0, "Частота морганий", "0 (0.0 / мин)")
        self.lbl_blink_dur = self._create_metric_tile(metrics_grid, 0, 1, "Длительность моргания", "0 мс")
        self.lbl_closure = self._create_metric_tile(metrics_grid, 0, 2, "Непрерывное закрытие", "0.0 с")
        self.lbl_ear = self._create_metric_tile(metrics_grid, 1, 0, "Апертура глаз (EAR)", "0.00 (100%)")
        self.lbl_head = self._create_metric_tile(metrics_grid, 1, 1, "Углы головы (P / Y / R)", "0° / 0° / 0°")
        self.lbl_yawn = self._create_metric_tile(metrics_grid, 1, 2, "Индекс зевания (MAR)", "0.00  |  0 зевков")

        # Правый блок внутри панели: кропы глаз 64x64 и счётчик тревог
        crops_box = ctk.CTkFrame(
            self.metrics_drawer,
            fg_color=BG_WHITE,
            corner_radius=10,
            border_width=1,
            border_color=BORDER_LIGHT,
        )
        crops_box.grid(row=1, column=2, padx=(8, 16), pady=12, sticky="nsew")
        crops_box.grid_columnconfigure((0, 1, 2), weight=1)

        ctk.CTkLabel(
            crops_box,
            text="Тензоры EyeCNN (64x64) и тревоги",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_MUTED,
        ).grid(row=0, column=0, columnspan=3, padx=12, pady=(8, 4), sticky="w")

        l_frame = ctk.CTkFrame(crops_box, fg_color="transparent")
        l_frame.grid(row=1, column=0, padx=6, pady=6)
        self.lbl_crop_l = ctk.CTkLabel(
            l_frame, text="L", width=58, height=58, fg_color=BG_SUBTLE, corner_radius=6
        )
        self.lbl_crop_l.pack()
        self.lbl_prob_l = ctk.CTkLabel(l_frame, text="L: --", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED)
        self.lbl_prob_l.pack(pady=(2, 0))

        r_frame = ctk.CTkFrame(crops_box, fg_color="transparent")
        r_frame.grid(row=1, column=1, padx=6, pady=6)
        self.lbl_crop_r = ctk.CTkLabel(
            r_frame, text="R", width=58, height=58, fg_color=BG_SUBTLE, corner_radius=6
        )
        self.lbl_crop_r.pack()
        self.lbl_prob_r = ctk.CTkLabel(r_frame, text="R: --", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED)
        self.lbl_prob_r.pack(pady=(2, 0))

        al_frame = ctk.CTkFrame(crops_box, fg_color="transparent")
        al_frame.grid(row=1, column=2, padx=6, pady=6)
        ctk.CTkLabel(al_frame, text="Тревог", font=ctk.CTkFont(size=11), text_color=TEXT_MUTED).pack()
        self.lbl_alerts = ctk.CTkLabel(
            al_frame, text="0", font=ctk.CTkFont(size=20, weight="bold"), text_color=TEXT_MAIN
        )
        self.lbl_alerts.pack(pady=(2, 0))

        # Живой график истории Score и PERCLOS внизу панели метрик
        chart_frame = ctk.CTkFrame(
            self.metrics_drawer,
            fg_color=BG_WHITE,
            corner_radius=10,
            border_width=1,
            border_color=BORDER_LIGHT,
        )
        chart_frame.grid(row=2, column=0, columnspan=3, padx=16, pady=(4, 16), sticky="ew")
        chart_frame.grid_columnconfigure(0, weight=1)

        chart_top = ctk.CTkFrame(chart_frame, fg_color="transparent")
        chart_top.grid(row=0, column=0, padx=12, pady=(6, 2), sticky="ew")
        ctk.CTkLabel(
            chart_top,
            text="Динамика индекса сонливости (Score) и PERCLOS в реальном времени",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=TEXT_MUTED,
        ).pack(side="left")

        self.chart_canvas = tk.Canvas(
            chart_frame,
            height=80,
            bg=BG_WHITE,
            highlightthickness=0,
        )
        self.chart_canvas.grid(row=1, column=0, padx=12, pady=(2, 8), sticky="ew")
        self.chart_canvas.bind("<Configure>", lambda e: self._draw_live_chart())

    def _create_metric_tile(self, parent, row: int, col: int, title: str, init_val: str):
        tile = ctk.CTkFrame(
            parent,
            fg_color=BG_WHITE,
            corner_radius=10,
            border_width=1,
            border_color=BORDER_LIGHT,
        )
        tile.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
        ctk.CTkLabel(
            tile, text=title, font=ctk.CTkFont(size=11), text_color=TEXT_MUTED
        ).pack(anchor="w", padx=12, pady=(8, 2))
        val_lbl = ctk.CTkLabel(
            tile, text=init_val, font=ctk.CTkFont(size=14, weight="bold"), text_color=TEXT_MAIN
        )
        val_lbl.pack(anchor="w", padx=12, pady=(0, 8))
        return val_lbl

    def toggle_metrics_panel(self):
        self.metrics_visible = not self.metrics_visible
        if self.metrics_visible:
            self.metrics_drawer.grid(row=3, column=0, padx=64, pady=(4, 28), sticky="ew")
            self.btn_toggle_metrics_link.configure(text="Скрыть детальные метрики  ^")
            self.btn_metrics_action.configure(text="Скрыть метрики")
            self._draw_live_chart()
        else:
            self.metrics_drawer.grid_forget()
            self.btn_toggle_metrics_link.configure(text="Показать детальные метрики  >")
            self.btn_metrics_action.configure(text="Открыть метрики")

    def _draw_live_chart(self):
        if not self.metrics_visible:
            return
        c = self.chart_canvas
        w = max(200, c.winfo_width())
        h = max(50, c.winfo_height())
        c.delete("all")

        for pct in (25, 50, 75):
            y = h - int((pct / 100.0) * (h - 12)) - 6
            c.create_line(0, y, w, y, fill="#F1F3F5", width=1)

        y_thr = h - int((65.0 / 100.0) * (h - 12)) - 6
        c.create_line(0, y_thr, w, y_thr, fill="#FCA5A5", dash=(4, 4), width=1)

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
            c.create_line(*perclos_pts, fill="#93C5FD", width=1.8, smooth=True)

        score_pts = []
        for i, val in enumerate(self.score_history):
            x = i * step_x
            y = h - int((min(100.0, max(0.0, val)) / 100.0) * (h - 12)) - 6
            score_pts.extend([x, y])
        if len(score_pts) >= 4:
            c.create_line(*score_pts, fill=ACCENT_BLUE, width=2.2, smooth=True)

    # =========================================================================
    # СТРАНИЦА 2: ИСТОРИЯ СЕССИЙ
    # =========================================================================
    def _build_history_page(self):
        self.page_history.grid_columnconfigure(0, weight=1)
        self.page_history.grid_rowconfigure(1, weight=1)

        summary_frame = ctk.CTkFrame(
            self.page_history,
            corner_radius=12,
            fg_color=BG_SUBTLE,
            border_width=1,
            border_color=BORDER_LIGHT,
        )
        summary_frame.grid(row=0, column=0, padx=48, pady=(24, 14), sticky="ew")
        summary_frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        self.sum_sessions = self._create_metric_tile(summary_frame, 0, 0, "Всего сессий", "0")
        self.sum_duration = self._create_metric_tile(summary_frame, 0, 1, "Суммарное время", "0.0 мин")
        self.sum_blinks = self._create_metric_tile(summary_frame, 0, 2, "Всего морганий", "0")
        self.sum_perclos = self._create_metric_tile(summary_frame, 0, 3, "Средний PERCLOS", "0.0%")
        self.sum_alerts = self._create_metric_tile(summary_frame, 0, 4, "Сигналов тревоги", "0")

        table_frame = ctk.CTkFrame(
            self.page_history,
            corner_radius=12,
            fg_color=BG_WHITE,
            border_width=1,
            border_color=BORDER_LIGHT,
        )
        table_frame.grid(row=1, column=0, padx=48, pady=(0, 14), sticky="nsew")
        table_frame.grid_columnconfigure(0, weight=1)
        table_frame.grid_rowconfigure(0, weight=1)

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Treeview",
            background=BG_WHITE,
            foreground=TEXT_MAIN,
            rowheight=30,
            fieldbackground=BG_WHITE,
            borderwidth=0,
            font=("Helvetica", 12),
        )
        style.configure(
            "Treeview.Heading",
            background=BG_SUBTLE,
            foreground=TEXT_MAIN,
            font=("Helvetica", 12, "bold"),
            relief="flat",
        )
        style.map("Treeview", background=[("selected", "#E0F2FE")], foreground=[("selected", TEXT_MAIN)])

        cols = (
            "id", "start_time", "duration", "blinks", "blink_rate",
            "perclos_avg", "perclos_max", "alerts", "t_norm", "t_elev", "t_high", "model"
        )
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings")

        headings = {
            "id": ("№", 45),
            "start_time": ("Дата и время запуска", 165),
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
        bottom_bar.grid(row=2, column=0, padx=48, pady=(0, 20), sticky="ew")

        self.lbl_report_status = ctk.CTkLabel(
            bottom_bar,
            text="База данных: results/sessions.db (SQLite) и results/sessions_history.csv",
            font=ctk.CTkFont(size=12),
            text_color=TEXT_MUTED,
        )
        self.lbl_report_status.pack(side="left")

        ctk.CTkButton(
            bottom_bar,
            text="Обновить таблицу",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=BG_WHITE,
            hover_color=BG_SUBTLE,
            text_color=TEXT_MAIN,
            border_width=1,
            border_color=BORDER_BUTTON,
            height=34,
            width=140,
            command=self.refresh_history_table,
        ).pack(side="right", padx=(8, 0))

        ctk.CTkButton(
            bottom_bar,
            text="Открыть папку отчётов",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=BG_WHITE,
            hover_color=BG_SUBTLE,
            text_color=TEXT_MAIN,
            border_width=1,
            border_color=BORDER_BUTTON,
            height=34,
            width=165,
            command=lambda: subprocess.Popen(["open", str(REPORTS_DIR)]),
        ).pack(side="right", padx=8)

        ctk.CTkButton(
            bottom_bar,
            text="Сгенерировать PNG-отчёт",
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=ACCENT_BLUE,
            hover_color=ACCENT_BLUE_HOVER,
            text_color=BG_WHITE,
            height=34,
            width=185,
            command=self.export_and_open_report_png,
        ).pack(side="right", padx=8)

        self.refresh_history_table()

    # =========================================================================
    # СТРАНИЦА 3: СРАВНЕНИЕ МОДЕЛЕЙ И МЕТОДИКА
    # =========================================================================
    def _build_models_page(self):
        self.page_models.grid_columnconfigure((0, 1), weight=1)
        self.page_models.grid_rowconfigure(1, weight=1)

        cards_row = ctk.CTkFrame(self.page_models, fg_color="transparent")
        cards_row.grid(row=0, column=0, columnspan=2, padx=48, pady=(24, 14), sticky="ew")
        cards_row.grid_columnconfigure((0, 1, 2), weight=1)

        models_info = [
            (
                "EyeCNN (Baseline — только MRL)",
                "Точность на веб-камере: 74.36%",
                "F1-Score: 0.7204  |  MRL Test: 98.75%",
                "Обучена только на инфракрасных снимках MRL Eye Dataset.\nДавала ложные срабатывания при прищуре на обычной веб-камере.",
            ),
            (
                "MobileNetV3 Small (Transfer Learning)",
                "Точность на веб-камере: 58.97%",
                "F1-Score: 0.4419  |  MRL Test: 97.60%",
                "Тяжёлая архитектура ImageNet переобучилась под инфракрасный домен\nи показала низкую устойчивость к теням от дневного света.",
            ),
            (
                "EyeCNN Fine-Tuned + Experience Replay",
                "Точность на веб-камере: 100.00%",
                "F1-Score: 1.0000  |  MRL Test: 97.50%",
                "Дообучена на собственном датасете (1914 кадров) со стабилизацией\nкропа 1:1 и защитой от забывания MRL Experience Replay.",
            ),
        ]

        for idx, (title, acc, f1, desc) in enumerate(models_info):
            card = ctk.CTkFrame(
                cards_row,
                corner_radius=12,
                fg_color=BG_SUBTLE if idx < 2 else BG_HERO,
                border_width=1,
                border_color=BORDER_LIGHT if idx < 2 else ACCENT_BLUE,
            )
            card.grid(row=0, column=idx, padx=8, pady=4, sticky="nsew")
            ctk.CTkLabel(
                card, text=title, font=ctk.CTkFont(size=13, weight="bold"), text_color=TEXT_MAIN
            ).pack(anchor="w", padx=16, pady=(14, 4))
            ctk.CTkLabel(
                card,
                text=acc,
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color=ACCENT_BLUE if idx == 2 else TEXT_MAIN,
            ).pack(anchor="w", padx=16, pady=2)
            ctk.CTkLabel(
                card, text=f1, font=ctk.CTkFont(size=12, weight="bold"), text_color=TEXT_MUTED
            ).pack(anchor="w", padx=16, pady=2)
            ctk.CTkLabel(
                card, text=desc, font=ctk.CTkFont(size=12), text_color=TEXT_MUTED, justify="left", wraplength=330
            ).pack(anchor="w", padx=16, pady=(6, 14))

        cm_frame = ctk.CTkFrame(
            self.page_models,
            corner_radius=12,
            fg_color=BG_SUBTLE,
            border_width=1,
            border_color=BORDER_LIGHT,
        )
        cm_frame.grid(row=1, column=0, padx=(48, 8), pady=(0, 24), sticky="nsew")
        cm_frame.grid_rowconfigure(1, weight=1)
        cm_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            cm_frame,
            text="Матрицы ошибок (Confusion Matrices) на тестовой выборке",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_MAIN,
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
            corner_radius=12,
            fg_color=BG_SUBTLE,
            border_width=1,
            border_color=BORDER_LIGHT,
        )
        algo_frame.grid(row=1, column=1, padx=(8, 48), pady=(0, 24), sticky="nsew")

        ctk.CTkLabel(
            algo_frame,
            text="Описание 7 признаков алгоритма оценки сонливости",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=TEXT_MAIN,
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
        self.lbl_pill_text.configure(
            text=f"Снимок кадра сохранён в файл: results/snapshots/{out_path.name}",
            text_color=ACCENT_BLUE,
        )

    def export_and_open_report_png(self):
        from src.analytics.report_generator import generate_session_report_png

        out_png = generate_session_report_png(self.current_log_path)
        if out_png and out_png.exists():
            self.lbl_report_status.configure(
                text=f"График отчёта сохранён: {out_png.relative_to(ROOT)}",
                text_color=STATUS_GREEN,
            )
            self.lbl_pill_text.configure(
                text=f"Отчёт сессии сформирован: {out_png.name}",
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
            self.video_source_name = f"Файл: {video_path.name}"
        else:
            self.cap = cv2.VideoCapture(0)
            self.is_video_file = False
            self.video_source_name = "Веб-камера"

        if not self.cap.isOpened():
            self.lbl_state.configure(text="Ошибка открытия видеопотока", text_color=STATUS_RED)
            return

        if not self.is_video_file:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

        for tr in self.trackers:
            tr.reset()
        self.evaluator.reset()
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

        self.btn_cam_action.configure(
            text="Остановить",
            fg_color="#FEE2E2",
            border_color="#FCA5A5",
            text_color=STATUS_RED,
        )
        self._update_frame()

    def recalibrate(self):
        if not self.is_running:
            return
        self.calibration_start_mono = time.monotonic()
        for tr in self.trackers:
            tr.reset()
        self.evaluator.reset()
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

        self.btn_cam_action.configure(
            text="Запустить",
            fg_color=BG_WHITE,
            border_color=BORDER_BUTTON,
            text_color=TEXT_MAIN,
        )
        self.lbl_state.configure(text="Мониторинг остановлен", text_color=TEXT_MAIN)
        self.lbl_pill_text.configure(
            text="Сессия завершена и сохранена в разделе «История сессий»",
            text_color=TEXT_MUTED,
        )
        self.video_label.configure(
            image=None,
            text="Мониторинг завершён.\nНажмите «Запустить» или «Создать отчёт» ниже.",
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
            text=f"{mm:02d}:{ss:02d}  •  {fps_val:.1f} FPS  •  {self.video_source_name}"
        )

        h, w = frame.shape[:2]
        elapsed_cal = now - self.calibration_start_mono
        cal_limit = 2.0 if self.is_video_file else CALIBRATION_SECONDS
        is_calibrating = elapsed_cal < cal_limit
        rem_cal = max(0.0, cal_limit - elapsed_cal)

        ts = max(int((now - self.app_start_mono) * 1000), self.last_timestamp_ms + 1)
        self.last_timestamp_ms = ts

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        if res.face_landmarks:
            lm = res.face_landmarks[0]

            if self.show_mesh_var.get():
                for idx in FACE_MESH_KEYPOINTS:
                    px = int(lm[idx].x * w)
                    py = int(lm[idx].y * h)
                    cv2.circle(frame, (px, py), 2, (235, 160, 20), -1, cv2.LINE_AA)

            left_crop = crop_square_eye(frame, lm, LEFT_EYE_LANDMARKS)
            right_crop = crop_square_eye(frame, lm, RIGHT_EYE_LANDMARKS)

            if left_crop is not None and right_crop is not None:
                batch = torch.stack([left_crop[1], right_crop[1]]).to(DEVICE)
                with torch.inference_mode():
                    probs = torch.softmax(self.model(batch), dim=1)[:, 0].cpu().tolist()

                l_ear = calculate_ear(lm, LEFT_EYE_LANDMARKS, w, h)
                r_ear = calculate_ear(lm, RIGHT_EYE_LANDMARKS, w, h)

                l_closed, l_sq, _, l_ratio = self.trackers[0].update(probs[0], l_ear, is_calibrating)
                r_closed, r_sq, _, r_ratio = self.trackers[1].update(probs[1], r_ear, is_calibrating)

                both_closed = l_closed and r_closed
                mean_ear = float(np.nanmean([l_ear, r_ear]))
                mean_ratio = float(np.nanmean([l_ratio, r_ratio]))

                status = self.evaluator.update(
                    now_sec=now,
                    elapsed_session_sec=max(0.1, elapsed_cal - cal_limit),
                    is_calibrating=is_calibrating,
                    both_closed=both_closed,
                    mean_ear=mean_ear,
                    mean_ear_ratio=mean_ratio,
                    landmarks=lm,
                    frame_w=w,
                    frame_h=h,
                )
                self.last_status = status
                if not is_calibrating:
                    self.perclos_samples.append(status["perclos"])

                self.score_history.append(status["score"])
                self.perclos_history.append(status["perclos"])
                if self.metrics_visible:
                    self._draw_live_chart()

                if self.show_boxes_var.get():
                    for crop_data, name, is_c, is_s, prob, ear_v in [
                        (left_crop, "L", l_closed, l_sq, probs[0], l_ear),
                        (right_crop, "R", r_closed, r_sq, probs[1], r_ear),
                    ]:
                        box = crop_data[0]
                        if is_c:
                            col, st = (40, 50, 230), "CLOSED"
                        elif is_s:
                            col, st = (0, 175, 240), "SQUINT"
                        else:
                            col, st = (70, 185, 45), "OPEN"
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), col, 2)
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
                        m_col = (180, 60, 230) if yawn["is_yawning"] else (0, 170, 230)
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

                if status["level"] == DrowsinessEvaluator.LEVEL_HIGH and not is_calibrating:
                    cv2.rectangle(frame, (w // 2 - 235, 14), (w // 2 + 235, 70), (30, 30, 210), -1)
                    cv2.putText(
                        frame,
                        "DROWSINESS ALERT",
                        (w // 2 - 145, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    self._trigger_alert_feedback()

                if self.csv_writer is not None:
                    pose = status["pose"]
                    yawn = status["yawn"]
                    self.csv_writer.writerow([
                        ts, status["level"], f"{status['score']:.2f}", f"{status['perclos']:.2f}",
                        f"{status['blink_rate_per_min']:.2f}", f"{status['avg_blink_duration_ms']:.1f}",
                        f"{status['closure_duration_sec']:.2f}", f"{probs[0]:.4f}", f"{probs[1]:.4f}",
                        f"{l_ear:.4f}", f"{r_ear:.4f}", int(l_closed), int(r_closed), int(both_closed),
                        f"{pose['pitch_deg']:.1f}", f"{pose['yaw_deg']:.1f}", f"{pose['roll_deg']:.1f}",
                        f"{yawn['mar']:.3f}", yawn["total_yawns"]
                    ])

                self._update_telemetry_widgets(
                    status,
                    is_calibrating,
                    rem_cal,
                    left_crop[2],
                    right_crop[2],
                    probs[0],
                    probs[1],
                    l_closed,
                    r_closed,
                )
        else:
            cv2.putText(
                frame,
                "FACE NOT DETECTED",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (40, 120, 220),
                2,
                cv2.LINE_AA,
            )

        self.last_annotated_frame = frame

        disp_w, disp_h = 640, 360
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
    ):
        score = status["score"]
        perclos = status["perclos"]

        self.lbl_score_val.configure(text=f"{score:.1f} / 100")
        self.bar_score.set(min(1.0, max(0.0, score / 100.0)))

        self.lbl_perclos_val.configure(text=f"{perclos:.1f}%")
        self.bar_perclos.set(min(1.0, max(0.0, perclos / 40.0)))

        if is_calibrating:
            state_txt = f"Калибровка ({rem_cal:.1f} с)"
            self.lbl_state.configure(text=state_txt, text_color=ACCENT_BLUE)
            self.bar_score.configure(progress_color=ACCENT_BLUE)
        else:
            lvl = status["level"]
            if lvl == DrowsinessEvaluator.LEVEL_HIGH:
                state_txt = "Высокая сонливость"
                self.lbl_state.configure(text=state_txt, text_color=STATUS_RED)
                self.bar_score.configure(progress_color=STATUS_RED)
            elif lvl == DrowsinessEvaluator.LEVEL_ELEVATED:
                state_txt = "Повышенная усталость"
                self.lbl_state.configure(text=state_txt, text_color=STATUS_AMBER)
                self.bar_score.configure(progress_color=STATUS_AMBER)
            else:
                state_txt = "Нормальное состояние"
                self.lbl_state.configure(text=state_txt, text_color=TEXT_MAIN)
                self.bar_score.configure(progress_color=STATUS_GREEN)

        self.lbl_pill_text.configure(
            text=f"Статус: {state_txt}  |  Индекс сонливости: {score:.1f} из 100  |  PERCLOS: {perclos:.1f}%",
            text_color=TEXT_MAIN if not is_calibrating else ACCENT_BLUE,
        )

        reasons = status.get("reasons", [])
        self.lbl_reasons.configure(
            text=f"Активные факторы риска: {', '.join(reasons)}"
            if reasons
            else "Все биометрические показатели находятся в пределах физиологической нормы."
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
