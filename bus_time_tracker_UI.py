import customtkinter as ctk
from tkinter import messagebox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import os
import threading
import time

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class BusTrackerApp(ctk.CTk):
    """Smart Bus Tracker Application"""

    def __init__(self):
        super().__init__()

        self.title("🚌 Smart Bus Tracker")
        self.geometry("960x750")
        self.resizable(False, False)

        # Refined blue color scheme (same as original, slightly richer)
        self.colors = {
            'primary':   '#1E3A8A',
            'secondary': '#3B82F6',
            'accent':    '#60A5FA',
            'success':   '#10B981',
            'warning':   '#F59E0B',
            'danger':    '#EF4444',
            'dark':      '#1E293B',
            'darker':    '#0F172A',
            'light':     '#F1F5F9',
            'text':      '#E2E8F0',
            'subtext':   '#94A3B8',
        }

        # Data
        self.df_processed    = None
        self.model_info      = None
        self.feature_columns = None
        self.bus_stops       = []
        self.buses           = []

        # Animation state
        self.animation_running = False
        self.loading_dots      = 0
        self._header_tick      = 0

        try:
            self.load_data()
        except Exception as e:
            print(f"Data load error: {e}")
            self.create_sample_data()

        if not self.bus_stops:
            self.create_sample_data()

        self.create_ui()
        self.center_window()
        self._start_clock()

    # ─────────────────────────────────────────────
    # Window helpers
    # ─────────────────────────────────────────────
    def center_window(self):
        w, h = 960, 750
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.resizable(False, False)

    # ─────────────────────────────────────────────
    # Data loading
    # ─────────────────────────────────────────────
    def load_data(self):
        if os.path.exists('bus_data.csv'):
            df = pd.read_csv('bus_data.csv')
            self.df_processed = self.preprocess_data(df)
            self.bus_stops = sorted([str(x) for x in self.df_processed['bus_stop'].unique()])
            self.buses     = sorted([str(x) for x in self.df_processed['deviceid'].unique()])
        else:
            self.create_sample_data()

        if os.path.exists('bus_arrival_mlr_model.pkl'):
            md = joblib.load('bus_arrival_mlr_model.pkl')
            self.model_info      = md
            self.feature_columns = md.get('feature_columns', [])
