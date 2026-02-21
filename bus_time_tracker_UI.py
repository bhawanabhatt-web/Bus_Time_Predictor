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

    
    # Window helpers
    def center_window(self):
        w, h = 960, 750
        sw = self.winfo_screenwidth()
        sh = self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        self.resizable(False, False)

    
    # Data loading

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
    def create_sample_data(self):
        self.bus_stops = [
            "Central Station", "City Mall", "University Gate",
            "Hospital Junction", "Park Avenue", "Airport Terminal",
            "Tech Park", "Railway Station", "Bus Depot", "Downtown"
        ]
        self.buses = [f"Bus-{i:03d}" for i in range(1, 21)]
        sample = {
            'bus_stop':                 np.random.choice(self.bus_stops, 1000),
            'deviceid':                 np.random.choice(self.buses, 1000),
            'travel_time_to_next_stop': np.random.uniform(2, 15, 1000)
        }
        self.df_processed = pd.DataFrame(sample)
        self.bus_stops = [str(x) for x in self.bus_stops]
        self.buses     = [str(x) for x in self.buses]

    def preprocess_data(self, df):
        d = df.copy()
        if 'date' in df.columns and 'arrival_time' in df.columns:
            d['arrival_datetime'] = pd.to_datetime(
                d['date'] + ' ' + d['arrival_time'], errors='coerce')
        return d.dropna()

    # Live clock

    def _start_clock(self):
        def tick():
            if hasattr(self, '_clock_label'):
                now = datetime.now().strftime("%I:%M:%S %p  |  %a %d %b %Y")
                try:
                    self._clock_label.configure(text=now)
                except Exception:
                    pass
            self.after(1000, tick)
        tick()

    # UI construction 
    
    def create_ui(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill="both", expand=True, padx=16, pady=14)

        self.create_header(main)
        self.create_search_section(main)
        self.create_results_section(main)
        self.create_footer(main)

    # Header 
    def create_header(self, parent):
        header = ctk.CTkFrame(parent, fg_color=self.colors['primary'],
                               corner_radius=15, height=72)
        header.pack(fill="x", pady=(0, 10))
        header.pack_propagate(False)

        # Left: icon + title
        left = ctk.CTkFrame(header, fg_color="transparent")
        left.pack(side="left", padx=20, pady=8, fill="y")

        ctk.CTkLabel(left, text="🚌",
                     font=ctk.CTkFont(size=28)).pack(side="left", padx=(0, 10))

        title_stack = ctk.CTkFrame(left, fg_color="transparent")
        title_stack.pack(side="left", fill="y")

        self._title_lbl = ctk.CTkLabel(
            title_stack,
            text="Smart Bus Tracker",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=self.colors['light'])
        self._title_lbl.pack(anchor="w")

        ctk.CTkLabel(
            title_stack,
            text="Real-time AI-powered bus predictions",
            font=ctk.CTkFont(size=11),
            text_color=self.colors['accent']).pack(anchor="w")

        # Right: live clock + online badge
        right = ctk.CTkFrame(header, fg_color="transparent")
        right.pack(side="right", padx=16, fill="y")

        pill = ctk.CTkFrame(right, fg_color="#0F2A50",
                             corner_radius=20,
                             border_width=1,
                             border_color=self.colors['success'])
        pill.pack(anchor="e", pady=(10, 4))

        ctk.CTkLabel(pill, text="● LIVE",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=self.colors['success']).pack(side="left", padx=(8, 4), pady=4)
        ctk.CTkLabel(pill, text="System Online",
                     font=ctk.CTkFont(size=10),
                     text_color=self.colors['text']).pack(side="left", padx=(0, 10), pady=4)

        self._clock_label = ctk.CTkLabel(
            right, text="",
            font=ctk.CTkFont(size=10),
            text_color=self.colors['subtext'])
        self._clock_label.pack(anchor="e")

        self._animate_header(header)

    def _animate_header(self, frame):
        shades = ['#1E3A8A', '#243EA0', '#1E3A8A', '#1A347E']

        def pulse():
            col = shades[self._header_tick % len(shades)]
            try:
                frame.configure(fg_color=col)
            except Exception:
                pass
            self._header_tick += 1
            self.after(900, pulse)

        pulse()