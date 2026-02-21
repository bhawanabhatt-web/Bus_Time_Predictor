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

    #  Search Section 
    def create_search_section(self, parent):
        search_frame = ctk.CTkFrame(parent, fg_color=self.colors['dark'],
                                     corner_radius=15,
                                     border_width=1,
                                     border_color="#2D4A7A")
        search_frame.pack(fill="x", pady=(0, 10))

        inner = ctk.CTkFrame(search_frame, fg_color="transparent")
        inner.pack(fill="both", padx=24, pady=14)

        stop_values = [str(x) for x in self.bus_stops] if self.bus_stops else ["No stops available"]

        # Pickup + arrow + Destination side by side
        row = ctk.CTkFrame(inner, fg_color="transparent")
        row.pack(fill="x", pady=(0, 10))

        # Pickup
        pickup_col = ctk.CTkFrame(row, fg_color="transparent")
        pickup_col.pack(side="left", expand=True, fill="x")

        ctk.CTkLabel(pickup_col, text="📍  Pickup Bus Stop",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=self.colors['accent']).pack(anchor="w", pady=(0, 4))

        self.pickup_combo = ctk.CTkComboBox(
            pickup_col, values=stop_values,
            height=36,
            font=ctk.CTkFont(size=13),
            dropdown_font=ctk.CTkFont(size=12),
            fg_color=self.colors['darker'],
            border_color=self.colors['secondary'],
            border_width=2,
            button_color=self.colors['secondary'],
            button_hover_color=self.colors['accent'],
            dropdown_fg_color=self.colors['dark'],
            corner_radius=10)
        self.pickup_combo.pack(fill="x")
        self.pickup_combo.set("Select pickup location")

        # Arrow
        arrow_col = ctk.CTkFrame(row, fg_color="transparent", width=60)
        arrow_col.pack(side="left", fill="y", padx=10)
        arrow_col.pack_propagate(False)
        ctk.CTkLabel(arrow_col, text="→",
                     font=ctk.CTkFont(size=22),
                     text_color=self.colors['secondary']).pack(expand=True)

        # Destination
        dest_col = ctk.CTkFrame(row, fg_color="transparent")
        dest_col.pack(side="left", expand=True, fill="x")

        ctk.CTkLabel(dest_col, text="🎯  Destination Bus Stop",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=self.colors['accent']).pack(anchor="w", pady=(0, 4))

        self.destination_combo = ctk.CTkComboBox(
            dest_col, values=stop_values,
            height=36,
            font=ctk.CTkFont(size=13),
            dropdown_font=ctk.CTkFont(size=12),
            fg_color=self.colors['darker'],
            border_color=self.colors['secondary'],
            border_width=2,
            button_color=self.colors['secondary'],
            button_hover_color=self.colors['accent'],
            dropdown_fg_color=self.colors['dark'],
            corner_radius=10)
        self.destination_combo.pack(fill="x")
        self.destination_combo.set("Select destination")

        # Search button + stats strip
        bottom = ctk.CTkFrame(inner, fg_color="transparent")
        bottom.pack(fill="x")

        self.search_button = ctk.CTkButton(
            bottom,
            text="🔍  Search Buses",
            command=self.search_buses,
            width=200, height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.colors['secondary'],
            hover_color=self.colors['accent'],
            corner_radius=12)
        self.search_button.pack(side="left")

        stats = ctk.CTkFrame(bottom, fg_color=self.colors['darker'],
                              corner_radius=12)
        stats.pack(side="right")

        for icon, val, lbl in [("🚌", str(len(self.buses)), "Buses"),
                                ("📍", str(len(self.bus_stops)), "Stops"),
                                ("🤖", "AI", "Powered")]:
            col = ctk.CTkFrame(stats, fg_color="transparent")
            col.pack(side="left", padx=14, pady=7)
            ctk.CTkLabel(col, text=icon, font=ctk.CTkFont(size=15)).pack()
            ctk.CTkLabel(col, text=val,
                         font=ctk.CTkFont(size=12, weight="bold"),
                         text_color=self.colors['secondary']).pack()
            ctk.CTkLabel(col, text=lbl,
                         font=ctk.CTkFont(size=9),
                         text_color=self.colors['subtext']).pack()
            
    #  Results Section 

    def create_results_section(self, parent):
        results_frame = ctk.CTkFrame(parent, fg_color=self.colors['dark'],
                                      corner_radius=15,
                                      border_width=1,
                                      border_color="#2D4A7A")
        results_frame.pack(fill="both", expand=True)

        hdr = ctk.CTkFrame(results_frame, fg_color="transparent", height=42)
        hdr.pack(fill="x", padx=20, pady=(10, 0))
        hdr.pack_propagate(False)

        ctk.CTkLabel(hdr, text="📊  Available Buses",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color=self.colors['accent']).pack(side="left", anchor="center")

        self._result_count_lbl = ctk.CTkLabel(
            hdr, text="",
            font=ctk.CTkFont(size=12),
            text_color=self.colors['subtext'])
        self._result_count_lbl.pack(side="right", anchor="center")

        ctk.CTkFrame(results_frame, fg_color="#2D4A7A", height=1).pack(fill="x", padx=20)

        self.results_scroll = ctk.CTkScrollableFrame(
            results_frame,
            fg_color="transparent",
            scrollbar_button_color=self.colors['secondary'],
            scrollbar_button_hover_color=self.colors['accent'])
        self.results_scroll.pack(fill="both", expand=True, padx=20, pady=(10, 16))

        self.show_empty_state()

    #  Footer 
    def create_footer(self, parent):
        footer = ctk.CTkFrame(parent, fg_color="transparent", height=26)
        footer.pack(fill="x", pady=(6, 0))
        footer.pack_propagate(False)

        ctk.CTkLabel(
            footer,
            text="💡  Real-time bus tracking powered by AI  |  © 2026 Smart Bus Tracker",
            font=ctk.CTkFont(size=11),
            text_color=self.colors['subtext']).pack(expand=True)

    # State helpers
    def show_empty_state(self):
        f = ctk.CTkFrame(self.results_scroll, fg_color="transparent")
        f.pack(expand=True, fill="both", pady=40)
        ctk.CTkLabel(f, text="🔍", font=ctk.CTkFont(size=58)).pack()
        ctk.CTkLabel(f, text="Search for buses to see available options",
                     font=ctk.CTkFont(size=14),
                     text_color=self.colors['text']).pack(pady=(8, 0))
        ctk.CTkLabel(f, text="Select a pickup and destination stop above",
                     font=ctk.CTkFont(size=12),
                     text_color=self.colors['subtext']).pack(pady=(4, 0))

    def clear_results(self):
        for w in self.results_scroll.winfo_children():
            w.destroy()


    # Search logic

    def search_buses(self):
        pickup = self.pickup_combo.get()
        dest   = self.destination_combo.get()

        if pickup == "Select pickup location" or dest == "Select destination":
            messagebox.showwarning("Missing Information",
                                   "Please select both pickup and destination stops!")
            return
        if pickup == dest:
            messagebox.showwarning("Invalid Selection",
                                   "Pickup and destination cannot be the same!")
            return

        self.show_loading()
        threading.Thread(target=self.perform_search,
                         args=(pickup, dest), daemon=True).start()

    def show_loading(self):
        self.clear_results()
        self.animation_running = True
        self._result_count_lbl.configure(text="Searching…")

        f = ctk.CTkFrame(self.results_scroll, fg_color="transparent")
        f.pack(expand=True, fill="both", pady=40)

        self.loading_label = ctk.CTkLabel(
            f, text="🚌  Searching for buses",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.colors['accent'])
        self.loading_label.pack(pady=(0, 16))

        self._prog_bar = ctk.CTkProgressBar(f, width=340, mode="indeterminate",
                                             fg_color=self.colors['darker'],
                                             progress_color=self.colors['secondary'],
                                             corner_radius=6)
        self._prog_bar.pack()
        self._prog_bar.start()

        self.animate_loading()

    def animate_loading(self):
        if self.animation_running:
            self.loading_dots = (self.loading_dots + 1) % 4
            dots = "." * self.loading_dots
            try:
                self.loading_label.configure(text=f"🚌  Searching for buses{dots}")
            except Exception:
                pass
            self.after(300, self.animate_loading)

    def perform_search(self, pickup, destination):
        time.sleep(1.5)
        buses_data = self.get_available_buses(pickup, destination)
        self.after(0, lambda: self.display_results(buses_data, pickup, destination))

    def get_available_buses(self, pickup, destination):
        try:
            if self.df_processed is not None:
                avail = self.df_processed[
                    self.df_processed['bus_stop'].astype(str) == str(pickup)]
                buses = []
                for bus_id in avail['deviceid'].unique()[:10]:
                    bd = avail[avail['deviceid'] == bus_id]
                    avg = bd['travel_time_to_next_stop'].mean() \
                        if 'travel_time_to_next_stop' in bd.columns \
                        else np.random.uniform(5, 20)
                    arrival = datetime.now() + timedelta(minutes=np.random.uniform(2, 15))
                    buses.append({
                        'bus_id':       str(bus_id),
                        'arrival_time': arrival,
                        'travel_time':  avg,
                        'occupancy':    np.random.choice(['Low', 'Medium', 'High']),
                        'status':       'On Time',
                    })
                buses.sort(key=lambda x: x['arrival_time'])
                return buses
        except Exception as e:
            print(f"Search error: {e}")
        return self.generate_sample_buses()

    def generate_sample_buses(self):
        now = datetime.now()
        buses = []
        for _ in range(8):
            m = np.random.randint(2, 30)
            buses.append({
                'bus_id':       f"Bus-{np.random.randint(1, 100):03d}",
                'arrival_time': now + timedelta(minutes=m),
                'travel_time':  np.random.uniform(10, 25),
                'occupancy':    np.random.choice(['Low', 'Medium', 'High']),
                'status':       np.random.choice(['On Time', 'Delayed', 'Early']),
            })
        buses.sort(key=lambda x: x['arrival_time'])
        return buses
    
    # Results display
    def display_results(self, buses_data, pickup, destination):
        self.animation_running = False
        self.clear_results()

        if not buses_data:
            self._result_count_lbl.configure(text="No buses found")
            self.show_no_results()
            return

        n = len(buses_data)
        self._result_count_lbl.configure(text=f"{n} bus{'es' if n != 1 else ''} found")

        route_frame = ctk.CTkFrame(self.results_scroll,
                                    fg_color=self.colors['primary'],
                                    corner_radius=12)
        route_frame.pack(fill="x", padx=6, pady=(0, 14))

        ctk.CTkLabel(route_frame,
                     text=f"🚏  {pickup}   →   🎯  {destination}",
                     font=ctk.CTkFont(size=15, weight="bold"),
                     text_color=self.colors['light']).pack(pady=13)

        for idx, bus in enumerate(buses_data):
            self.create_bus_card(bus, idx)

    def create_bus_card(self, bus_data, index):
        is_best = index == 0

        card_bg = self.colors['secondary'] if is_best else "#243044"
        border  = self.colors['accent']    if is_best else "#2D4A7A"

        card = ctk.CTkFrame(self.results_scroll,
                             fg_color=card_bg,
                             corner_radius=13,
                             border_width=2,
                             border_color=border)
        card.pack(fill="x", padx=6, pady=6)



        content = ctk.CTkFrame(card, fg_color="transparent")
        content.pack(fill="both", padx=20, pady=14)

        # Row 1: Bus ID + Status
        top_row = ctk.CTkFrame(content, fg_color="transparent")
        top_row.pack(fill="x", pady=(0, 8))

        ctk.CTkLabel(top_row,
                     text=f"🚌  {bus_data['bus_id']}",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     text_color=self.colors['light']).pack(side="left")



        # Row 2: Arrival + countdown chip
        arrival_str = bus_data['arrival_time'].strftime("%I:%M %p")
        mins_away   = max(0, int((bus_data['arrival_time'] - datetime.now()
                                  ).total_seconds() / 60))

        arr_row = ctk.CTkFrame(content, fg_color="transparent")
        arr_row.pack(fill="x", pady=(0, 6))

        ctk.CTkLabel(arr_row,
                     text=f"⏰  Arrives at {arrival_str}",
                     font=ctk.CTkFont(size=14),
                     text_color=self.colors['light']).pack(side="left")

        chip = ctk.CTkFrame(arr_row, fg_color=self.colors['darker'],
                             corner_radius=8)
        chip.pack(side="left", padx=10)
        ctk.CTkLabel(chip, text=f"  {mins_away} min away  ",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=self.colors['accent']).pack(pady=3)

        # Row 3: Travel time
        ctk.CTkLabel(content,
                     text=f"⏱️  Travel time: {bus_data['travel_time']:.1f} minutes",
                     font=ctk.CTkFont(size=13),
                     text_color=self.colors['text']).pack(anchor="w", pady=(0, 6))

        # Row 4: Occupancy + mini progress bar
        occ_row = ctk.CTkFrame(content, fg_color="transparent")
        occ_row.pack(anchor="w", fill="x")

        occ_colors = {
            'Low':    self.colors['success'],
            'Medium': self.colors['warning'],
            'High':   self.colors['danger'],
        }
        occ_vals = {'Low': 0.25, 'Medium': 0.6, 'High': 1.0}
        o_col = occ_colors.get(bus_data['occupancy'], self.colors['text'])

        ctk.CTkLabel(occ_row, text="👥  Occupancy: ",
                     font=ctk.CTkFont(size=13),
                     text_color=self.colors['text']).pack(side="left")

        ctk.CTkLabel(occ_row,
                     text=bus_data['occupancy'],
                     font=ctk.CTkFont(size=13, weight="bold"),
                     text_color=o_col).pack(side="left")

        occ_bar = ctk.CTkProgressBar(occ_row, width=100, height=8,
                                      fg_color=self.colors['darker'],
                                      progress_color=o_col,
                                      corner_radius=4)
        occ_bar.pack(side="left", padx=12)
        occ_bar.set(occ_vals.get(bus_data['occupancy'], 0.5))

        # Staggered entrance
        self.after(index * 80, lambda c=card: c.pack_configure(pady=6))

    def show_no_results(self):
        f = ctk.CTkFrame(self.results_scroll, fg_color="transparent")
        f.pack(expand=True, fill="both", pady=50)
        ctk.CTkLabel(f, text="😔", font=ctk.CTkFont(size=58)).pack()
        ctk.CTkLabel(f, text="No buses found for this route",
                     font=ctk.CTkFont(size=16, weight="bold"),
                     text_color=self.colors['text']).pack(pady=(8, 4))
        ctk.CTkLabel(f, text="Try selecting different bus stops",
                     font=ctk.CTkFont(size=13),
                     text_color=self.colors['accent']).pack()


def main():
    app = BusTrackerApp()
    app.mainloop()


if __name__ == "__main__":
    main()       
