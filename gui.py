import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading
import time
from collections import deque
import numpy as np
from utils import load_model, process_frame, process_video_frame, process_image, EnhancedRoadTracker

class TrafficOptimizationApp:
    
    def __init__(self):
        # System variables - ENHANCED
        self.model = None
        self.road_tracker = EnhancedRoadTracker(roads=2, base_cycle=120, min_green=8, max_green=50)
        self.current_frame = None
        self.is_video_playing = False
        self.video_capture = None
        self.processing_thread = None
        self.current_road_boundaries = []
        self.detected_roads = 2
        
        # Performance optimization variables
        self.frame_skip = 2
        self.frame_counter = 0
        self.target_fps = 20
        self.resize_factor = 0.8
        
        # GUI variables
        self.model_path = tk.StringVar(value="yolov8s.pt")
        self.media_path = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="Ready - Load model to start")
        self.confidence = tk.DoubleVar(value=0.40)
        
        # Shadow removal variables - NEW!
        self.shadow_removal_enabled = tk.BooleanVar(value=False)
        self.shadow_removal_performance = {
            'accuracy_improvement': 0.0,
            'processing_time_ms': 0.0
        }
        
        # Enhanced analytics variables
        self.last_frame_time = time.time()
        self.enhanced_analytics_data = {}
        
        # Create both windows with positioning
        self.create_main_display_window()
        self.create_analytics_window()
        
        # Start UI updates
        self.start_ui_updates()
        
        # Try to load default model
        try:
            self.model = load_model(self.model_path.get())
            self.status_text.set("✅ YOLO model loaded - Ready to process")
        except:
            self.status_text.set("⚠️ Default model not found - Please load a model")
    
    def create_main_display_window(self):
        self.main_window = tk.Toplevel()
        self.main_window.title("🛣️ Traffic Display")
        self.main_window.geometry("800x650+50+50")
        self.main_window.configure(bg="#1a1a1a")
        
        # Setup styles for main window
        self.setup_styles()
        
        # Create header with controls
        self.create_main_header()
        
        # Create media display section
        self.create_media_display()
        
        # Create improved control buttons
        self.create_main_controls()
        
        # Create status bar for main window
        self.create_main_status_bar()
        
        # Handle window close
        self.main_window.protocol("WM_DELETE_WINDOW", self.on_main_window_close)
    
    def create_analytics_window(self):
        self.analytics_window = tk.Toplevel()
        self.analytics_window.title("📊 Traffic Analytics")
        self.analytics_window.geometry("1200x750+900+50")  # WIDER for dual tables
        self.analytics_window.configure(bg="#1a1a1a")
        
        # Create analytics sections
        self.create_control_panel()
        self.create_priority_section()
        self.create_dual_analytics_dashboard()  # NEW: DUAL TABLES!
        
        # Handle window close
        self.analytics_window.protocol("WM_DELETE_WINDOW", self.on_analytics_window_close)
    
    def setup_styles(self):
        try:
            style = ttk.Style()
            style.theme_use("clam")
            
            # Configure custom styles
            style.configure("Title.TLabel", font=("Helvetica", 14, "bold"),
                          background="#1a1a1a", foreground="#ffffff")
            style.configure("Header.TLabel", font=("Helvetica", 11, "bold"),
                          background="#1a1a1a", foreground="#00ff00")
            style.configure("Priority.TLabel", font=("Helvetica", 14, "bold"),
                          background="#1a1a1a", foreground="#ff4444")
            style.configure("Status.TLabel", font=("Helvetica", 9),
                          background="#1a1a1a", foreground="#ffffff")
            
            # IMPROVED BUTTON STYLES
            style.configure("Load.TButton", 
                          background="#ff6b00", foreground="white", font=("Helvetica", 10, "bold"),
                          borderwidth=2, relief="raised")
            style.configure("LoadVideo.TButton", 
                          background="#00aa44", foreground="white", font=("Helvetica", 10, "bold"),
                          borderwidth=2, relief="raised")
            style.configure("Start.TButton", 
                          background="#0066ff", foreground="white", font=("Helvetica", 12, "bold"),
                          borderwidth=3, relief="raised")
            style.configure("Stop.TButton", 
                          background="#ff4444", foreground="white", font=("Helvetica", 10, "bold"),
                          borderwidth=2, relief="raised")
            style.configure("Process.TButton", 
                          background="#aa00aa", foreground="white", font=("Helvetica", 10, "bold"),
                          borderwidth=2, relief="raised")
            
        except Exception as e:
            print(f"Style setup error: {e}")
    
    def create_main_header(self):
        try:
            header_frame = tk.Frame(self.main_window, bg="#1a1a1a", height=80)
            header_frame.pack(fill="x", padx=10, pady=5)
            header_frame.pack_propagate(False)
            
            # Title
            title_label = ttk.Label(header_frame, text="Traffic System",
                                   style="Title.TLabel")
            title_label.pack(pady=(5, 10))
            
            # File paths display
            paths_frame = tk.Frame(header_frame, bg="#1a1a1a")
            paths_frame.pack(fill="x", padx=5)
            
            # Model path
            model_frame = tk.Frame(paths_frame, bg="#1a1a1a")
            model_frame.pack(fill="x", pady=2)
            
            tk.Label(model_frame, text="Model:", bg="#1a1a1a", fg="white", 
                    font=("Helvetica", 9, "bold"), width=8, anchor="w").pack(side="left")
            tk.Label(model_frame, textvariable=self.model_path, bg="#1a1a1a", fg="#ffaa00",
                    font=("Helvetica", 8), anchor="w").pack(side="left", fill="x", expand=True)
            
            # Media path
            media_frame = tk.Frame(paths_frame, bg="#1a1a1a")
            media_frame.pack(fill="x", pady=2)
            
            tk.Label(media_frame, text="Video:", bg="#1a1a1a", fg="white", 
                    font=("Helvetica", 9, "bold"), width=8, anchor="w").pack(side="left")
            tk.Label(media_frame, textvariable=self.media_path, bg="#1a1a1a", fg="#ffaa00",
                    font=("Helvetica", 8), anchor="w").pack(side="left", fill="x", expand=True)
            
        except Exception as e:
            print(f"Main header creation error: {e}")
    
    def create_media_display(self):
        try:
            # Media display frame
            display_frame = tk.Frame(self.main_window, bg="#2a2a2a")
            display_frame.pack(fill="both", expand=True, padx=10, pady=5)
            
            # Media display header
            media_header = tk.Frame(display_frame, bg="#2a2a2a", height=35)
            media_header.pack(fill="x", padx=5, pady=5)
            media_header.pack_propagate(False)
            
            ttk.Label(media_header, text="📺 Feed", style="Header.TLabel").pack(side="left")
            
            # Shadow removal indicator
            self.shadow_status_indicator = tk.Label(media_header, text="🌑 OFF", bg="#2a2a2a", fg="#888888",
                                                   font=("Helvetica", 9, "bold"))
            self.shadow_status_indicator.pack(side="right")
            
            # Canvas for media display
            self.media_canvas = tk.Canvas(display_frame, bg="#000000", highlightthickness=2,
                                        highlightcolor="#444444")
            self.media_canvas.pack(fill="both", expand=True, padx=5, pady=5)
            
        except Exception as e:
            print(f"Media display creation error: {e}")
    
    def create_main_controls(self):
        try:
            controls_frame = tk.Frame(self.main_window, bg="#1a1a1a", height=80)
            controls_frame.pack(fill="x", padx=10, pady=5)
            controls_frame.pack_propagate(False)
            
            # Button container
            button_container = tk.Frame(controls_frame, bg="#1a1a1a")
            button_container.pack(expand=True)
            
            # LOAD MODEL BUTTON with icon
            load_model_btn = ttk.Button(button_container, text="Load YOLO Model", 
                                       command=self.load_model, style="Load.TButton", width=18)
            load_model_btn.pack(side="left", padx=10, pady=20)
            
            # LOAD VIDEO BUTTON with icon
            load_video_btn = ttk.Button(button_container, text="Load Video File", 
                                       command=self.load_video, style="LoadVideo.TButton", width=18)
            load_video_btn.pack(side="left", padx=10, pady=20)
            
            # START BUTTON - BIG and prominent
            start_btn = ttk.Button(button_container, text="Start Processing", 
                                  command=self.start_processing, style="Start.TButton", width=20)
            start_btn.pack(side="left", padx=15, pady=20)
            
            # PROCESS SINGLE FRAME BUTTON
            process_btn = ttk.Button(button_container, text="Process Frame", 
                                    command=self.process_single_frame, style="Process.TButton", width=16)
            process_btn.pack(side="left", padx=10, pady=20)
            
            # STOP BUTTON
            stop_btn = ttk.Button(button_container, text="Stop", 
                                 command=self.stop_processing, style="Stop.TButton", width=12)
            stop_btn.pack(side="left", padx=10, pady=20)
            
        except Exception as e:
            print(f"Main controls creation error: {e}")
    
    def create_main_status_bar(self):
        try:
            status_frame = tk.Frame(self.main_window, bg="#2a2a2a", height=30)
            status_frame.pack(fill="x", side="bottom")
            status_frame.pack_propagate(False)
            
            ttk.Label(status_frame, textvariable=self.status_text, style="Status.TLabel").pack(side="left", padx=10, pady=6)
            
            self.time_label = tk.Label(status_frame, text="", bg="#2a2a2a", fg="#888888",
                                     font=("Helvetica", 8))
            self.time_label.pack(side="right", padx=10, pady=6)
            
        except Exception as e:
            print(f"Main status bar creation error: {e}")
    
    def create_control_panel(self):
        try:
            control_frame = tk.LabelFrame(self.analytics_window, text="⚙️ Detection & Shadow Settings",
                                        bg="#2a2a2a", fg="white", font=("Helvetica", 12, "bold"))
            control_frame.pack(fill="x", padx=10, pady=10)
            
            # Confidence threshold
            conf_frame = tk.Frame(control_frame, bg="#2a2a2a")
            conf_frame.pack(fill="x", padx=20, pady=15)
            
            tk.Label(conf_frame, text="Detection Confidence:", bg="#2a2a2a", fg="white",
                    font=("Helvetica", 10, "bold")).pack(anchor="w")
            
            conf_control_frame = tk.Frame(conf_frame, bg="#2a2a2a")
            conf_control_frame.pack(fill="x", pady=(6, 0))
            
            conf_scale = ttk.Scale(conf_control_frame, from_=0.05, to=0.8, variable=self.confidence,
                                 orient="horizontal", length=350)
            conf_scale.pack(side="left", expand=True, fill="x")
            
            self.conf_label = tk.Label(conf_control_frame, text="0.40", bg="#2a2a2a", fg="#00ff00",
                                     font=("Helvetica", 12, "bold"), width=6)
            self.conf_label.pack(side="right", padx=(10, 0))
            self.confidence.trace("w", self.update_conf_label)
            
            # SHADOW REMOVAL SECTION
            shadow_frame = tk.Frame(control_frame, bg="#2a2a2a")
            shadow_frame.pack(fill="x", padx=20, pady=(15, 15))
            
            tk.Label(shadow_frame, text="🌑 Shadow Removal (HSV + Morphological):", bg="#2a2a2a", fg="#ffaa00",
                    font=("Helvetica", 10, "bold")).pack(anchor="w")
            
            shadow_control_frame = tk.Frame(shadow_frame, bg="#2a2a2a")
            shadow_control_frame.pack(fill="x", pady=(6, 0))
            
            # Shadow removal toggle
            shadow_toggle = tk.Checkbutton(
                shadow_control_frame, 
                text="Enable Shadow Removal for Enhanced Accuracy", 
                variable=self.shadow_removal_enabled,
                bg="#2a2a2a", fg="white", selectcolor="#4a4a4a",
                activebackground="#2a2a2a", activeforeground="white",
                font=("Helvetica", 9, "bold"),
                command=self.on_shadow_removal_toggle
            )
            shadow_toggle.pack(anchor="w")
            
            # Shadow removal performance info (NO FPS MENTIONED)
            
        except Exception as e:
            print(f"Control panel creation error: {e}")
    
    def create_priority_section(self):
        try:
            priority_frame = tk.LabelFrame(self.analytics_window, text="🏆 Traffic Solution",
                                         bg="#2a2a2a", fg="white", font=("Helvetica", 12, "bold"))
            priority_frame.pack(fill="x", padx=10, pady=10)
            
            # Priority road display
            priority_display_frame = tk.Frame(priority_frame, bg="#2a2a2a")
            priority_display_frame.pack(fill="x", padx=15, pady=10)
            
            tk.Label(priority_display_frame, text="Priority Road (Clear First):",
                    bg="#2a2a2a", fg="white", font=("Helvetica", 10, "bold")).pack(anchor="w")
            
            self.priority_road_label = tk.Label(priority_display_frame, text="Road 2 (50s green)",
                                              bg="#2a2a2a", fg="#ff4444",
                                              font=("Helvetica", 16, "bold"))
            self.priority_road_label.pack(anchor="w", pady=(6, 10))
            
            # Congested roads display
            tk.Label(priority_display_frame, text="Congested Roads:",
                    bg="#2a2a2a", fg="white", font=("Helvetica", 10, "bold")).pack(anchor="w")
            
            self.congested_roads_label = tk.Label(priority_display_frame, text="None detected",
                                                 bg="#2a2a2a", fg="#ffa500",
                                                 font=("Helvetica", 11, "bold"))
            self.congested_roads_label.pack(anchor="w", pady=(4, 0))
            
        except Exception as e:
            print(f"Priority section creation error: {e}")
    
    def create_dual_analytics_dashboard(self):
        try:
            analytics_frame = tk.LabelFrame(self.analytics_window, text="📊 Analytics Dashboard",
                                          bg="#2a2a2a", fg="white", font=("Helvetica", 12, "bold"))
            analytics_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Container for dual tables
            tables_container = tk.Frame(analytics_frame, bg="#2a2a2a")
            tables_container.pack(fill="both", expand=True, padx=5, pady=5)
            
            # LEFT SIDE - Traffic Analytics Table
            left_frame = tk.Frame(tables_container, bg="#2a2a2a")
            left_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
            
            tk.Label(left_frame, text="Traffic Analytics", bg="#2a2a2a", fg="#00ff00",
                    font=("Helvetica", 11, "bold")).pack(pady=(0, 5))
            
            # Traffic Analytics table (WITHOUT Speed column)
            traffic_columns = ("Road", "Vehicles", "Density%", "Priority", "Green_Time", "Wait_Time", "Accuracy%")
            self.traffic_tree = ttk.Treeview(left_frame, columns=traffic_columns, show="headings", height=8)
            
            # Configure Traffic columns
            traffic_widths = {"Road": 50, "Vehicles": 60, "Density%": 65, "Priority": 65,
                            "Green_Time": 75, "Wait_Time": 75, "Accuracy%": 75}
            
            for col in traffic_columns:
                self.traffic_tree.heading(col, text=col)
                self.traffic_tree.column(col, width=traffic_widths[col], anchor="center")
            
            self.traffic_tree.pack(fill="both", expand=True)
            
            # RIGHT SIDE - ML Metrics Table
            right_frame = tk.Frame(tables_container, bg="#2a2a2a")
            right_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
            
            tk.Label(right_frame, text="ML Performance Metrics", bg="#2a2a2a", fg="#ff8800",
                    font=("Helvetica", 11, "bold")).pack(pady=(0, 5))
            
            # ML Metrics table - NEW!
            ml_columns = ("Road", "Precision", "Recall", "F1_Score", "True_Pos", "False_Pos")
            self.ml_tree = ttk.Treeview(right_frame, columns=ml_columns, show="headings", height=8)
            
            # Configure ML columns
            ml_widths = {"Road": 50, "Precision": 70, "Recall": 70, "F1_Score": 70,
                        "True_Pos": 65, "False_Pos": 65}
            
            for col in ml_columns:
                self.ml_tree.heading(col, text=col)
                self.ml_tree.column(col, width=ml_widths[col], anchor="center")
            
            self.ml_tree.pack(fill="both", expand=True)
            
            # Bottom metrics (NO FPS)
            metrics_frame = tk.Frame(analytics_frame, bg="#2a2a2a")
            metrics_frame.pack(fill="x", padx=10, pady=5)
            
            # Performance metrics without FPS
            self.total_vehicles_label = tk.Label(metrics_frame, text="Total Vehicles: 0",
                                               bg="#2a2a2a", fg="#00ff00",
                                               font=("Helvetica", 11, "bold"))
            self.total_vehicles_label.pack(side="left")
            
            self.avg_accuracy_label = tk.Label(metrics_frame, text="Avg Accuracy: 0.0%", bg="#2a2a2a",
                                             fg="#ffaa00", font=("Helvetica", 11, "bold"))
            self.avg_accuracy_label.pack(side="left", padx=(30, 0))
            
        except Exception as e:
            print(f"Dual analytics dashboard creation error: {e}")
    
    def on_shadow_removal_toggle(self):
        """Handle shadow removal toggle"""
        try:
            if self.shadow_removal_enabled.get():
                print("✅ HSV Shadow Removal enabled")
                self.shadow_status_indicator.config(text="🌑 ON", fg="#00ff00")
                self.status_text.set("🌑 Shadow removal enabled - Enhanced accuracy mode")
            else:
                print("❌ HSV Shadow Removal disabled")
                self.shadow_status_indicator.config(text="🌑 OFF", fg="#888888")
                self.shadow_improvement_label.config(text="Shadow: OFF", fg="#888888")
                self.status_text.set("⚙️ Shadow removal disabled - Standard mode")
        except Exception as e:
            print(f"Shadow removal toggle error: {e}")
    
    def load_model(self):
        """Load YOLO model with improved feedback"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select YOLO Model File",
                filetypes=[("PyTorch models", "*.pt"), ("ONNX models", "*.onnx"), ("All files", "*.*")]
            )
            
            if file_path:
                self.model_path.set(file_path)
                try:
                    self.model = load_model(file_path)
                    model_name = os.path.basename(file_path)
                    self.status_text.set(f"✅ Model loaded: {model_name}")
                    messagebox.showinfo("Success", f"YOLO model '{model_name}' loaded successfully!")
                except Exception as e:
                    self.status_text.set(f"❌ Model failed: {str(e)}")
                    messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
                    
        except Exception as e:
            print(f"Model loading error: {e}")
    
    def load_video(self):
        """Load video for processing"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"), ("All files", "*.*")]
            )
            
            if file_path:
                self.media_path.set(file_path)
                self.stop_processing()
                video_name = os.path.basename(file_path)
                self.status_text.set(f"📁 Video loaded: {video_name}")
                
        except Exception as e:
            print(f"Video loading error: {e}")
    
    def process_single_frame(self):
        """Process a single frame for testing"""
        try:
            if not self.model:
                messagebox.showwarning("Warning", "Please load a YOLO model first!")
                return
                
            if not self.media_path.get():
                messagebox.showwarning("Warning", "Please load a video first!")
                return
            
            # Capture single frame
            cap = cv2.VideoCapture(self.media_path.get())
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Process single frame
                shadow_removal = self.shadow_removal_enabled.get()
                annotated_frame, vehicle_counts, densities, road_boundaries = process_video_frame(
                    self.model, frame, conf=self.confidence.get(), 
                    road_tracker=self.road_tracker, shadow_removal=shadow_removal
                )
                
                # Display result
                self.current_frame = annotated_frame
                self.current_road_boundaries = road_boundaries
                self.detected_roads = len(road_boundaries)
                
                self.status_text.set("✅ Single frame processed successfully")
            else:
                self.status_text.set("❌ Failed to read frame from video")
                
        except Exception as e:
            print(f"Single frame processing error: {e}")
            self.status_text.set(f"❌ Frame processing error: {str(e)}")
    
    def display_frame_on_canvas(self, frame):
        try:
            canvas_width = self.media_canvas.winfo_width()
            canvas_height = self.media_canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:
                # Resize frame to fit canvas
                h, w = frame.shape[:2]
                scale = min(canvas_width/w, canvas_height/h)
                new_w, new_h = int(w*scale), int(h*scale)
                
                frame_resized = cv2.resize(frame, (new_w, new_h))
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage
                pil_image = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Display on canvas
                self.media_canvas.delete("all")
                self.media_canvas.create_image(canvas_width//2, canvas_height//2,
                                             image=photo, anchor="center")
                self.media_canvas.image = photo
                
        except Exception as e:
            print(f"Frame display error: {e}")
    
    def start_processing(self):
        try:
            if not self.model:
                messagebox.showwarning("Warning", "Please load a YOLO model first!")
                return
                
            if not self.media_path.get():
                messagebox.showwarning("Warning", "Please load a video first!")
                return
            
            # Reset enhanced road tracker
            self.road_tracker = EnhancedRoadTracker(roads=2, base_cycle=120, min_green=8, max_green=50)
            
            # Update status based on shadow removal setting
            if self.shadow_removal_enabled.get():
                self.status_text.set("PROCESSING with shadow removal...")
            else:
                self.status_text.set("PROCESSING standard mode...")
            
            if self.media_path.get():
                self.start_video_processing()
                
        except Exception as e:
            print(f"Start processing error: {e}")
    
    def start_video_processing(self):
        if not self.is_video_playing:
            self.is_video_playing = True
            self.processing_thread = threading.Thread(target=self.video_processing_loop, daemon=True)
            self.processing_thread.start()
    
    def video_processing_loop(self):
        try:
            cap = cv2.VideoCapture(self.media_path.get())
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            frame_count = 0
            
            while self.is_video_playing and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % self.frame_skip != 0:
                    continue
                
                try:
                    # Resize for faster processing
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * self.resize_factor), int(w * self.resize_factor)
                    frame_resized = cv2.resize(frame, (new_w, new_h))
                    
                    # CLEAN frame processing (no text overlays)
                    shadow_removal = self.shadow_removal_enabled.get()
                    annotated_frame, vehicle_counts, densities, road_boundaries = process_video_frame(
                        self.model, frame_resized, conf=self.confidence.get(), 
                        road_tracker=self.road_tracker, shadow_removal=shadow_removal
                    )
                    
                    # Update road boundaries and detected count
                    self.current_road_boundaries = road_boundaries
                    self.detected_roads = len(road_boundaries)
                    
                    # Resize back for display
                    annotated_frame = cv2.resize(annotated_frame, (w, h))
                    
                    # Store for UI update
                    self.current_frame = annotated_frame
                    
                    # Update timing
                    self.last_frame_time = time.time()
                    
                    # Processing delay
                    time.sleep(0.05)  # Maintain reasonable processing speed
                    
                except Exception as e:
                    print(f"Frame processing error: {e}")
                    continue
            
            cap.release()
            
        except Exception as e:
            self.status_text.set(f"❌ Processing error: {str(e)}")
            print(f"Video processing loop error: {e}")
            self.is_video_playing = False
    
    def stop_processing(self):
        try:
            self.is_video_playing = False
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            
            self.current_frame = None
            self.media_canvas.delete("all")
            self.status_text.set("Processing stopped")
            
        except Exception as e:
            print(f"Stop processing error: {e}")
    
    def update_conf_label(self, *args):
        """Update confidence label"""
        try:
            self.conf_label.config(text=f"{self.confidence.get():.2f}")
        except Exception as e:
            print(f"Confidence label update error: {e}")
    
    def start_ui_updates(self):
        self.update_ui()
    
    def update_ui(self):
        try:
            # Update current time
            current_time = time.strftime("%H:%M:%S")
            if hasattr(self, 'time_label'):
                self.time_label.config(text=current_time)
            
            # Update road count display
            if hasattr(self, 'road_count_label'):
                self.road_count_label.config(text=f"Roads: {self.detected_roads}")
            
            # Update video frame display
            if self.current_frame is not None:
                self.display_frame_on_canvas(self.current_frame)
            
            # Update analytics dashboard
            if self.road_tracker and hasattr(self.road_tracker, 'get_enhanced_analytics'):
                self.update_analytics_dashboard()
            
        except Exception as e:
            print(f"UI update error: {e}")
        
        # Schedule next update
        try:
            if self.main_window.winfo_exists() or self.analytics_window.winfo_exists():
                self.main_window.after(100, self.update_ui)
        except:
            pass
    
    def update_analytics_dashboard(self):
        try:
            if not self.road_tracker:
                return
            
            # Get enhanced analytics data
            analytics = self.road_tracker.get_enhanced_analytics()
            self.enhanced_analytics_data = analytics
            
            # Update priority road display
            priority_road_idx = analytics['priority_road']
            if self.current_road_boundaries and priority_road_idx < len(self.current_road_boundaries):
                road_name = self.current_road_boundaries[priority_road_idx].get('name', f'Road {priority_road_idx + 1}')
                green_time = analytics['dynamic_green_times'][priority_road_idx] if priority_road_idx < len(analytics['dynamic_green_times']) else 0
                
                if hasattr(self, 'priority_road_label'):
                    self.priority_road_label.config(text=f"{road_name} ({green_time}s green)")
            
            # Update congested roads
            congested = [i for i, d in enumerate(analytics['enhanced_densities']) if d > 50]
            if congested and hasattr(self, 'congested_roads_label'):
                congested_names = []
                for i in congested:
                    if self.current_road_boundaries and i < len(self.current_road_boundaries):
                        congested_names.append(self.current_road_boundaries[i].get('name', f'Road {i + 1}'))
                congested_text = ", ".join(congested_names) if congested_names else "None detected"
                self.congested_roads_label.config(text=congested_text)
            
            # Update DUAL tables
            self.update_traffic_analytics_table(analytics)
            self.update_ml_metrics_table(analytics)
            
            # Update performance metrics (NO FPS)
            if hasattr(self, 'total_vehicles_label'):
                self.total_vehicles_label.config(text=f"Total Vehicles: {analytics['total_vehicles']}")
            
            if hasattr(self, 'avg_accuracy_label'):
                avg_accuracy = np.mean(analytics['accuracy_scores']) if analytics['accuracy_scores'] else 0
                
                # Show shadow removal benefit
                if analytics['shadow_removal_stats']['enabled']:
                    improvement = analytics['shadow_removal_stats']['accuracy_improvement']
                    self.avg_accuracy_label.config(text=f"Avg Accuracy: {avg_accuracy:.1f}% (+{improvement:.1f}% shadow)")
                else:
                    self.avg_accuracy_label.config(text=f"Avg Accuracy: {avg_accuracy:.1f}%")
            
        except Exception as e:
            print(f"Analytics dashboard update error: {e}")
    
    def update_traffic_analytics_table(self, analytics):
        """Update Traffic Analytics table"""
        try:
            if not hasattr(self, 'traffic_tree'):
                return
            
            # Clear existing items
            for item in self.traffic_tree.get_children():
                self.traffic_tree.delete(item)
            
            # Add traffic analytics data
            for i in range(self.road_tracker.roads):
                # Get road name
                road_name = f'R{i+1}'
                if self.current_road_boundaries and i < len(self.current_road_boundaries):
                    full_name = self.current_road_boundaries[i].get('name', f'Road {i+1}')
                    road_name = full_name[:8] if len(full_name) <= 8 else full_name[:6] + ".."
                
                # Traffic data
                vehicles = analytics['vehicles'][i] if i < len(analytics['vehicles']) else 0
                density = f"{analytics['enhanced_densities'][i]:.1f}" if i < len(analytics['enhanced_densities']) else "0.0"
                priority = f"{analytics['enhanced_priorities'][i]:.3f}" if i < len(analytics['enhanced_priorities']) else "0.000"
                green_time = f"{analytics['dynamic_green_times'][i]}s" if i < len(analytics['dynamic_green_times']) else "0s"
                wait_time = f"{analytics['dynamic_wait_times'][i]:.1f}s" if i < len(analytics['dynamic_wait_times']) else "0.0s"
                accuracy = f"{analytics['accuracy_scores'][i]:.1f}" if i < len(analytics['accuracy_scores']) else "0.0"
                
                # Insert row
                item = self.traffic_tree.insert("", "end", values=(
                    road_name, vehicles, density, priority, green_time, wait_time, accuracy
                ))
                
                # Highlight priority road
                if i == analytics['priority_road']:
                    self.traffic_tree.item(item, tags=("priority",))
                elif analytics['enhanced_densities'][i] > 70:
                    self.traffic_tree.item(item, tags=("congested",))
            
            # Configure tags
            self.traffic_tree.tag_configure("priority", background="#ccffcc", foreground="#006600")
            self.traffic_tree.tag_configure("congested", background="#ffeecc", foreground="#cc6600")
            
        except Exception as e:
            print(f"Traffic table update error: {e}")
    
    def update_ml_metrics_table(self, analytics):
        """Update ML Metrics table - NEW!"""
        try:
            if not hasattr(self, 'ml_tree'):
                return
            
            # Clear existing items
            for item in self.ml_tree.get_children():
                self.ml_tree.delete(item)
            
            # Add ML metrics data
            for i in range(self.road_tracker.roads):
                # Get road name
                road_name = f'R{i+1}'
                if self.current_road_boundaries and i < len(self.current_road_boundaries):
                    full_name = self.current_road_boundaries[i].get('name', f'Road {i+1}')
                    road_name = full_name[:8] if len(full_name) <= 8 else full_name[:6] + ".."
                
                # ML Metrics data
                precision = f"{analytics['precision_scores'][i]:.3f}" if i < len(analytics['precision_scores']) else "0.000"
                recall = f"{analytics['recall_scores'][i]:.3f}" if i < len(analytics['recall_scores']) else "0.000"
                f1_score = f"{analytics['f1_scores'][i]:.3f}" if i < len(analytics['f1_scores']) else "0.000"
                true_pos = analytics['true_positives'][i] if i < len(analytics['true_positives']) else 0
                false_pos = analytics['false_positives'][i] if i < len(analytics['false_positives']) else 0
    
                
                # Insert row
                item = self.ml_tree.insert("", "end", values=(
                    road_name, precision, recall, f1_score, true_pos, false_pos
                ))
                
                # Highlight high-performance roads
                if analytics['f1_scores'][i] > 0.8:
                    self.ml_tree.item(item, tags=("high_performance",))
                elif analytics['f1_scores'][i] < 0.5:
                    self.ml_tree.item(item, tags=("low_performance",))
            
            # Configure ML metrics tags
            self.ml_tree.tag_configure("high_performance", background="#ccffcc", foreground="#006600")
            self.ml_tree.tag_configure("low_performance", background="#ffcccc", foreground="#cc0000")
            
        except Exception as e:
            print(f"ML metrics table update error: {e}")
    
    def on_main_window_close(self):
        """Handle main window close"""
        self.stop_processing()
        self.main_window.destroy()
        if hasattr(self, 'analytics_window'):
            self.analytics_window.destroy()
    
    def on_analytics_window_close(self):
        """Handle analytics window close"""
        self.analytics_window.destroy()
        if hasattr(self, 'main_window'):
            self.main_window.destroy()

def main():
    """Main function for CLEAN application with DUAL analytics tables"""
    try:
        app = TrafficOptimizationApp()
        app.main_window.mainloop()
    except Exception as e:
        print(f"Application error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()