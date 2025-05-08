#!/usr/bin/env python3
"""
Simple Face Recognition Application using Tkinter
Features:
- Start/stop face recognition
- Choose between YOLOv5 and MTCNN detectors
- Adjust confidence thresholds
"""

import os
import cv2
import tkinter as tk
from tkinter import ttk, Scale, HORIZONTAL
import threading
import time
import numpy as np
from PIL import Image, ImageTk

from face_detection import FaceDetector
from face_recognition import FaceRecognizer

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1000x700")
        self.root.resizable(True, True)
        
        # Initialize variables
        self.is_running = False
        self.thread = None
        self.cap = None
        
        # Default settings
        self.detector_type = tk.StringVar(value="yolov5")
        self.detection_confidence = tk.DoubleVar(value=0.7)
        self.recognition_threshold = tk.DoubleVar(value=0.7)
        self.unknown_threshold = tk.DoubleVar(value=0.5)
        self.min_confidence_margin = tk.DoubleVar(value=0.2)
        self.display_fps = tk.BooleanVar(value=True)
        self.display_confidence = tk.BooleanVar(value=True)
        self.display_margin = tk.BooleanVar(value=True)
        
        # Create UI
        self.create_ui()
        
        # Initialize detector and recognizer
        self.initialize_models()
        
    def create_ui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create left panel for controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Detector selection
        ttk.Label(control_frame, text="Face Detector:").grid(row=0, column=0, sticky=tk.W, pady=5)
        detector_combo = ttk.Combobox(control_frame, textvariable=self.detector_type, 
                                     values=["mtcnn", "yolov5"], state="readonly")
        detector_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        detector_combo.bind("<<ComboboxSelected>>", self.on_detector_change)
        
        # Detection confidence
        ttk.Label(control_frame, text="Detection Confidence:").grid(row=1, column=0, sticky=tk.W, pady=5)
        detection_scale = Scale(control_frame, variable=self.detection_confidence, 
                               from_=0.1, to=1.0, resolution=0.05, orient=HORIZONTAL, length=150)
        detection_scale.grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # Recognition threshold
        ttk.Label(control_frame, text="Recognition Threshold:").grid(row=2, column=0, sticky=tk.W, pady=5)
        recognition_scale = Scale(control_frame, variable=self.recognition_threshold, 
                                 from_=0.1, to=1.0, resolution=0.05, orient=HORIZONTAL, length=150)
        recognition_scale.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Unknown threshold
        ttk.Label(control_frame, text="Unknown Threshold:").grid(row=3, column=0, sticky=tk.W, pady=5)
        unknown_scale = Scale(control_frame, variable=self.unknown_threshold, 
                             from_=0.1, to=1.0, resolution=0.05, orient=HORIZONTAL, length=150)
        unknown_scale.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Confidence margin
        ttk.Label(control_frame, text="Confidence Margin:").grid(row=4, column=0, sticky=tk.W, pady=5)
        margin_scale = Scale(control_frame, variable=self.min_confidence_margin, 
                            from_=0.0, to=0.5, resolution=0.05, orient=HORIZONTAL, length=150)
        margin_scale.grid(row=4, column=1, sticky=tk.W, pady=5)
        
        # Display options
        ttk.Label(control_frame, text="Display Options:").grid(row=5, column=0, sticky=tk.W, pady=5)
        display_frame = ttk.Frame(control_frame)
        display_frame.grid(row=5, column=1, sticky=tk.W, pady=5)
        
        ttk.Checkbutton(display_frame, text="FPS", variable=self.display_fps).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Confidence", variable=self.display_confidence).pack(anchor=tk.W)
        ttk.Checkbutton(display_frame, text="Margin", variable=self.display_margin).pack(anchor=tk.W)
        
        # Start/Stop button
        self.start_button = ttk.Button(control_frame, text="Start Recognition", command=self.toggle_recognition)
        self.start_button.grid(row=6, column=0, columnspan=2, pady=20)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Status: Ready")
        self.status_label.grid(row=7, column=0, columnspan=2, pady=5)
        
        # Create right panel for video display
        video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding=10)
        video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Video canvas
        self.canvas = tk.Canvas(video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
    def initialize_models(self):
        try:
            self.status_label.config(text="Status: Loading models...")
            self.root.update()
            
            # Initialize face detector
            self.face_detector = FaceDetector(
                detector_type=self.detector_type.get(),
                confidence_threshold=self.detection_confidence.get()
            )
            
            # Initialize face recognizer
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
            facenet_model_path = os.path.join(model_dir, 'facenet_keras.h5')
            classifier_path = os.path.join(model_dir, 'face_classifier_model.h5')
            
            self.face_recognizer = FaceRecognizer(
                model_path=facenet_model_path,
                classifier_path=classifier_path
            )
            
            self.status_label.config(text="Status: Models loaded successfully")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
    
    def on_detector_change(self, event=None):
        if not self.is_running:
            try:
                self.status_label.config(text=f"Status: Switching to {self.detector_type.get()} detector...")
                self.root.update()
                
                # Reinitialize face detector
                self.face_detector = FaceDetector(
                    detector_type=self.detector_type.get(),
                    confidence_threshold=self.detection_confidence.get()
                )
                
                self.status_label.config(text=f"Status: Using {self.detector_type.get()} detector")
            except Exception as e:
                self.status_label.config(text=f"Error: {str(e)}")
    
    def toggle_recognition(self):
        if self.is_running:
            # Stop recognition
            self.is_running = False
            self.start_button.config(text="Start Recognition")
            self.status_label.config(text="Status: Stopped")
        else:
            # Start recognition
            self.is_running = True
            self.start_button.config(text="Stop Recognition")
            self.status_label.config(text="Status: Running")
            
            # Start recognition in a separate thread
            self.thread = threading.Thread(target=self.recognition_loop)
            self.thread.daemon = True
            self.thread.start()
    
    def recognition_loop(self):
        # Open camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Variables for FPS calculation
        frame_count = 0
        fps = 0
        start_time = time.time()
        
        while self.is_running:
            # Read frame
            ret, frame = self.cap.read()
            
            if not ret:
                self.status_label.config(text="Error: Could not read frame from camera")
                break
            
            # Process frame
            processed_frame = self.process_frame(frame, fps)
            
            # Convert to Tkinter format
            img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.canvas.config(width=img.width, height=img.height)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk  # Keep a reference to prevent garbage collection
            
            # Calculate FPS
            frame_count += 1
            elapsed_time = time.time() - start_time
            
            if elapsed_time >= 1.0:
                fps = frame_count / elapsed_time
                frame_count = 0
                start_time = time.time()
        
        # Release resources
        if self.cap is not None:
            self.cap.release()
    
    def process_frame(self, frame, fps):
        # Detect faces
        faces = self.face_detector.detect_faces(frame)
        
        # Process each detected face
        for face_info in faces:
            # Extract face coordinates
            x, y, width, height = face_info['box']
            
            # Extract face
            face_img = self.face_detector.extract_face(frame, face_info)
            
            # Recognize face
            result = self.face_recognizer.recognize(
                face_img, 
                threshold=self.recognition_threshold.get(),
                unknown_threshold=self.unknown_threshold.get(),
                min_confidence_margin=self.min_confidence_margin.get()
            )
            
            # Draw bounding box
            color = (0, 255, 0) if result['recognized'] else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+width, y+height), color, 2)
            
            # Display name and confidence
            name = result['name']
            confidence = result['confidence']
            margin = result['confidence_margin']
            
            text = name
            if self.display_confidence.get() and self.display_margin.get():
                text = f"{name} ({confidence:.2f}, margin: {margin:.2f})"
            elif self.display_confidence.get():
                text = f"{name} ({confidence:.2f})"
            elif self.display_margin.get():
                text = f"{name} (margin: {margin:.2f})"
            
            cv2.putText(frame, text, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display FPS
        if self.display_fps.get():
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def on_closing(self):
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
